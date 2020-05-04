use crate::{
    model::gltf::GltfAsset,
    pipelines::{
        pbr::{DynamicUniformBufferObject, UniformBufferObject},
        skybox::UniformBufferObject as SkyboxUniformBufferObject,
    },
    render::Renderer,
    sync::{SynchronizationSet, SynchronizationSetConstants},
};
use ash::vk;
use dragonglass_core::{
    camera::CameraState,
    components::{AssetIndex, AssetName, Transform},
    input::Input,
    AnimationState, AppState, DeltaTime,
};
use legion::prelude::*;
use nalgebra_glm as glm;
use winit::event::VirtualKeyCode;

pub fn prepare_renderer(renderer: &mut Renderer, mut world: &mut World) {
    let query = Read::<AssetName>::query();

    let data = query
        .iter_entities(&mut world)
        .map(|(entity, asset_name)| (entity, asset_name.0.to_string()))
        .collect::<Vec<_>>();

    let mut asset_names = Vec::new();
    for (index, (entity, asset_name)) in data.iter().enumerate() {
        world.add_component(*entity, AssetIndex(index));
        asset_names.push(asset_name.to_string());
    }

    renderer.load_assets(&asset_names);
    renderer.allocate_command_buffers();
    renderer.record_command_buffers();
}

pub fn reload_system() -> Box<dyn Schedulable> {
    SystemBuilder::new("hot_reload")
        .write_resource::<Renderer>()
        .read_resource::<Input>()
        .build(move |_, _, (renderer, input), _| {
            if input.is_key_pressed(VirtualKeyCode::F5) && renderer.can_reload {
                renderer.can_reload = false;
                renderer.reload_pbr_pipeline();
            }

            if !input.is_key_pressed(VirtualKeyCode::F5) {
                renderer.can_reload = true;
            }
        })
}

pub fn render_system() -> Box<dyn Runnable> {
    SystemBuilder::new("render")
        .write_resource::<Renderer>()
        .read_resource::<CameraState>()
        .read_resource::<AppState>()
        .with_query(<(Read<Transform>, Read<AssetIndex>)>::query())
        .build_thread_local(
            move |_, mut world, (renderer, camera_state, app_state), query| {
                let context = renderer.context.clone();

                let current_frame_synchronization = renderer
                    .synchronization_set
                    .current_frame_synchronization(renderer.current_frame);

                context
                    .logical_device()
                    .wait_for_fence(&current_frame_synchronization);

                // Acquire the next image from the swapchain
                let image_index_result = renderer.vulkan_swapchain().swapchain.acquire_next_image(
                    current_frame_synchronization.image_available(),
                    vk::Fence::null(),
                );

                let dimensions = glm::vec2(
                    app_state.window.width as f32,
                    app_state.window.height as f32,
                );
                let image_index = match image_index_result {
                    Ok((image_index, _)) => image_index,
                    Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                        renderer.recreate_swapchain(dimensions);
                        return;
                    }
                    Err(error) => panic!("Error while acquiring next image. Cause: {}", error),
                };
                let image_indices = [image_index];

                context
                    .logical_device()
                    .reset_fence(&current_frame_synchronization);

                // Update UBOS

                let projection = glm::perspective_zo(
                    renderer
                        .vulkan_swapchain()
                        .swapchain
                        .properties()
                        .aspect_ratio(),
                    90_f32.to_radians(),
                    0.1_f32,
                    1000_f32,
                );

                if let Some(skybox_data) = &renderer.skybox_pipeline_data.as_ref() {
                    let skybox_ubo = SkyboxUniformBufferObject {
                        view: camera_state.view,
                        projection,
                    };
                    let skybox_ubos = [skybox_ubo];

                    skybox_data.uniform_buffer.upload_to_buffer(&skybox_ubos, 0);
                }

                let mut ubo = UniformBufferObject {
                    camera_position: glm::vec4(
                        camera_state.position.x,
                        camera_state.position.y,
                        camera_state.position.z,
                        1.0,
                    ),
                    view: camera_state.view,
                    projection,
                    joint_matrices: [glm::Mat4::identity(); UniformBufferObject::MAX_NUM_JOINTS],
                };

                let mut mesh_offset = 0;
                for (transform, asset_index) in query.iter(&mut world) {
                    let asset_transform = transform.translate * transform.rotate * transform.scale;
                    let asset = &renderer.assets[asset_index.0];
                    asset.walk_mut(|node_index, graph| {
                        let global_transform =
                            GltfAsset::calculate_global_transform(node_index, graph);
                        if let Some(mesh) = graph[node_index].mesh.as_ref() {
                            if let Some(pbr_data) = &renderer.pbr_pipeline_data.as_ref() {
                                let mut dynamic_ubo = DynamicUniformBufferObject {
                                    model: asset_transform * global_transform,
                                    joint_info: glm::vec4(0.0, 0.0, 0.0, 0.0),
                                };

                                if let Some(skin) = graph[node_index].skin.as_ref() {
                                    dynamic_ubo.joint_info = glm::vec4(skin.joints.len() as f32, 0.0, 0.0, 0.0);
                                    for (index, joint) in skin.joints.iter().enumerate() {
                                        if index > UniformBufferObject::MAX_NUM_JOINTS {
                                            eprintln!("Skin joint count {} is greater than the maximum joint limit of {}!", dynamic_ubo.joint_info, UniformBufferObject::MAX_NUM_JOINTS);
                                        }

                                        let joint_node_index = GltfAsset::matching_node_index(joint.target_gltf_index, &graph)
                                            .expect("Failed to find joint target node index!");

                                        let joint_global_transform =
                                            GltfAsset::calculate_global_transform(joint_node_index, &graph);

                                        let joint_matrix = glm::inverse(&global_transform)
                                            * joint_global_transform
                                            * joint.inverse_bind_matrix;

                                        ubo.joint_matrices[index] = joint_matrix;
                                    }
                                }

                                let dynamic_ubos = [dynamic_ubo];
                                let buffer = &pbr_data.dynamic_uniform_buffer;
                                let offset = (pbr_data.dynamic_alignment
                                    * (mesh_offset + mesh.mesh_id) as u64)
                                    as usize;

                                buffer.upload_to_buffer_aligned(
                                    &dynamic_ubos,
                                    offset,
                                    pbr_data.dynamic_alignment,
                                );

                                let dynamic_ubo_size = (asset.number_of_meshes as u64
                                    * pbr_data.dynamic_alignment)
                                    as u64;
                                buffer
                                    .flush(offset, dynamic_ubo_size as _)
                                    .expect("Failed to flush buffer!");
                            }
                        }
                        // TODO: Handle skins
                    });
                    mesh_offset += asset.number_of_meshes;
                }

                let ubos = [ubo];
                if let Some(pbr_data) = &renderer.pbr_pipeline_data.as_ref() {
                    pbr_data.uniform_buffer.upload_to_buffer(&ubos, 0);
                }

                let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
                renderer.command_pool.submit_command_buffer(
                    image_index as usize,
                    renderer.context.graphics_queue(),
                    &wait_stages,
                    &current_frame_synchronization,
                );

                let swapchain_presentation_result = renderer
                    .vulkan_swapchain()
                    .swapchain
                    .present_rendered_image(
                        &current_frame_synchronization,
                        &image_indices,
                        renderer.context.present_queue(),
                    );

                match swapchain_presentation_result {
                    Ok(is_suboptimal) if is_suboptimal => {
                        renderer.recreate_swapchain(dimensions);
                    }
                    Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                        renderer.recreate_swapchain(dimensions);
                    }
                    Err(error) => panic!("Failed to present queue. Cause: {}", error),
                    _ => {}
                }

                renderer.current_frame += (1 + renderer.current_frame)
                    % SynchronizationSet::MAX_FRAMES_IN_FLIGHT as usize;
            },
        )
}

// TODO: Replace this with a system that modifies an animation state component
pub fn animation_system() -> Box<dyn Schedulable> {
    SystemBuilder::new("animation_system")
        .write_resource::<Renderer>()
        .read_resource::<DeltaTime>()
        .with_query(<(Write<AnimationState>, Read<AssetIndex>)>::query())
        .build(move |_, mut world, (renderer, delta_time), query| {
            for (_, asset_index) in query.iter(&mut world) {
                // TODO: Correlate the animation to the animation state and update the animation time there
                for animation in renderer.assets[asset_index.0].animations.iter_mut() {
                    animation.time += 0.75 * delta_time.0 as f32;
                }

                // TODO: Turn animation into system
                renderer.assets[asset_index.0].animate();
            }
        })
}
