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
    components::{AssetName, Transform},
    input::Input,
    AnimationState, AppState,
};
use legion::prelude::*;
use nalgebra_glm as glm;
use winit::event::VirtualKeyCode;

pub fn prepare_renderer_system() -> Box<dyn Schedulable> {
    SystemBuilder::new("prepare_renderer")
        .write_resource::<Renderer>()
        .with_query(<Read<AssetName>>::query())
        .build(|_, mut world, renderer, query| {
            let asset_names = query
                .iter(&mut world)
                .map(|asset_name| asset_name.0.to_string())
                .collect::<Vec<_>>();
            renderer.load_assets(&asset_names);
            renderer.allocate_command_buffers();
            renderer.record_command_buffers();
        })
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
        .with_query(<Read<Transform>>::query())
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

                    skybox_data.uniform_buffer.upload_to_buffer(
                        &skybox_ubos,
                        0,
                        std::mem::align_of::<SkyboxUniformBufferObject>() as _,
                    );
                }

                let ubo = UniformBufferObject {
                    cameraposition: camera_state.position,
                    view: camera_state.view,
                    projection,
                };
                let ubos = [ubo];

                if let Some(pbr_data) = &renderer.pbr_pipeline_data.as_ref() {
                    pbr_data.uniform_buffer.upload_to_buffer(
                        &ubos,
                        0,
                        std::mem::align_of::<UniformBufferObject>() as _,
                    );
                }

                let mut mesh_offset = 0;
                for (index, transform) in query.iter(&mut world).enumerate() {
                    // TODO: Keep track of the global transform using the gltf document
                    // and render meshes at the correct transform
                    // TODO: Go through all assets
                    let asset_transform = transform.translate * transform.rotate * transform.scale;
                    let asset = &renderer.assets[index];
                    asset.walk(|node_index, graph| {
                        let global_transform =
                            GltfAsset::calculate_global_transform(node_index, graph);
                        if let Some(mesh) = graph[node_index].mesh.as_ref() {
                            if let Some(pbr_data) = &renderer.pbr_pipeline_data.as_ref() {
                                let dynamic_ubo = DynamicUniformBufferObject {
                                    model: asset_transform * global_transform,
                                };
                                let ubos = [dynamic_ubo];
                                let buffer = &pbr_data.dynamic_uniform_buffer;
                                let offset = (pbr_data.dynamic_alignment
                                    * (mesh_offset + mesh.mesh_id) as u64)
                                    as usize;

                                buffer.upload_to_buffer(&ubos, offset, pbr_data.dynamic_alignment);

                                // let full_dynamic_ubo_size = (asset.number_of_meshes as u64 * pbr_data.dynamic_alignment) as u64;
                                // buffer
                                //     .flush(dynamic_buffer_offset, full_dynamic_ubo_size as _)
                                //     .expect("Failed to flush buffer!");
                            }
                        }

                        // TODO: Handle skins
                    });
                    mesh_offset += asset.number_of_meshes;
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
        .with_query(<Write<AnimationState>>::query())
        .build(move |_, mut world, renderer, query| {
            let animation_states = &mut query.iter(&mut world).collect::<Vec<_>>();
            if animation_states.is_empty() {
                return;
            }
            for animation in renderer.assets[0].animations.iter_mut() {
                animation.time += 0.0005;
            }
            renderer.assets[0].animate();
        })
}
