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
};
use legion::prelude::*;
use nalgebra_glm as glm;

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

pub fn render_system() -> Box<dyn Runnable> {
    SystemBuilder::new("render")
        .write_resource::<Renderer>()
        .read_resource::<CameraState>()
        .with_query(<Read<Transform>>::query())
        .build_thread_local(move |_, mut world, (renderer, camera_state), query| {
            let context = renderer.context.clone();

            let current_frame_synchronization = renderer
                .synchronization_set
                .current_frame_synchronization(renderer.current_frame);

            context
                .logical_device()
                .wait_for_fence(&current_frame_synchronization);

            // Acquire the next image from the swapchain
            let image_index_result = renderer.vulkan_swapchain.swapchain.acquire_next_image(
                current_frame_synchronization.image_available(),
                vk::Fence::null(),
            );

            let image_index = match image_index_result {
                Ok((image_index, _)) => image_index,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    // TODO: Recreate the swapchain
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
                    .vulkan_swapchain
                    .swapchain
                    .properties()
                    .aspect_ratio(),
                90_f32.to_radians(),
                0.1_f32,
                1000_f32,
            );

            if let Some(skybox_data) = &renderer.skybox_pipeline_data.as_ref() {
                let skybox_ubo = SkyboxUniformBufferObject {
                    model: glm::translate(&glm::Mat4::identity(), &glm::vec3(0.0, 2.0, 0.0)),
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

            for transform in query.iter(&mut world) {
                // TODO: Keep track of the global transform using the gltf document
                // and render meshes at the correct transform
                // TODO: Go through all assets
                let asset_transform = transform.translate * transform.rotate * transform.scale;
                let asset_index = 0;
                let asset = &renderer.assets[asset_index];

                asset.walk(|node_index, graph| {
                    let global_transform = GltfAsset::calculate_global_transform(node_index, graph);
                    if let Some(mesh) = graph[node_index].mesh.as_ref() {
                        if let Some(pbr_data) = &renderer.pbr_pipeline_data.as_ref() {
                            pbr_data.uniform_buffer.upload_to_buffer(
                                &ubos,
                                0,
                                std::mem::align_of::<UniformBufferObject>() as _,
                            );

                            let full_dynamic_ubo_size =
                                (asset.number_of_meshes as u64 * pbr_data.dynamic_alignment) as u64;

                            let dynamic_ubo = DynamicUniformBufferObject {
                                model: asset_transform * global_transform,
                            };
                            let ubos = [dynamic_ubo];
                            let buffer = &pbr_data.dynamic_uniform_buffer;
                            let offset =
                                (pbr_data.dynamic_alignment * mesh.mesh_id as u64) as usize;

                            buffer.upload_to_buffer(&ubos, offset, pbr_data.dynamic_alignment);
                            buffer
                                .flush(0, full_dynamic_ubo_size as _)
                                .expect("Failed to flush buffer!");
                        }
                    }
                });
            }

            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            renderer.command_pool.submit_command_buffer(
                image_index as usize,
                renderer.context.graphics_queue(),
                &wait_stages,
                &current_frame_synchronization,
            );

            let swapchain_presentation_result =
                renderer.vulkan_swapchain.swapchain.present_rendered_image(
                    &current_frame_synchronization,
                    &image_indices,
                    renderer.context.present_queue(),
                );

            match swapchain_presentation_result {
                Ok(is_suboptimal) if is_suboptimal => {
                    // TODO: Recreate the swapchain
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    // TODO: Recreate the swapchain
                }
                Err(error) => panic!("Failed to present queue. Cause: {}", error),
                _ => {}
            }

            // TODO: Recreate the swapchain if resize was requested

            renderer.current_frame +=
                (1 + renderer.current_frame) % SynchronizationSet::MAX_FRAMES_IN_FLIGHT as usize;
        })
}
