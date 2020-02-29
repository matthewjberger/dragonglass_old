use crate::{
    render::{
        component::TransformComponent,
        gltf::{calculate_global_transform, DynamicUniformBufferObject, UniformBufferObject},
        Renderer,
    },
    sync::{SynchronizationSet, SynchronizationSetConstants},
};
use ash::vk;
use legion::prelude::*;
use nalgebra_glm as glm;
use petgraph::{graph::NodeIndex, visit::Dfs};

pub fn render_system() -> Box<dyn Runnable> {
    SystemBuilder::new("render")
        .write_resource::<Renderer>()
        .with_query(<Read<TransformComponent>>::query())
        .build_thread_local(move |_, mut world, renderer, query| {
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

            let camera_position = glm::vec3(1.0, 0.4, 1.0);
            let view = glm::look_at(
                &camera_position,
                &glm::vec3(0.0, 0.0, 0.0),
                &glm::vec3(0.0, 1.0, 0.0),
            );

            for transform in query.iter(&mut world) {
                // TODO: Keep track of the global transform using the gltf document
                // and render meshes at the correct transform
                // TODO: Go through all assets
                let asset_transform = transform.translate * transform.rotate * transform.scale;
                let asset_index = 0;
                let vulkan_gltf_asset =
                    &renderer.pipeline_gltf.as_ref().unwrap().assets[asset_index];

                let ubo = UniformBufferObject { view, projection };
                let ubos = [ubo];
                let buffer = &vulkan_gltf_asset.uniform_buffer;
                buffer.upload_to_buffer(&ubos, 0, std::mem::align_of::<UniformBufferObject>() as _);

                // FIXME: SIZE HERE
                let full_dynamic_ubo_size =
                    (400 as u64 * vulkan_gltf_asset.dynamic_alignment) as u64;

                for scene in vulkan_gltf_asset.scenes.iter() {
                    for graph in scene.node_graphs.iter() {
                        let mut dfs = Dfs::new(&graph, NodeIndex::new(0));
                        while let Some(node_index) = dfs.next(&graph) {
                            let global_transform = calculate_global_transform(node_index, graph);
                            if let Some(mesh) = graph[node_index].mesh.as_ref() {
                                let dynamic_ubo = DynamicUniformBufferObject {
                                    model: asset_transform * global_transform,
                                };
                                let ubos = [dynamic_ubo];
                                let buffer = &vulkan_gltf_asset.dynamic_uniform_buffer;
                                let offset = (vulkan_gltf_asset.dynamic_alignment
                                    * mesh.ubo_index as u64)
                                    as usize;

                                buffer.upload_to_buffer(
                                    &ubos,
                                    offset,
                                    vulkan_gltf_asset.dynamic_alignment,
                                );
                                buffer
                                    .flush(0, full_dynamic_ubo_size as _)
                                    .expect("Failed to flush buffer!");
                            }
                        }
                    }
                }
            }

            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            renderer.command_pool.submit_command_buffer(
                image_index as usize,
                renderer.graphics_queue,
                &wait_stages,
                &current_frame_synchronization,
            );

            let swapchain_presentation_result =
                renderer.vulkan_swapchain.swapchain.present_rendered_image(
                    &current_frame_synchronization,
                    &image_indices,
                    renderer.present_queue,
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
