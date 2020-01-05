use crate::{
    render::{component::TransformComponent, system::UniformBufferObject, Renderer},
    sync::{SynchronizationSet, SynchronizationSetConstants},
};
use ash::vk;
use dragonglass_model_gltf::gltf::calculate_global_transform;
use nalgebra_glm as glm;
use petgraph::{prelude::*, visit::Dfs};
use specs::prelude::*;

pub struct RenderSystem;

impl RenderSystem {
    pub fn update_ubos(data: &<RenderSystem as System>::SystemData, image_index: usize) {
        let (renderer, transform) = data;

        let projection = glm::perspective_zo(
            renderer.swapchain.properties().aspect_ratio(),
            90_f32.to_radians(),
            0.1_f32,
            1000_f32,
        );

        let view = glm::look_at(
            &glm::vec3(100.0, 100.0, 100.0),
            &glm::vec3(0.0, 0.0, 0.0),
            &glm::vec3(0.0, 1.0, 0.0),
        );

        for transform in (&transform).join() {
            // TODO: Add a component when preparing the renderer
            // that tags the entity with the asset_index
            // and use it here to update the ubo
            let asset_index = 0;
            let vulkan_gltf_asset = &renderer.assets[asset_index];

            for (scene_index, scene) in vulkan_gltf_asset.asset.scenes.iter().enumerate() {
                for (graph_index, graph) in scene.node_graphs.iter().enumerate() {
                    let mut dfs = Dfs::new(&graph, NodeIndex::new(0));
                    while let Some(node_index) = dfs.next(&graph) {
                        let global_transform = calculate_global_transform(node_index, graph);
                        if graph[node_index].mesh.as_ref().is_some() {
                            let mesh = vulkan_gltf_asset
                                .meshes
                                .iter()
                                .find(|mesh|
                                      // TODO: Implement PartialEq trait for MeshLocation
                                      mesh.location.scene_index == scene_index
                                      && mesh.location.graph_index == graph_index
                                      && mesh.location.node_index == node_index)
                                .expect("Couldn't find matching mesh!");

                            let local_transformation =
                                transform.translate * transform.rotate * transform.scale;

                            let ubo = UniformBufferObject {
                                model: local_transformation * global_transform,
                                view,
                                projection,
                            };
                            let ubos = [ubo];
                            let buffer = &mesh.uniform_buffers[image_index];
                            buffer.upload_to_buffer(&ubos, 0);
                        }
                    }
                }
            }
        }
    }
}

impl<'a> System<'a> for RenderSystem {
    type SystemData = (
        WriteExpect<'a, Renderer>,
        ReadStorage<'a, TransformComponent>,
    );

    fn run(&mut self, data: Self::SystemData) {
        let current_frame_synchronization = data
            .0
            .synchronization_set
            .current_frame_synchronization(data.0.current_frame);

        data.0
            .context
            .logical_device()
            .wait_for_fence(&current_frame_synchronization);

        // Acquire the next image from the swapchain
        let image_index_result = data.0.swapchain.acquire_next_image(
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

        data.0
            .context
            .logical_device()
            .reset_fence(&current_frame_synchronization);

        Self::update_ubos(&data, image_index as usize);

        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        data.0.command_pool.submit_command_buffer(
            image_index as usize,
            data.0.graphics_queue,
            &wait_stages,
            &current_frame_synchronization,
        );

        let swapchain_presentation_result = data.0.swapchain.present_rendered_image(
            &current_frame_synchronization,
            &image_indices,
            data.0.present_queue,
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

        let (mut renderer, _) = data;
        renderer.current_frame +=
            (1 + renderer.current_frame) % SynchronizationSet::MAX_FRAMES_IN_FLIGHT as usize;
    }
}
