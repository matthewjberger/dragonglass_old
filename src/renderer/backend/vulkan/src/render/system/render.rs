use crate::{
    render::{
        component::TransformComponent, system::UniformBufferObject,
        Renderer,
        pipeline_gltf::VulkanGltfAsset,
    },
    sync::{SynchronizationSet, SynchronizationSetConstants},
};
use ash::vk;
use legion::prelude::*;
use nalgebra_glm as glm;

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
            let image_index_result = renderer.swapchain.acquire_next_image(
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
                renderer.swapchain.properties().aspect_ratio(),
                90_f32.to_radians(),
                0.1_f32,
                1000_f32,
            );

            let camera_position = glm::vec3(100.0, 4.0, 100.0);
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
                let vulkan_gltf_asset = &renderer.pipeline_gltf.as_ref().unwrap().assets[asset_index];
                for (scene_index, scene) in vulkan_gltf_asset.gltf.scenes().enumerate() {
                    for node in scene.nodes() {
                        visit_node(
                            &node,
                            glm::Mat4::identity(),
                            &vulkan_gltf_asset,
                            asset_transform,
                            image_index as usize,
                            view,
                            projection,
                            scene_index,
                        );
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

            let swapchain_presentation_result = renderer.swapchain.present_rendered_image(
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

fn visit_node(
    node: &gltf::Node,
    parent_global_transform: glm::Mat4,
    vulkan_gltf_asset: &VulkanGltfAsset,
    asset_transform: glm::Mat4,
    image_index: usize,
    view: glm::Mat4,
    projection: glm::Mat4,
    scene_index: usize,
) {
    let transform: Vec<f32> = node
        .transform()
        .matrix()
        .iter()
        .flat_map(|array| array.iter())
        .cloned()
        .collect();
    let local_transform = glm::make_mat4(&transform.as_slice());
    let global_transform = parent_global_transform * local_transform;

    if node.mesh().is_some() {
        let vulkan_mesh = vulkan_gltf_asset.meshes[scene_index]
            .iter()
            .find(|mesh| mesh.node_index == node.index())
            .expect("Could not find corresponding mesh!");

        let ubo = UniformBufferObject {
            model: asset_transform * global_transform,
            view,
            projection,
        };
        let ubos = [ubo];
        let buffer = &vulkan_mesh.uniform_buffers[image_index];
        buffer.upload_to_buffer(&ubos, 0);
    }

    for child_node in node.children() {
        visit_node(
            &child_node,
            global_transform,
            vulkan_gltf_asset,
            asset_transform,
            image_index,
            view,
            projection,
            scene_index,
        )
    }
}
