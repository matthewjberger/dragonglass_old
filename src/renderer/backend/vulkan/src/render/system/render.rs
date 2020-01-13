use crate::{
    render::{
        component::TransformComponent, renderer::VulkanGltfAsset, system::UniformBufferObject,
        Renderer,
    },
    sync::{SynchronizationSet, SynchronizationSetConstants},
};
use ash::vk;
use nalgebra_glm as glm;
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

        let camera_position = glm::vec3(0.5, 0.0, 0.5);
        let view = glm::look_at(
            &camera_position,
            &glm::vec3(0.0, 0.0, 0.0),
            &glm::vec3(0.0, 1.0, 0.0),
        );

        for transform in (&transform).join() {
            // TODO: Keep track of the global transform using the gltf document
            // and render meshes at the correct transform
            // TODO: Go through all assets
            let asset_transform = transform.translate * transform.rotate * transform.scale;
            let asset_index = 0;
            let vulkan_gltf_asset = &renderer.assets[asset_index];
            for scene in vulkan_gltf_asset.gltf.scenes() {
                for node in scene.nodes() {
                    Self::visit_node(
                        &node,
                        glm::Mat4::identity(),
                        &vulkan_gltf_asset,
                        asset_transform,
                        image_index,
                        view,
                        projection,
                        camera_position,
                    );
                }
            }
        }
    }

    pub fn visit_node(
        node: &gltf::Node,
        parent_global_transform: glm::Mat4,
        vulkan_gltf_asset: &VulkanGltfAsset,
        asset_transform: glm::Mat4,
        image_index: usize,
        view: glm::Mat4,
        projection: glm::Mat4,
        camera_position: glm::Vec3,
    ) {
        // TODO: Check that the global transform is correct here
        let transform: Vec<f32> = node
            .transform()
            .matrix()
            .iter()
            .flat_map(|array| array.iter())
            .cloned()
            .collect();
        let local_transform = glm::make_mat4(&transform.as_slice());
        let global_transform = parent_global_transform * local_transform;

        if let Some(mesh) = node.mesh() {
            let index = mesh.index();
            let vulkan_mesh = vulkan_gltf_asset
                .meshes
                .iter()
                .find(|mesh| mesh.index == index)
                .expect("Could not find corresponding mesh!");

            let ubo = UniformBufferObject {
                model: asset_transform * global_transform,
                view,
                projection,
                camera_position,
                shininess: 32.0,
            };
            let ubos = [ubo];
            let buffer = &vulkan_mesh.uniform_buffers[image_index];
            buffer.upload_to_buffer(&ubos, 0);
        }

        for child_node in node.children() {
            Self::visit_node(
                &child_node,
                global_transform,
                vulkan_gltf_asset,
                asset_transform,
                image_index,
                view,
                projection,
                camera_position,
            )
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
