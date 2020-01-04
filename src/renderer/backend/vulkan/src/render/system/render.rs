use crate::{
    render::{component::TransformComponent, system::UniformBufferObject, Renderer},
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

        let view = glm::look_at(
            &glm::vec3(300.0, 300.0, 300.0),
            &glm::vec3(0.0, 0.0, 0.0),
            &glm::vec3(0.0, 1.0, 0.0),
        );

        for (index, transform) in (&transform).join().enumerate() {
            let ubo = UniformBufferObject {
                model: transform.translate * transform.rotate * transform.scale,
                view,
                projection,
            };

            let ubos = [ubo];
            // FIXME: UPDATE UBOS WITH INFO FROM COMPONENTS
            // let buffer = &renderer.models[index].uniform_buffers[image_index];
            // buffer.upload_to_buffer(&ubos, 0);
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
