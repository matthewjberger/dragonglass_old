// TODO: Make a type alias for the current device version (DeviceV1_0)
use crate::{sync::CurrentFrameSynchronization, VulkanContext};
use ash::{version::DeviceV1_0, vk};
use std::sync::Arc;

// TODO: Add snafu errors

pub struct CommandPool {
    pool: vk::CommandPool,
    context: Arc<VulkanContext>,
    command_buffers: Vec<vk::CommandBuffer>,
}

impl CommandPool {
    pub fn new(context: Arc<VulkanContext>, flags: vk::CommandPoolCreateFlags) -> Self {
        let command_pool_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(context.graphics_queue_family_index())
            .flags(flags)
            .build();

        let pool = unsafe {
            context
                .logical_device()
                .logical_device()
                .create_command_pool(&command_pool_info, None)
                .unwrap()
        };

        CommandPool {
            pool,
            context,
            command_buffers: Vec::new(),
        }
    }

    pub fn pool(&self) -> vk::CommandPool {
        self.pool
    }

    pub fn command_buffers(&self) -> &[vk::CommandBuffer] {
        &self.command_buffers
    }

    pub fn allocate_command_buffers(&mut self, size: vk::DeviceSize) {
        let allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(self.pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(size as _)
            .build();

        self.command_buffers = unsafe {
            self.context
                .logical_device()
                .logical_device()
                .allocate_command_buffers(&allocate_info)
                .unwrap()
        };
    }

    pub fn clear_command_buffers(&mut self) {
        if !self.command_buffers.is_empty() {
            unsafe {
                self.context
                    .logical_device()
                    .logical_device()
                    .free_command_buffers(self.pool, &self.command_buffers);
            }
        }
        self.command_buffers.clear();
    }

    // TODO: refactor this to use less parameters
    pub fn submit_command_buffer(
        &self,
        index: usize,
        queue: vk::Queue,
        wait_stages: &[vk::PipelineStageFlags],
        current_frame_synchronization: &CurrentFrameSynchronization,
    ) {
        let image_available_semaphores = [current_frame_synchronization.image_available()];
        let render_finished_semaphores = [current_frame_synchronization.render_finished()];
        // TODO: Add error handling, index may be invalid
        let command_buffers_to_use = [self.command_buffers()[index]];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(&image_available_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers_to_use)
            .signal_semaphores(&render_finished_semaphores)
            .build();
        let submit_info_arr = [submit_info];
        unsafe {
            self.context
                .logical_device()
                .logical_device()
                .queue_submit(
                    queue,
                    &submit_info_arr,
                    current_frame_synchronization.in_flight(),
                )
                .unwrap()
        }
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        self.clear_command_buffers();
        unsafe {
            self.context
                .logical_device()
                .logical_device()
                .destroy_command_pool(self.pool, None);
        }
    }
}
