// TODO: Make a type alias for the current device version (DeviceV1_0)
use crate::VulkanContext;
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
                .allocate_command_buffers(&allocate_info)
                .unwrap()
        };
    }

    pub fn clear_command_buffers(&mut self) {
        if !self.command_buffers.is_empty() {
            unsafe {
                self.context
                    .logical_device()
                    .free_command_buffers(self.pool, &self.command_buffers);
            }
        }
        self.command_buffers.clear();
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        self.clear_command_buffers();
        unsafe {
            self.context
                .logical_device()
                .destroy_command_pool(self.pool, None);
        }
    }
}
