// TODO: Make a type alias for the current device version (DeviceV1_0)
use crate::VulkanContext;
use ash::{version::DeviceV1_0, vk};
use std::sync::Arc;

// TODO: Add snafu errors

pub struct CommandPool {
    pool: vk::CommandPool,
    context: Arc<VulkanContext>,
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

        CommandPool { pool, context }
    }

    pub fn pool(&self) -> vk::CommandPool {
        self.pool
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        unsafe {
            self.context
                .logical_device()
                .destroy_command_pool(self.pool, None);
        }
    }
}
