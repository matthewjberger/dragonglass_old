// TODO: Make a type alias for the current device version (DeviceV1_0)
use crate::VulkanContext;
use ash::{version::DeviceV1_0, vk};
use std::sync::Arc;

// TODO: Add snafu errors

pub struct DescriptorPool {
    pool: vk::DescriptorPool,
    context: Arc<VulkanContext>,
}

impl DescriptorPool {
    pub fn new(context: Arc<VulkanContext>, size: u32) -> Self {
        let pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: size,
        };
        let pool_sizes = [pool_size];

        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes)
            .max_sets(size)
            .build();

        let pool = unsafe {
            context
                .logical_device()
                .create_descriptor_pool(&pool_info, None)
                .unwrap()
        };

        DescriptorPool { pool, context }
    }

    pub fn pool(&self) -> vk::DescriptorPool {
        self.pool
    }
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        unsafe {
            self.context
                .logical_device()
                .destroy_descriptor_pool(self.pool, None);
        }
    }
}
