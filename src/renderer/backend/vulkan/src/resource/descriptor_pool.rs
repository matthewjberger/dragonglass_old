// TODO: Make a type alias for the current device version (DeviceV1_0)
use crate::core::VulkanContext;
use ash::{version::DeviceV1_0, vk};
use std::sync::Arc;

// TODO: Add snafu errors

pub struct DescriptorPool {
    pool: vk::DescriptorPool,
    context: Arc<VulkanContext>,
}

impl DescriptorPool {
    pub fn new(context: Arc<VulkanContext>, pool_info: vk::DescriptorPoolCreateInfo) -> Self {
        let pool = unsafe {
            context
                .logical_device()
                .logical_device()
                .create_descriptor_pool(&pool_info, None)
                .expect("Failed to create descriptor pool!")
        };

        DescriptorPool { pool, context }
    }

    pub fn allocate_descriptor_sets(
        &self,
        layout: vk::DescriptorSetLayout,
        number_of_sets: u32,
    ) -> Vec<vk::DescriptorSet> {
        let layouts = (0..number_of_sets).map(|_| layout).collect::<Vec<_>>();
        let allocation_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.pool)
            .set_layouts(&layouts)
            .build();
        unsafe {
            self.context
                .logical_device()
                .logical_device()
                .allocate_descriptor_sets(&allocation_info)
                .expect("Failed to allocate descriptor sets!")
        }
    }
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        unsafe {
            self.context
                .logical_device()
                .logical_device()
                .destroy_descriptor_pool(self.pool, None);
        }
    }
}
