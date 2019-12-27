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
    pub fn new(
        context: Arc<VulkanContext>,
        ubo_pool_size: u32,
        sampler_pool_size: u32,
        max_sets: u32,
    ) -> Self {
        let ubo_pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: ubo_pool_size,
        };

        let sampler_pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: sampler_pool_size,
        };

        let pool_sizes = [ubo_pool_size, sampler_pool_size];

        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes)
            .max_sets(max_sets)
            .build();

        let pool = unsafe {
            context
                .logical_device()
                .logical_device()
                .create_descriptor_pool(&pool_info, None)
                .unwrap()
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
                .unwrap()
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
