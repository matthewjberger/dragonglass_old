use crate::core::VulkanContext;
use ash::{version::DeviceV1_0, vk};
use std::sync::Arc;

// Add snafu errors

pub struct Sampler {
    sampler: vk::Sampler,
    context: Arc<VulkanContext>,
}

impl Sampler {
    pub fn new(context: Arc<VulkanContext>, create_info: vk::SamplerCreateInfo) -> Self {
        let sampler = unsafe {
            context
                .logical_device()
                .logical_device()
                .create_sampler(&create_info, None)
                .expect("Failed to create sampler!")
        };

        Sampler { sampler, context }
    }

    pub fn sampler(&self) -> vk::Sampler {
        self.sampler
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe {
            self.context
                .logical_device()
                .logical_device()
                .destroy_sampler(self.sampler, None)
        };
    }
}
