use crate::VulkanContext;
use ash::{version::DeviceV1_0, vk};
use std::sync::Arc;

// TODO: Add semaphore snafu errors
// TODO: Add fence abstraction

pub struct Semaphore {
    semaphore: vk::Semaphore,
    context: Arc<VulkanContext>,
}

impl Semaphore {
    pub fn new(context: Arc<VulkanContext>) -> Self {
        let semaphore_info = vk::SemaphoreCreateInfo::builder().build();
        let semaphore = unsafe {
            context
                .logical_device()
                .create_semaphore(&semaphore_info, None)
                .unwrap()
        };
        Semaphore { semaphore, context }
    }

    pub fn semaphore(&self) -> &vk::Semaphore {
        &self.semaphore
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        unsafe {
            self.context
                .logical_device()
                .destroy_semaphore(self.semaphore, None)
        }
    }
}
