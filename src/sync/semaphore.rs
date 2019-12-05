use crate::core::LogicalDevice;
use ash::{version::DeviceV1_0, vk};
use std::sync::Arc;

// TODO: Add semaphore snafu errors
// TODO: Add fence abstraction

pub struct Semaphore {
    semaphore: vk::Semaphore,
    logical_device: Arc<LogicalDevice>,
}

impl Semaphore {
    pub fn new(logical_device: Arc<LogicalDevice>) -> Self {
        let semaphore_info = vk::SemaphoreCreateInfo::builder().build();
        let semaphore = unsafe {
            logical_device
                .logical_device()
                .create_semaphore(&semaphore_info, None)
                .unwrap()
        };
        Semaphore {
            semaphore,
            logical_device,
        }
    }

    pub fn semaphore(&self) -> &vk::Semaphore {
        &self.semaphore
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        unsafe {
            self.logical_device
                .logical_device()
                .destroy_semaphore(self.semaphore, None)
        }
    }
}
