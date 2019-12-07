use crate::VulkanContext;
use ash::{version::DeviceV1_0, vk};
use std::sync::Arc;

use snafu::{ResultExt, Snafu};

type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, Snafu)]
#[snafu(visibility = "pub(crate)")]
pub enum Error {
    #[snafu(display("Failed to create fence: {}", source))]
    SemaphoreCreation { source: vk::Result },
}

pub struct Semaphore {
    semaphore: vk::Semaphore,
    context: Arc<VulkanContext>,
}

impl Semaphore {
    pub fn new(context: Arc<VulkanContext>) -> Result<Self> {
        let semaphore_info = vk::SemaphoreCreateInfo::builder().build();
        let semaphore = unsafe {
            context
                .logical_device()
                .create_semaphore(&semaphore_info, None)
                .context(SemaphoreCreation)?
        };
        Ok(Semaphore { semaphore, context })
    }

    pub fn semaphore(&self) -> vk::Semaphore {
        self.semaphore
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
