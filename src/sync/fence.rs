use crate::core::Instance;
use ash::{version::DeviceV1_0, vk};
use std::sync::Arc;

use snafu::{ResultExt, Snafu};

type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, Snafu)]
#[snafu(visibility = "pub(crate)")]
pub enum Error {
    #[snafu(display("Failed to create fence: {}", source))]
    FenceCreation { source: vk::Result },
}

pub struct Fence {
    fence: vk::Fence,
    instance: Arc<Instance>,
}

impl Fence {
    pub fn new(instance: Arc<Instance>, flags: vk::FenceCreateFlags) -> Result<Self> {
        let fence_info = vk::FenceCreateInfo::builder().flags(flags).build();
        let fence = unsafe {
            instance
                .logical_device()
                .logical_device()
                .create_fence(&fence_info, None)
                .context(FenceCreation)?
        };

        Ok(Fence { fence, instance })
    }

    pub fn fence(&self) -> vk::Fence {
        self.fence
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        unsafe {
            self.instance
                .logical_device()
                .logical_device()
                .destroy_fence(self.fence, None)
        }
    }
}
