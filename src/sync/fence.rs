use crate::VulkanContext;
use ash::{version::DeviceV1_0, vk};
use std::sync::Arc;

// TODO: Add semaphore snafu errors

pub struct Fence {
    fence: vk::Fence,
    context: Arc<VulkanContext>,
}

impl Fence {
    pub fn new(context: Arc<VulkanContext>, flags: vk::FenceCreateFlags) -> Self {
        let fence_info = vk::FenceCreateInfo::builder().flags(flags).build();
        let fence = unsafe {
            context
                .logical_device()
                .create_fence(&fence_info, None)
                .unwrap()
        };

        Fence { fence, context }
    }

    pub fn fence(&self) -> &vk::Fence {
        &self.fence
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        unsafe {
            self.context
                .logical_device()
                .destroy_fence(self.fence, None)
        }
    }
}
