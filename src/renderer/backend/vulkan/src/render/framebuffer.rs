use crate::core::VulkanContext;
use ash::{version::DeviceV1_0, vk};
use std::sync::Arc;

// TODO: Add snafu errors

pub struct Framebuffer {
    framebuffer: vk::Framebuffer,
    context: Arc<VulkanContext>,
}

impl Framebuffer {
    pub fn new(context: Arc<VulkanContext>, create_info: vk::FramebufferCreateInfo) -> Self {
        let framebuffer = unsafe {
            context
                .logical_device()
                .logical_device()
                .create_framebuffer(&create_info, None)
                .expect("Failed to create framebuffer!")
        };

        Framebuffer {
            framebuffer,
            context,
        }
    }

    pub fn framebuffer(&self) -> vk::Framebuffer {
        self.framebuffer
    }
}

impl Drop for Framebuffer {
    fn drop(&mut self) {
        unsafe {
            self.context
                .logical_device()
                .logical_device()
                .destroy_framebuffer(self.framebuffer, None);
        }
    }
}
