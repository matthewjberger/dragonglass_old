use crate::core::{Instance, SwapchainProperties};
use ash::{version::DeviceV1_0, vk};
use std::sync::Arc;

// TODO: Add snafu errors

pub struct Framebuffer {
    framebuffer: vk::Framebuffer,
    instance: Arc<Instance>,
}

impl Framebuffer {
    // TODO: Refactor this to use less parameters
    pub fn new(
        instance: Arc<Instance>,
        swapchain_properties: &SwapchainProperties,
        render_pass: vk::RenderPass,
        attachments: &[vk::ImageView],
    ) -> Self {
        let framebuffer_info = vk::FramebufferCreateInfo::builder()
            .render_pass(render_pass)
            .attachments(attachments)
            .width(swapchain_properties.extent.width)
            .height(swapchain_properties.extent.height)
            .layers(1)
            .build();
        let framebuffer = unsafe {
            instance
                .logical_device()
                .logical_device()
                .create_framebuffer(&framebuffer_info, None)
                .unwrap()
        };

        Framebuffer {
            framebuffer,
            instance,
        }
    }

    pub fn framebuffer(&self) -> vk::Framebuffer {
        self.framebuffer
    }
}

impl Drop for Framebuffer {
    fn drop(&mut self) {
        unsafe {
            self.instance
                .logical_device()
                .logical_device()
                .destroy_framebuffer(self.framebuffer, None);
        }
    }
}
