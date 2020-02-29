use crate::core::VulkanContext;
use ash::{version::DeviceV1_0, vk};
use std::sync::Arc;

// TODO: Add snafu errors

pub struct RenderPass {
    render_pass: vk::RenderPass,
    context: Arc<VulkanContext>,
}

impl RenderPass {
    pub fn new(context: Arc<VulkanContext>, create_info: &vk::RenderPassCreateInfo) -> Self {
        let render_pass = unsafe {
            context
                .logical_device()
                .logical_device()
                .create_render_pass(&create_info, None)
                .expect("Failed to create renderpass!")
        };

        RenderPass {
            render_pass,
            context,
        }
    }

    pub fn render_pass(&self) -> vk::RenderPass {
        self.render_pass
    }
}

impl Drop for RenderPass {
    fn drop(&mut self) {
        unsafe {
            self.context
                .logical_device()
                .logical_device()
                .destroy_render_pass(self.render_pass, None);
        }
    }
}
