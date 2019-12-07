use crate::{core::SwapchainProperties, VulkanContext};
use ash::{version::DeviceV1_0, vk};
use std::sync::Arc;

// TODO: Add snafu errors

pub struct RenderPass {
    render_pass: vk::RenderPass,
    context: Arc<VulkanContext>,
}

impl RenderPass {
    pub fn new(context: Arc<VulkanContext>, swapchain_properties: &SwapchainProperties) -> Self {
        let attachment_description = vk::AttachmentDescription::builder()
            .format(swapchain_properties.format.format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .build();
        let attachment_descriptions = [attachment_description];

        let attachment_reference = vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build();
        let attachment_references = [attachment_reference];

        let subpass_description = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&attachment_references)
            .build();
        let subpass_descriptions = [subpass_description];

        let subpass_dependency = vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            )
            .build();
        let subpass_dependencies = [subpass_dependency];

        let render_pass_info = vk::RenderPassCreateInfo::builder()
            .attachments(&attachment_descriptions)
            .subpasses(&subpass_descriptions)
            .dependencies(&subpass_dependencies)
            .build();

        let render_pass = unsafe {
            context
                .logical_device()
                .logical_device()
                .create_render_pass(&render_pass_info, None)
                .unwrap()
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
