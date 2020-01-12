// TODO: Make a type alias for the current device version (DeviceV1_0)
use crate::core::VulkanContext;
use ash::{version::DeviceV1_0, vk};
use std::sync::Arc;

// TODO: Add snafu errors

pub struct PipelineLayout {
    layout: vk::PipelineLayout,
    context: Arc<VulkanContext>,
}

impl PipelineLayout {
    pub fn new(context: Arc<VulkanContext>, create_info: vk::PipelineLayoutCreateInfo) -> Self {
        let layout = unsafe {
            context
                .logical_device()
                .logical_device()
                .create_pipeline_layout(&create_info, None)
                .expect("Failed to create pipeline layout!")
        };

        PipelineLayout { layout, context }
    }

    pub fn layout(&self) -> vk::PipelineLayout {
        self.layout
    }
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        unsafe {
            self.context
                .logical_device()
                .logical_device()
                .destroy_pipeline_layout(self.layout, None);
        }
    }
}
