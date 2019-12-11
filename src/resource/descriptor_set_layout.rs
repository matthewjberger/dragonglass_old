// TODO: Make a type alias for the current device version (DeviceV1_0)
use crate::core::VulkanContext;
use ash::{version::DeviceV1_0, vk};
use std::sync::Arc;

// TODO: Add snafu errors

pub struct DescriptorSetLayout {
    layout: vk::DescriptorSetLayout,
    context: Arc<VulkanContext>,
}

impl DescriptorSetLayout {
    pub fn new(context: Arc<VulkanContext>, bindings: &[vk::DescriptorSetLayoutBinding]) -> Self {
        let layout_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(&bindings)
            .build();
        let layout = unsafe {
            context
                .logical_device()
                .logical_device()
                .create_descriptor_set_layout(&layout_info, None)
                .unwrap()
        };

        DescriptorSetLayout { layout, context }
    }

    pub fn layout(&self) -> vk::DescriptorSetLayout {
        self.layout
    }
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        unsafe {
            self.context
                .logical_device()
                .logical_device()
                .destroy_descriptor_set_layout(self.layout, None);
        }
    }
}
