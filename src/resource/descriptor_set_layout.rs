// TODO: Make a type alias for the current device version (DeviceV1_0)
use crate::core::Instance;
use ash::{version::DeviceV1_0, vk};
use std::sync::Arc;

// TODO: Add snafu errors

pub struct DescriptorSetLayout {
    layout: vk::DescriptorSetLayout,
    instance: Arc<Instance>,
}

impl DescriptorSetLayout {
    pub fn new(instance: Arc<Instance>, bindings: &[vk::DescriptorSetLayoutBinding]) -> Self {
        let layout_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(&bindings)
            .build();
        let layout = unsafe {
            instance
                .logical_device()
                .logical_device()
                .create_descriptor_set_layout(&layout_info, None)
                .unwrap()
        };

        DescriptorSetLayout { layout, instance }
    }

    pub fn layout(&self) -> vk::DescriptorSetLayout {
        self.layout
    }
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        unsafe {
            self.instance
                .logical_device()
                .logical_device()
                .destroy_descriptor_set_layout(self.layout, None);
        }
    }
}
