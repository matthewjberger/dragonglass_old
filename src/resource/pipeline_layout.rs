// TODO: Make a type alias for the current device version (DeviceV1_0)
use crate::core::Instance;
use ash::{version::DeviceV1_0, vk};
use std::sync::Arc;

// TODO: Add snafu errors

pub struct PipelineLayout {
    layout: vk::PipelineLayout,
    instance: Arc<Instance>,
}

impl PipelineLayout {
    pub fn new(
        instance: Arc<Instance>,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
    ) -> Self {
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&descriptor_set_layouts) // needed for uniforms in shaders
            // .push_constant_ranges()
            .build();

        let layout = unsafe {
            instance
                .logical_device()
                .logical_device()
                .create_pipeline_layout(&pipeline_layout_info, None)
                .unwrap()
        };

        PipelineLayout { layout, instance }
    }

    pub fn layout(&self) -> vk::PipelineLayout {
        self.layout
    }
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        unsafe {
            self.instance
                .logical_device()
                .logical_device()
                .destroy_pipeline_layout(self.layout, None);
        }
    }
}
