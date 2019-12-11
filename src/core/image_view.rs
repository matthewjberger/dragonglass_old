// TODO: Make a type alias for the current device version (DeviceV1_0)
use crate::core::VulkanContext;
use ash::{version::DeviceV1_0, vk};
use std::sync::Arc;

// TODO: Add snafu errors

pub struct ImageView {
    view: vk::ImageView,
    context: Arc<VulkanContext>,
}

impl ImageView {
    pub fn new(context: Arc<VulkanContext>, create_info: vk::ImageViewCreateInfo) -> Self {
        let view = unsafe {
            context
                .logical_device()
                .logical_device()
                .create_image_view(&create_info, None)
                .unwrap()
        };

        ImageView { view, context }
    }

    pub fn view(&self) -> vk::ImageView {
        self.view
    }
}

impl Drop for ImageView {
    fn drop(&mut self) {
        unsafe {
            self.context
                .logical_device()
                .logical_device()
                .destroy_image_view(self.view, None);
        }
    }
}
