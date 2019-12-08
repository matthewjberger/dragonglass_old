// TODO: Make a type alias for the current device version (DeviceV1_0)
use crate::core::Instance;
use ash::{version::DeviceV1_0, vk};
use std::sync::Arc;

// TODO: Add snafu errors

pub struct ImageView {
    view: vk::ImageView,
    instance: Arc<Instance>,
}

impl ImageView {
    pub fn new(instance: Arc<Instance>, image: vk::Image, format: vk::Format) -> Self {
        let create_info = vk::ImageViewCreateInfo::builder()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .components(vk::ComponentMapping {
                r: vk::ComponentSwizzle::IDENTITY,
                g: vk::ComponentSwizzle::IDENTITY,
                b: vk::ComponentSwizzle::IDENTITY,
                a: vk::ComponentSwizzle::IDENTITY,
            })
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .build();

        let view = unsafe {
            instance
                .logical_device()
                .logical_device()
                .create_image_view(&create_info, None)
                .unwrap()
        };

        ImageView { view, instance }
    }

    pub fn view(&self) -> vk::ImageView {
        self.view
    }
}

impl Drop for ImageView {
    fn drop(&mut self) {
        unsafe {
            self.instance
                .logical_device()
                .logical_device()
                .destroy_image_view(self.view, None);
        }
    }
}
