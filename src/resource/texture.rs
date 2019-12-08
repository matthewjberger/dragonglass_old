use image::{DynamicImage, GenericImageView};

use crate::{resource::Buffer, VulkanContext};
use ash::{version::DeviceV1_0, vk};
use std::{ffi::CStr, mem, sync::Arc};

// TODO: Add snafu errors

pub struct Texture {
    context: Arc<VulkanContext>,
    // image: vk::Image,
    // memory: vk::DeviceMemory,
}

impl Texture {
    pub fn from_file(context: Arc<VulkanContext>, path: &str) -> Self {
        let mut image = image::open(path).unwrap();
        let image_as_rgb = image.to_rgba();
        let width = image_as_rgb.width();
        let height = image_as_rgb.height();
        let pixels = image_as_rgb.into_raw();
        let image_size = (pixels.len() * mem::size_of::<u8>()) as vk::DeviceSize;

        let buffer = Buffer::new(
            context.clone(),
            image_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
        );

        buffer.upload_to_entire_buffer::<u8, _>(&pixels);

        Texture { context }
    }
}

impl Drop for Texture {
    fn drop(&mut self) {
        unsafe {
            // self.context.logical_device().logical_device()
            // .destroy_shader_module(self.module, None);
        }
    }
}
