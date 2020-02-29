use crate::{
    core::VulkanContext,
    resource::{Buffer, CommandPool},
};
use ash::{version::DeviceV1_0, vk};
use image::DynamicImage;
use std::{mem, sync::Arc};

// TODO: Add snafu errors

// TODO: Allow creating texture from passed in pixel data

pub struct Dimension {
    pub width: u32,
    pub height: u32,
}

// These are parameters needed for
// *both* creating and uploading image data
pub struct TextureDescription {
    pub format: vk::Format,
    pub dimensions: Dimension,
    pub pixels: Vec<u8>,
}

#[allow(dead_code)]
impl TextureDescription {
    fn from_file(path: &str, format: vk::Format) -> Self {
        let image = image::open(path).expect("Failed to open image path!");
        Self::from_image(&image, format)
    }

    fn from_image(image: &DynamicImage, format: vk::Format) -> Self {
        let image = image.to_rgba();
        let width = image.width();
        let height = image.height();
        TextureDescription {
            format,
            dimensions: Dimension { width, height },
            pixels: image.into_raw(),
        }
    }
}

// The order of the struct fields matters here
// because it determines drop order
pub struct Texture {
    image: vk::Image,
    memory: vk::DeviceMemory,
    context: Arc<VulkanContext>,
}

impl Texture {
    pub fn new(context: Arc<VulkanContext>, create_info: vk::ImageCreateInfo) -> Self {
        let image = unsafe {
            context
                .logical_device()
                .logical_device()
                .create_image(&create_info, None)
                .expect("Failed to create image!")
        };

        let memory_requirements = unsafe {
            context
                .logical_device()
                .logical_device()
                .get_image_memory_requirements(image)
        };

        let memory_type_index = Buffer::determine_memory_type_index(
            memory_requirements,
            context.physical_device_memory_properties(),
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        let allocation_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(memory_requirements.size)
            .memory_type_index(memory_type_index)
            .build();

        let memory = unsafe {
            let logical_device = context.logical_device().logical_device();
            let memory_handle = logical_device
                .allocate_memory(&allocation_info, None)
                .expect("Failed to allocate memory!");
            logical_device
                .bind_image_memory(image, memory_handle, 0)
                .expect("Failed to bind image");
            memory_handle
        };

        Texture {
            image,
            memory,
            context,
        }
    }

    pub fn upload_data(
        &self,
        command_pool: &CommandPool,
        graphics_queue: vk::Queue,
        description: TextureDescription,
    ) {
        let image_size = (description.pixels.len() * mem::size_of::<u8>()) as vk::DeviceSize;
        let buffer = Buffer::new(
            self.context.clone(),
            image_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
        );

        buffer.upload_to_buffer(
            &description.pixels,
            0,
            std::mem::align_of::<u8>() as _,
            false,
        );

        command_pool.transition_image_layout(
            graphics_queue,
            self.image(),
            description.format,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );

        command_pool.copy_buffer_to_image(
            graphics_queue,
            buffer.buffer(),
            self.image(),
            description.dimensions.width,
            description.dimensions.height,
        );

        command_pool.transition_image_layout(
            graphics_queue,
            self.image(),
            description.format,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );
    }

    pub fn image(&self) -> vk::Image {
        self.image
    }
}

impl Drop for Texture {
    fn drop(&mut self) {
        unsafe {
            self.context
                .logical_device()
                .logical_device()
                .destroy_image(self.image, None);
            self.context
                .logical_device()
                .logical_device()
                .free_memory(self.memory, None);
        }
    }
}
