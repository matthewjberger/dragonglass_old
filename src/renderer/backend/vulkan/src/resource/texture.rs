use crate::{
    core::VulkanContext,
    resource::{Buffer, CommandPool},
};
use ash::vk;
use image::DynamicImage;
use std::sync::Arc;

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
    allocation: vk_mem::Allocation,
    allocation_info: vk_mem::AllocationInfo,
    context: Arc<VulkanContext>,
}

impl Texture {
    pub fn new(
        context: Arc<VulkanContext>,
        allocation_create_info: &vk_mem::AllocationCreateInfo,
        image_create_info: &vk::ImageCreateInfo,
    ) -> Self {
        let (image, allocation, allocation_info) = context
            .allocator()
            .create_image(&image_create_info, &allocation_create_info)
            .expect("Failed to create image!");

        Self {
            image,
            allocation,
            allocation_info,
            context,
        }
    }

    pub fn upload_data(
        &self,
        command_pool: &CommandPool,
        graphics_queue: vk::Queue,
        description: TextureDescription,
    ) {
        let buffer = Buffer::new_mapped_basic(
            self.context.clone(),
            self.allocation_info.get_size() as _,
            vk::BufferUsageFlags::TRANSFER_SRC,
            //vk::MemoryPropertyFlags::HOST_VISIBLE,
            vk_mem::MemoryUsage::CpuToGpu,
        );
        buffer.upload_to_buffer(&description.pixels, 0, std::mem::align_of::<u8>() as _);

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

    pub fn allocation(&self) -> &vk_mem::Allocation {
        &self.allocation
    }

    pub fn allocation_info(&self) -> &vk_mem::AllocationInfo {
        &self.allocation_info
    }
}

impl Drop for Texture {
    fn drop(&mut self) {
        self.context
            .allocator()
            .destroy_image(self.image, &self.allocation)
            .expect("Failed to destroy image!");
    }
}
