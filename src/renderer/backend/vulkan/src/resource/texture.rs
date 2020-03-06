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

        let barrier = vk::ImageMemoryBarrier::builder()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(self.image())
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .build();
        let barriers = [barrier];

        command_pool.transition_image_layout(
            graphics_queue,
            &barriers,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
        );

        command_pool.copy_buffer_to_image(
            graphics_queue,
            buffer.buffer(),
            self.image(),
            description.dimensions.width,
            description.dimensions.height,
        );

        let barrier = vk::ImageMemoryBarrier::builder()
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(self.image())
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .build();
        let barriers = [barrier];

        command_pool.transition_image_layout(
            graphics_queue,
            &barriers,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
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
