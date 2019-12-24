use crate::{
    core::VulkanContext,
    resource::{Buffer, CommandPool},
};
use ash::{version::DeviceV1_0, vk};
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
                .unwrap()
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
                .unwrap();
            logical_device
                .bind_image_memory(image, memory_handle, 0)
                .unwrap();
            memory_handle
        };

        Texture {
            image,
            memory,
            context,
        }
    }

    // TODO: Refactor this to use less parameters
    #[allow(dead_code)]
    pub fn from_file(
        context: Arc<VulkanContext>,
        command_pool: &CommandPool,
        graphics_queue: vk::Queue,
        path: &str,
        format: vk::Format,
        create_info_builder: vk::ImageCreateInfoBuilder,
    ) -> Self {
        let image = image::open(path).unwrap();
        let image = image.to_rgba();
        let width = image.width();
        let height = image.height();
        let description = TextureDescription {
            format,
            dimensions: Dimension { width, height },
            pixels: image.into_raw(),
        };
        Self::from_data(
            context.clone(),
            command_pool,
            graphics_queue,
            description,
            create_info_builder
                .extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })
                .build(),
        )
    }

    // TODO: Refactor this to use less parameters
    pub fn from_data(
        context: Arc<VulkanContext>,
        command_pool: &CommandPool,
        graphics_queue: vk::Queue,
        description: TextureDescription,
        create_info: vk::ImageCreateInfo,
    ) -> Self {
        let image_size = (description.pixels.len() * mem::size_of::<u8>()) as vk::DeviceSize;
        let texture = Self::new(context.clone(), create_info);

        let buffer = Buffer::new(
            context.clone(),
            image_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
        );

        buffer.upload_to_buffer(&description.pixels, 0);

        command_pool.transition_image_layout(
            graphics_queue,
            texture.image(),
            description.format,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );

        command_pool.copy_buffer_to_image(
            graphics_queue,
            buffer.buffer(),
            texture.image(),
            description.dimensions.width,
            description.dimensions.height,
        );

        command_pool.transition_image_layout(
            graphics_queue,
            texture.image(),
            description.format,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );

        texture
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
