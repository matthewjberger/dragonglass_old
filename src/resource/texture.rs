use crate::{
    core::{ImageView, VulkanContext},
    resource::{Buffer, CommandPool},
};
use ash::{version::DeviceV1_0, vk};
use std::{mem, sync::Arc};

// TODO: Add snafu errors

// TODO: Allow creating texture from passed in pixel data

// The order of the struct fields matters here
// because it determines drop order
pub struct Texture {
    view: Option<ImageView>,
    image: vk::Image,
    memory: vk::DeviceMemory,
    context: Arc<VulkanContext>,
}

// TODO: Make a texture builder to shorten the creation parameters
impl Texture {
    pub fn new(
        context: Arc<VulkanContext>,
        width: u32,
        height: u32,
        format: vk::Format,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
    ) -> Self {
        let image_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .format(format)
            .tiling(tiling)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(vk::SampleCountFlags::TYPE_1)
            .flags(vk::ImageCreateFlags::empty())
            .build();

        let image = unsafe {
            context
                .logical_device()
                .logical_device()
                .create_image(&image_info, None)
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
            view: None,
            memory,
            context,
        }
    }

    // TODO: Refactor this to use less parameters
    pub fn from_file(
        context: Arc<VulkanContext>,
        command_pool: &CommandPool,
        graphics_queue: vk::Queue,
        path: &str,
        format: vk::Format,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
        aspect_mask: vk::ImageAspectFlags,
        view_type: vk::ImageViewType,
    ) -> Self {
        let image = image::open(path).unwrap();
        let image_as_rgb = image.to_rgba();
        let width = image_as_rgb.width();
        let height = image_as_rgb.height();
        let pixels = image_as_rgb.into_raw();
        Self::from_data(
            context.clone(),
            command_pool,
            graphics_queue,
            format,
            tiling,
            usage,
            aspect_mask,
            view_type,
            width,
            height,
            &pixels,
        )
    }

    // TODO: Refactor this to use less parameters
    pub fn from_data(
        context: Arc<VulkanContext>,
        command_pool: &CommandPool,
        graphics_queue: vk::Queue,
        format: vk::Format,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
        aspect_mask: vk::ImageAspectFlags,
        view_type: vk::ImageViewType,
        width: u32,
        height: u32,
        pixels: &[u8],
    ) -> Self {
        let image_size = (pixels.len() * mem::size_of::<u8>()) as vk::DeviceSize;
        let mut texture = Self::new(context.clone(), width, height, format, tiling, usage);

        let buffer = Buffer::new(
            context.clone(),
            image_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
        );

        buffer.upload_to_entire_buffer(&pixels);

        command_pool.transition_image_layout(
            graphics_queue,
            texture.image(),
            format,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );

        command_pool.copy_buffer_to_image(
            graphics_queue,
            buffer.buffer(),
            texture.image(),
            width,
            height,
        );

        command_pool.transition_image_layout(
            graphics_queue,
            texture.image(),
            format,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );

        texture.create_view(format, aspect_mask, view_type);
        texture
    }

    pub fn image(&self) -> vk::Image {
        self.image
    }

    pub fn create_view(
        &mut self,
        format: vk::Format,
        aspect_mask: vk::ImageAspectFlags,
        view_type: vk::ImageViewType,
    ) {
        self.view = Some(ImageView::new(
            self.context.clone(),
            self.image(),
            format,
            aspect_mask,
            view_type,
        ));
    }

    pub fn view(&self) -> Option<&ImageView> {
        self.view.as_ref()
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
