use crate::{core::VulkanContext, resource::CommandPool};
use ash::{version::DeviceV1_0, vk};
use gltf::image::Format;
use image::{DynamicImage, ImageBuffer, Pixel, RgbImage};
use std::sync::Arc;

// TODO: Add snafu errors

pub struct TextureDescription {
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
    pub mip_levels: u32,
}

#[allow(dead_code)]
impl TextureDescription {
    pub fn from_file(path: &str, format: vk::Format) -> Self {
        let image = image::open(path).expect("Failed to open image path!");
        Self::from_image(&image, format)
    }

    pub fn from_image(image: &DynamicImage, format: vk::Format) -> Self {
        let image = image.to_rgba();
        let width = image.width();
        let height = image.height();
        Self {
            format,
            width,
            height,
            pixels: image.into_raw(),
            mip_levels: Self::calculate_mip_levels(width, height),
        }
    }

    pub fn from_gltf(data: &gltf::image::Data) -> Self {
        let format = Self::convert_to_vulkan_format(data.format);
        let (pixels, format) = Self::convert_pixels(&data, format);
        Self {
            format,
            width: data.width,
            height: data.height,
            pixels,
            mip_levels: Self::calculate_mip_levels(data.width, data.height),
        }
    }

    fn calculate_mip_levels(width: u32, height: u32) -> u32 {
        ((width.min(height) as f32).log2().floor() + 1.0) as u32
    }

    fn convert_pixels(
        texture_properties: &gltf::image::Data,
        mut texture_format: vk::Format,
    ) -> (Vec<u8>, vk::Format) {
        // 24-bit formats are unsupported, so they
        // need to have an alpha channel added to make them 32-bit
        let pixels: Vec<u8> = match texture_format {
            vk::Format::R8G8B8_UNORM => {
                texture_format = vk::Format::R8G8B8A8_UNORM;
                Self::attach_alpha_channel(&texture_properties)
            }
            vk::Format::B8G8R8_UNORM => {
                texture_format = vk::Format::B8G8R8A8_UNORM;
                Self::attach_alpha_channel(&texture_properties)
            }
            _ => texture_properties.pixels.to_vec(),
        };
        (pixels, texture_format)
    }

    fn attach_alpha_channel(texture_properties: &gltf::image::Data) -> Vec<u8> {
        let image_buffer: RgbImage = ImageBuffer::from_raw(
            texture_properties.width,
            texture_properties.height,
            texture_properties.pixels.to_vec(),
        )
        .expect("Failed to create an image buffer");

        image_buffer
            .pixels()
            .flat_map(|pixel| pixel.to_rgba().channels().to_vec())
            .collect::<Vec<_>>()
    }

    fn convert_to_vulkan_format(format: Format) -> vk::Format {
        match format {
            Format::R8 => vk::Format::R8_UNORM,
            Format::R8G8 => vk::Format::R8G8_UNORM,
            Format::R8G8B8A8 => vk::Format::R8G8B8A8_UNORM,
            Format::B8G8R8A8 => vk::Format::B8G8R8A8_UNORM,
            Format::R8G8B8 => vk::Format::R8G8B8_UNORM,
            Format::B8G8R8 => vk::Format::B8G8R8_UNORM,
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

    pub fn generate_mipmaps(
        &self,
        command_pool: &CommandPool,
        texture_description: &TextureDescription,
    ) {
        let format_properties = self
            .context
            .physical_device_format_properties(texture_description.format);

        if !format_properties
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
        {
            panic!(
                "Linear blitting is not supported for format: {:?}",
                texture_description.format
            );
        }

        let mut mip_width = texture_description.width as i32;
        let mut mip_height = texture_description.height as i32;
        for level in 1..texture_description.mip_levels {
            let next_mip_width = if mip_width > 1 {
                mip_width / 2
            } else {
                mip_width
            };

            let next_mip_height = if mip_height > 1 {
                mip_height / 2
            } else {
                mip_height
            };

            let barrier = vk::ImageMemoryBarrier::builder()
                .image(self.image())
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_array_layer: 0,
                    layer_count: 1,
                    level_count: 1,
                    base_mip_level: level - 1,
                })
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .build();
            let barriers = [barrier];

            command_pool.transition_image_layout(
                &barriers,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
            );

            let blit = vk::ImageBlit::builder()
                .src_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: mip_width,
                        y: mip_height,
                        z: 1,
                    },
                ])
                .src_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: level - 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .dst_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: next_mip_width,
                        y: next_mip_height,
                        z: 1,
                    },
                ])
                .dst_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: level,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .build();
            let blits = [blit];

            command_pool.execute_command_once(
                self.context.graphics_queue(),
                |command_buffer| unsafe {
                    self.context
                        .logical_device()
                        .logical_device()
                        .cmd_blit_image(
                            command_buffer,
                            self.image(),
                            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                            self.image(),
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            &blits,
                            vk::Filter::LINEAR,
                        )
                },
            );

            let barrier = vk::ImageMemoryBarrier::builder()
                .image(self.image())
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_array_layer: 0,
                    layer_count: 1,
                    level_count: 1,
                    base_mip_level: level - 1,
                })
                .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .build();
            let barriers = [barrier];

            command_pool.transition_image_layout(
                &barriers,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
            );

            mip_width = next_mip_width;
            mip_height = next_mip_height;
        }

        let barrier = vk::ImageMemoryBarrier::builder()
            .image(self.image())
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_array_layer: 0,
                layer_count: 1,
                level_count: 1,
                base_mip_level: texture_description.mip_levels - 1,
            })
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .build();
        let barriers = [barrier];

        command_pool.transition_image_layout(
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
