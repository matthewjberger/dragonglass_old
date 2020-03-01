use crate::{
    core::ImageView,
    render::Renderer,
    resource::{Dimension, Sampler, Texture, TextureDescription},
};
use ash::vk;
use gltf::image::Format;
use image::{ImageBuffer, Pixel, RgbImage};

pub struct GltfTextureBundle {
    pub texture: Texture,
    pub view: ImageView,
    pub sampler: Sampler,
}

impl GltfTextureBundle {
    pub fn new(renderer: &Renderer, texture_properties: &gltf::image::Data) -> Self {
        let mut texture_format = convert_to_vulkan_format(texture_properties.format);

        let pixels: Vec<u8> = match texture_format {
            vk::Format::R8G8B8_UNORM => {
                texture_format = vk::Format::R8G8B8A8_UNORM;

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
            vk::Format::B8G8R8_UNORM => {
                texture_format = vk::Format::R8G8B8A8_UNORM;

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
            _ => texture_properties.pixels.to_vec(),
        };

        let image_create_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width: texture_properties.width,
                height: texture_properties.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .format(texture_format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(vk::SampleCountFlags::TYPE_1)
            .flags(vk::ImageCreateFlags::empty())
            .build();

        let description = TextureDescription {
            format: texture_format,
            dimensions: Dimension {
                width: texture_properties.width,
                height: texture_properties.height,
            },
            pixels,
        };

        let allocation_create_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::GpuOnly,
            ..Default::default()
        };
        let texture = Texture::new(
            renderer.context.clone(),
            &allocation_create_info,
            &image_create_info,
        );
        texture.upload_data(&renderer.command_pool, renderer.graphics_queue, description);

        let create_info = vk::ImageViewCreateInfo::builder()
            .image(texture.image())
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(texture_format)
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
        let view = ImageView::new(renderer.context.clone(), create_info);

        let sampler_info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(true)
            .max_anisotropy(16.0)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(0.0)
            .build();
        let sampler = Sampler::new(renderer.context.clone(), sampler_info);

        GltfTextureBundle {
            texture,
            view,
            sampler,
        }
    }
}

pub fn convert_to_vulkan_format(format: Format) -> vk::Format {
    match format {
        Format::R8 => vk::Format::R8_UNORM,
        Format::R8G8 => vk::Format::R8G8_UNORM,
        Format::R8G8B8A8 => vk::Format::R8G8B8A8_UNORM,
        Format::B8G8R8A8 => vk::Format::B8G8R8A8_UNORM,
        // 24-bit formats will have an alpha channel added
        // to make them 32-bit
        Format::R8G8B8 => vk::Format::R8G8B8_UNORM,
        Format::B8G8R8 => vk::Format::B8G8R8_UNORM,
    }
}
