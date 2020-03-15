use ash::vk;
use gltf::image::Format;
use image::{DynamicImage, ImageBuffer, Pixel, RgbImage};

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
