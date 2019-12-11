use crate::core::VulkanContext;
use ash::{version::DeviceV1_0, vk};
use std::sync::Arc;

// Add snafu errors

pub struct Sampler {
    sampler: vk::Sampler,
    context: Arc<VulkanContext>,
}

impl Sampler {
    pub fn new(context: Arc<VulkanContext>) -> Self {
        let sampler_info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            // TODO: Request the anisotropy feature when getting the physical device
            // .anisotropy_enable(true)
            // .max_anisotropy(16.0)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(0.0)
            .build();

        let sampler = unsafe {
            context
                .logical_device()
                .logical_device()
                .create_sampler(&sampler_info, None)
                .unwrap()
        };

        Sampler { sampler, context }
    }

    pub fn sampler(&self) -> vk::Sampler {
        self.sampler
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe {
            self.context
                .logical_device()
                .logical_device()
                .destroy_sampler(self.sampler, None)
        };
    }
}
