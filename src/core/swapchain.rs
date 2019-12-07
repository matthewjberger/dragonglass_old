use crate::{core::ImageView, VulkanContext};
use ash::{extensions::khr::Swapchain as AshSwapchain, vk};
use std::sync::Arc;

// TODO: Break out swapchain properties to seperate file
// TODO: Add snafu errors

#[derive(Clone, Copy, Debug)]
pub struct SwapchainProperties {
    pub format: vk::SurfaceFormatKHR,
    pub present_mode: vk::PresentModeKHR,
    pub extent: vk::Extent2D,
}

impl SwapchainProperties {
    pub fn aspect_ratio(&self) -> f32 {
        let height = if self.extent.height == 0 {
            0
        } else {
            self.extent.height
        };
        self.extent.width as f32 / height as f32
    }
}

pub struct SwapchainSupportDetails {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupportDetails {
    pub fn new(context: &VulkanContext) -> Self {
        // Get the surface capabilities
        let capabilities = unsafe {
            context
                .surface()
                .get_physical_device_surface_capabilities(
                    context.physical_device(),
                    context.surface_khr(),
                )
                .expect("Failed to get physical device surface capabilities")
        };

        // Get the supported surface formats
        let formats = unsafe {
            context
                .surface()
                .get_physical_device_surface_formats(
                    context.physical_device(),
                    context.surface_khr(),
                )
                .expect("Failed to get physical device surface formats")
        };

        // Get the supported present modes
        let present_modes = unsafe {
            context
                .surface()
                .get_physical_device_surface_present_modes(
                    context.physical_device(),
                    context.surface_khr(),
                )
                .expect("Failed to get physical device surface present modes")
        };

        Self {
            capabilities,
            formats,
            present_modes,
        }
    }

    pub fn suitable_properties(&self, preferred_dimensions: [u32; 2]) -> SwapchainProperties {
        let format = Self::choose_surface_format(&self.formats);
        let present_mode = Self::choose_surface_present_mode(&self.present_modes);
        let extent = Self::choose_swapchain_extent(self.capabilities, preferred_dimensions);
        SwapchainProperties {
            format,
            present_mode,
            extent,
        }
    }

    fn choose_surface_format(available_formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
        // Specify a default format and color space
        let (default_format, default_color_space) = (
            vk::Format::B8G8R8A8_UNORM,
            vk::ColorSpaceKHR::SRGB_NONLINEAR,
        );

        // Choose the default format if available or choose the first available format
        if available_formats.len() == 1 && available_formats[0].format == vk::Format::UNDEFINED {
            // If only one format is available
            // but it is undefined, assign a default
            vk::SurfaceFormatKHR {
                format: default_format,
                color_space: default_color_space,
            }
        } else {
            *available_formats
                .iter()
                .find(|format| {
                    format.format == default_format && format.color_space == default_color_space
                })
                .unwrap_or_else(|| {
                    available_formats
                        .first()
                        .expect("Failed to get first surface format")
                })
        }
    }

    fn choose_surface_present_mode(
        available_present_modes: &[vk::PresentModeKHR],
    ) -> vk::PresentModeKHR {
        if available_present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else if available_present_modes.contains(&vk::PresentModeKHR::FIFO) {
            vk::PresentModeKHR::FIFO
        } else {
            vk::PresentModeKHR::IMMEDIATE
        }
    }

    fn choose_swapchain_extent(
        capabilities: vk::SurfaceCapabilitiesKHR,
        preferred_dimensions: [u32; 2],
    ) -> vk::Extent2D {
        if capabilities.current_extent.width != std::u32::MAX {
            capabilities.current_extent
        } else {
            let min = capabilities.min_image_extent;
            let max = capabilities.max_image_extent;
            let width = preferred_dimensions[0].min(max.width).max(min.width);
            let height = preferred_dimensions[1].min(max.height).max(min.height);
            vk::Extent2D { width, height }
        }
    }
}

pub struct Swapchain {
    swapchain: AshSwapchain,
    swapchain_khr: vk::SwapchainKHR,
    swapchain_properties: SwapchainProperties,
    images: Vec<vk::Image>,
    image_views: Vec<ImageView>,
}

impl Swapchain {
    pub fn new(context: Arc<VulkanContext>) -> Swapchain {
        let swapchain_support_details = SwapchainSupportDetails::new(&context);
        let capabilities = &swapchain_support_details.capabilities;

        let swapchain_properties = swapchain_support_details.suitable_properties([800, 600]);
        let surface_format = swapchain_properties.format;
        let present_mode = swapchain_properties.present_mode;
        let extent = swapchain_properties.extent;

        let image_count = {
            let max = capabilities.max_image_count;
            let mut preferred = capabilities.min_image_count + 1;
            if max > 0 && preferred > max {
                preferred = max;
            }
            preferred
        };

        let swapchain_create_info = {
            let mut builder = vk::SwapchainCreateInfoKHR::builder()
                .surface(context.surface_khr())
                .min_image_count(image_count)
                .image_format(surface_format.format)
                .image_color_space(surface_format.color_space)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT);

            let mut queue_family_indices = vec![
                context.graphics_queue_family_index(),
                context.present_queue_family_index(),
            ];
            queue_family_indices.dedup();

            builder =
                if context.graphics_queue_family_index() != context.present_queue_family_index() {
                    builder
                        .image_sharing_mode(vk::SharingMode::CONCURRENT)
                        .queue_family_indices(&queue_family_indices)
                } else {
                    builder.image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                };

            builder
                .pre_transform(capabilities.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
                .build()
        };

        let swapchain = AshSwapchain::new(context.instance(), context.logical_device());
        let swapchain_khr = unsafe {
            swapchain
                .create_swapchain(&swapchain_create_info, None)
                .unwrap()
        };

        log::debug!(
            "Creating swapchain.\n\tFormat: {:?}\n\tColorSpace: {:?}\n\tPresentMode: {:?}\n\tExtent: {:?}\n\tImageCount: {}",
            surface_format.format,
            surface_format.color_space,
            present_mode,
            extent,
            image_count
        );

        let images = unsafe { swapchain.get_swapchain_images(swapchain_khr).unwrap() };
        let image_views = images
            .iter()
            .map(|image| ImageView::new(context.clone(), *image, &swapchain_properties))
            .collect::<Vec<_>>();

        Swapchain {
            swapchain,
            swapchain_khr,
            swapchain_properties,
            images: images.to_vec(),
            image_views,
        }
    }

    pub fn swapchain(&self) -> &AshSwapchain {
        &self.swapchain
    }

    pub fn swapchain_khr(&self) -> vk::SwapchainKHR {
        self.swapchain_khr
    }

    pub fn properties(&self) -> &SwapchainProperties {
        &self.swapchain_properties
    }

    pub fn images(&self) -> &[vk::Image] {
        &self.images
    }

    pub fn image_views(&self) -> &[ImageView] {
        &self.image_views
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            self.swapchain.destroy_swapchain(self.swapchain_khr, None);
        }
    }
}
