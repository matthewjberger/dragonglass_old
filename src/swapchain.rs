use ash::{extensions::khr::Surface, vk};

#[derive(Clone, Copy, Debug)]
pub struct SwapchainProperties {
    pub format: vk::SurfaceFormatKHR,
    pub present_mode: vk::PresentModeKHR,
    pub extent: vk::Extent2D,
}

pub struct SwapchainSupportDetails {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupportDetails {
    pub fn new(
        physical_device: vk::PhysicalDevice,
        surface: &Surface,
        surface_khr: vk::SurfaceKHR,
    ) -> Self {
        // Get the surface capabilities
        let capabilities = unsafe {
            surface
                .get_physical_device_surface_capabilities(physical_device, surface_khr)
                .expect("Failed to get physical device surface capabilities")
        };

        // Get the supported surface formats
        let formats = unsafe {
            surface
                .get_physical_device_surface_formats(physical_device, surface_khr)
                .expect("Failed to get physical device surface formats")
        };

        // Get the supported present modes
        let present_modes = unsafe {
            surface
                .get_physical_device_surface_present_modes(physical_device, surface_khr)
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
