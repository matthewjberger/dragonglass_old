use crate::core::{Instance, LogicalDevice, PhysicalDevice, Surface};
use ash::{version::InstanceV1_0, vk};
use snafu::{ResultExt, Snafu};

type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, Snafu)]
#[snafu(visibility = "pub(crate)")]
pub enum Error {
    #[snafu(display("Failed to create instance for context: {}", source))]
    InstanceCreation {
        source: crate::core::instance::Error,
    },

    #[snafu(display("Failed to create logical device for context: {}", source))]
    LogicalDeviceCreation {
        source: crate::core::logical_device::Error,
    },
}

// The order the struct members here are declared in
// is important because it determines the order
// the fields are 'Drop'ped in
//
// The drop order should be:
// logical device -> physical device -> surface -> instance
pub struct VulkanContext {
    logical_device: LogicalDevice,
    physical_device: PhysicalDevice,
    surface: Surface,
    instance: Instance,
}

// TODO: Replace constructor return value with a result
impl VulkanContext {
    pub fn new(window: &winit::Window) -> Result<Self> {
        let instance = Instance::new().context(InstanceCreation)?;
        let surface = Surface::new(&instance, window);
        let physical_device = PhysicalDevice::new(&instance, &surface).unwrap();
        let logical_device =
            LogicalDevice::new(&instance, &physical_device).context(LogicalDeviceCreation)?;
        Ok(VulkanContext {
            instance,
            physical_device,
            logical_device,
            surface,
        })
    }

    // TODO: Move this to a more specific module
    pub fn determine_depth_format(&self) -> vk::Format {
        let tiling = vk::ImageTiling::OPTIMAL;
        let candidates = vec![
            vk::Format::D32_SFLOAT,
            vk::Format::D32_SFLOAT_S8_UINT,
            vk::Format::D24_UNORM_S8_UINT,
        ];
        let features = vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT;

        candidates
            .iter()
            .map(|f| *f)
            .find(|candidate| {
                let properties = unsafe {
                    self.instance
                        .instance()
                        .get_physical_device_format_properties(self.physical_device(), *candidate)
                };

                if tiling == vk::ImageTiling::LINEAR
                    && properties.linear_tiling_features.contains(features)
                {
                    true
                } else if tiling == vk::ImageTiling::OPTIMAL
                    && properties.optimal_tiling_features.contains(features)
                {
                    true
                } else {
                    false
                }
            })
            .expect("Failed to find supported depth format")
    }

    // TODO: Move this to a more specific module
    pub fn has_stencil_component(format: vk::Format) -> bool {
        format == vk::Format::D32_SFLOAT_S8_UINT || format == vk::Format::D24_UNORM_S8_UINT
    }

    // TODO: Replace accessors with accessors to wrappers
    // e.g. surface and surface_khr can be replaced with one
    // method to return the Surface wrapper

    pub fn instance(&self) -> &ash::Instance {
        self.instance.instance()
    }

    pub fn physical_device(&self) -> ash::vk::PhysicalDevice {
        self.physical_device.physical_device()
    }

    pub fn surface(&self) -> &ash::extensions::khr::Surface {
        self.surface.surface()
    }

    pub fn surface_khr(&self) -> ash::vk::SurfaceKHR {
        self.surface.surface_khr()
    }

    pub fn physical_device_memory_properties(&self) -> &ash::vk::PhysicalDeviceMemoryProperties {
        &self.physical_device.physical_device_memory_properties()
    }

    pub fn logical_device(&self) -> &LogicalDevice {
        &self.logical_device
    }

    pub fn graphics_queue_family_index(&self) -> u32 {
        self.physical_device
            .queue_family_index_set()
            .graphics_queue_family_index()
    }

    pub fn present_queue_family_index(&self) -> u32 {
        self.physical_device
            .queue_family_index_set()
            .present_queue_family_index()
    }
}
