use crate::core::{Instance, LogicalDevice, PhysicalDevice, Surface};
use ash::{
    version::{DeviceV1_0, InstanceV1_0},
    vk,
};
use snafu::{ResultExt, Snafu};
use vk_mem::{Allocator, AllocatorCreateInfo};

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
    allocator: vk_mem::Allocator,
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
        let physical_device =
            PhysicalDevice::new(&instance, &surface).expect("Failed to get physical device!");
        let logical_device =
            LogicalDevice::new(&instance, &physical_device).context(LogicalDeviceCreation)?;

        let allocator_create_info = AllocatorCreateInfo {
            device: (*logical_device.logical_device()).clone(),
            instance: (*instance.instance()).clone(),
            physical_device: physical_device.physical_device(),
            ..Default::default()
        };

        let allocator = Allocator::new(&allocator_create_info).expect("Allocator creation failed");

        Ok(VulkanContext {
            allocator,
            instance,
            physical_device,
            logical_device,
            surface,
        })
    }

    pub fn determine_depth_format(
        &self,
        tiling: vk::ImageTiling,
        features: vk::FormatFeatureFlags,
    ) -> vk::Format {
        let candidates = vec![
            vk::Format::D32_SFLOAT,
            vk::Format::D32_SFLOAT_S8_UINT,
            vk::Format::D24_UNORM_S8_UINT,
        ];
        candidates
            .iter()
            .copied()
            .find(|candidate| {
                let properties = unsafe {
                    self.instance()
                        .get_physical_device_format_properties(self.physical_device(), *candidate)
                };

                let linear_tiling_feature_support = tiling == vk::ImageTiling::LINEAR
                    && properties.linear_tiling_features.contains(features);

                let optimal_tiling_feature_support = tiling == vk::ImageTiling::OPTIMAL
                    && properties.optimal_tiling_features.contains(features);

                linear_tiling_feature_support || optimal_tiling_feature_support
            })
            .expect("Failed to find a supported depth format")
    }

    pub fn physical_device_properties(&self) -> vk::PhysicalDeviceProperties {
        unsafe {
            self.instance()
                .get_physical_device_properties(self.physical_device())
        }
    }

    pub fn allocator(&self) -> &vk_mem::Allocator {
        &self.allocator
    }

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

    // TODO: Move these down to the logical device
    pub fn graphics_queue(&self) -> vk::Queue {
        unsafe {
            self.logical_device()
                .logical_device()
                .get_device_queue(self.graphics_queue_family_index(), 0)
        }
    }

    pub fn present_queue(&self) -> vk::Queue {
        unsafe {
            self.logical_device()
                .logical_device()
                .get_device_queue(self.present_queue_family_index(), 0)
        }
    }

    pub fn wait_idle(&self) {
        unsafe {
            self.logical_device()
                .logical_device()
                .device_wait_idle()
                .expect("Failed to wait for the logical device to be idle!")
        }
    }
}
