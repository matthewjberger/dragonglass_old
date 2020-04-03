use crate::core::{DebugLayer, Instance, LogicalDevice, PhysicalDevice, Surface};
use ash::{
    extensions::khr::Swapchain,
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

    #[snafu(display("Failed to create physical device for context: {}", source))]
    PhysicalDeviceCreation {
        source: crate::core::physical_device::Error,
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
            PhysicalDevice::new(&instance, &surface).context(PhysicalDeviceCreation)?;

        let logical_device = Self::create_logical_device(&instance, &physical_device)?;

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

    fn create_logical_device(
        instance: &Instance,
        physical_device: &PhysicalDevice,
    ) -> Result<LogicalDevice> {
        let device_extensions = [Swapchain::name().as_ptr()];
        let queue_creation_info_list = physical_device.build_queue_creation_info_list();
        let device_features = vk::PhysicalDeviceFeatures::builder()
            //.robust_buffer_access(true) // FIXME: Disable this in release builds
            .sample_rate_shading(true)
            .sampler_anisotropy(true)
            .build();
        let mut device_create_info_builder = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_creation_info_list)
            .enabled_extension_names(&device_extensions)
            .enabled_features(&device_features);

        let layer_name_vec = Instance::required_layers();
        let layer_name_pointers = layer_name_vec.layer_name_pointers();
        if DebugLayer::validation_layers_enabled() {
            device_create_info_builder =
                device_create_info_builder.enabled_layer_names(&layer_name_pointers)
        }

        LogicalDevice::new(
            &instance,
            &physical_device,
            device_create_info_builder.build(),
        )
        .context(LogicalDeviceCreation)
    }

    pub fn max_usable_samples(&self) -> vk::SampleCountFlags {
        let properties = self.physical_device_properties();
        let color_sample_counts = properties.limits.framebuffer_color_sample_counts;
        let depth_sample_counts = properties.limits.framebuffer_depth_sample_counts;
        let sample_counts = color_sample_counts.min(depth_sample_counts);

        if sample_counts.contains(vk::SampleCountFlags::TYPE_64) {
            vk::SampleCountFlags::TYPE_64
        } else if sample_counts.contains(vk::SampleCountFlags::TYPE_32) {
            vk::SampleCountFlags::TYPE_32
        } else if sample_counts.contains(vk::SampleCountFlags::TYPE_16) {
            vk::SampleCountFlags::TYPE_16
        } else if sample_counts.contains(vk::SampleCountFlags::TYPE_8) {
            vk::SampleCountFlags::TYPE_8
        } else if sample_counts.contains(vk::SampleCountFlags::TYPE_4) {
            vk::SampleCountFlags::TYPE_4
        } else if sample_counts.contains(vk::SampleCountFlags::TYPE_2) {
            vk::SampleCountFlags::TYPE_2
        } else {
            vk::SampleCountFlags::TYPE_1
        }
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

    pub fn physical_device_format_properties(&self, format: vk::Format) -> vk::FormatProperties {
        unsafe {
            self.instance()
                .get_physical_device_format_properties(self.physical_device(), format)
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
