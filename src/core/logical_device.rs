use ash::{
    extensions::khr::Swapchain,
    version::{DeviceV1_0, InstanceV1_0},
    vk,
};

use crate::core::{DebugLayer, Instance, PhysicalDevice};

use snafu::{ResultExt, Snafu};

type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, Snafu)]
#[snafu(visibility = "pub(crate)")]
pub enum Error {
    #[snafu(display("Failed to create logical device: {}", source))]
    LogicalDeviceCreationFailed { source: vk::Result },
}

pub struct LogicalDevice {
    logical_device: ash::Device,
}

impl LogicalDevice {
    pub fn new(instance: &Instance, physical_device: &PhysicalDevice) -> Result<Self> {
        let device_extensions = [Swapchain::name().as_ptr()];
        let queue_creation_info_list = Self::build_queue_creation_info_list(physical_device);
        let device_features = vk::PhysicalDeviceFeatures::builder().build();
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

        let logical_device = unsafe {
            instance
                .instance()
                .create_device(
                    physical_device.physical_device(),
                    &device_create_info_builder.build(),
                    None,
                )
                .context(LogicalDeviceCreationFailed)?
        };

        Ok(LogicalDevice { logical_device })
    }

    pub fn logical_device(&self) -> &ash::Device {
        &self.logical_device
    }

    fn build_queue_creation_info_list(
        physical_device: &PhysicalDevice,
    ) -> Vec<vk::DeviceQueueCreateInfo> {
        // Build an array of DeviceQueueCreateInfo,
        // one for each different family index
        physical_device
            .queue_family_index_set()
            .indices()
            .iter()
            .map(|index| {
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(*index)
                    .queue_priorities(&[1.0f32])
                    .build()
            })
            .collect::<Vec<_>>()
    }
}

impl Drop for LogicalDevice {
    fn drop(&mut self) {
        unsafe {
            self.logical_device.destroy_device(None);
        }
    }
}
