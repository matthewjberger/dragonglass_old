use ash::{
    version::{DeviceV1_0, InstanceV1_0},
    vk,
};

use crate::{
    core::{Instance, PhysicalDevice},
    sync::CurrentFrameSynchronization,
};

use snafu::{ResultExt, Snafu};

type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, Snafu)]
#[snafu(visibility = "pub(crate)")]
pub enum Error {
    #[snafu(display("Failed to create logical device: {}", source))]
    LogicalDeviceCreation { source: vk::Result },
}

pub struct LogicalDevice {
    logical_device: ash::Device,
}

impl LogicalDevice {
    pub fn new(
        instance: &Instance,
        physical_device: &PhysicalDevice,
        device_create_info: vk::DeviceCreateInfo,
    ) -> Result<Self> {
        let logical_device = unsafe {
            instance
                .instance()
                .create_device(physical_device.physical_device(), &device_create_info, None)
                .context(LogicalDeviceCreation)?
        };

        Ok(LogicalDevice { logical_device })
    }

    pub fn logical_device(&self) -> &ash::Device {
        &self.logical_device
    }

    // TODO: Add error handling
    pub fn wait_for_fence(&self, current_frame_synchronization: &CurrentFrameSynchronization) {
        let in_flight_fences = [current_frame_synchronization.in_flight()];
        unsafe {
            self.logical_device
                .wait_for_fences(&in_flight_fences, true, std::u64::MAX)
                .expect("Failed to wait for fences!");
        }
    }

    pub fn reset_fence(&self, current_frame_synchronization: &CurrentFrameSynchronization) {
        let in_flight_fences = [current_frame_synchronization.in_flight()];
        unsafe {
            self.logical_device()
                .reset_fences(&in_flight_fences)
                .expect("Failed to reset fences!");
        }
    }

    pub fn wait_idle(&self) {
        unsafe {
            self.logical_device
                .device_wait_idle()
                .expect("Failed to wait for the logical device to be idle!")
        };
    }
}

impl Drop for LogicalDevice {
    fn drop(&mut self) {
        unsafe {
            self.logical_device.destroy_device(None);
        }
    }
}
