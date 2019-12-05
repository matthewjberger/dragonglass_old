use crate::core::{error::Result, QueueFamilyIndexSet, Surface};
use ash::version::InstanceV1_0;
use std::ffi::CStr;

pub struct PhysicalDevice {
    physical_device: ash::vk::PhysicalDevice,
    physical_device_memory_properties: ash::vk::PhysicalDeviceMemoryProperties,
    queue_family_index_set: QueueFamilyIndexSet,
}

impl PhysicalDevice {
    pub fn new(instance: &ash::Instance, surface: &Surface) -> Result<Self> {
        let physical_device = Self::pick_physical_device(instance, surface);
        let physical_device_memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        // TODO: This is called twice on the physical device that is deemed suitable.
        // reduce it to one call, storing the set on the first pass
        let queue_family_index_set =
            QueueFamilyIndexSet::new(instance, physical_device, surface).unwrap();

        Ok(PhysicalDevice {
            physical_device,
            physical_device_memory_properties,
            queue_family_index_set,
        })
    }

    pub fn physical_device(&self) -> ash::vk::PhysicalDevice {
        self.physical_device
    }

    pub fn physical_device_memory_properties(&self) -> &ash::vk::PhysicalDeviceMemoryProperties {
        &self.physical_device_memory_properties
    }

    pub fn queue_family_index_set(&self) -> &QueueFamilyIndexSet {
        &self.queue_family_index_set
    }

    fn pick_physical_device(
        instance: &ash::Instance,
        surface: &Surface,
    ) -> ash::vk::PhysicalDevice {
        // Pick a physical device
        let devices = unsafe {
            instance
                .enumerate_physical_devices()
                .expect("Couldn't get physical devices")
        };

        // Pick the first suitable physical device
        let physical_device = devices
            .into_iter()
            .find(|physical_device| {
                Self::is_physical_device_suitable(instance, *physical_device, surface)
            })
            .expect("Failed to find a suitable physical device");

        // Log the name of the physical device that was selected
        let props = unsafe { instance.get_physical_device_properties(physical_device) };
        log::debug!("Selected physical device: {:?}", unsafe {
            CStr::from_ptr(props.device_name.as_ptr())
        });

        physical_device
    }

    // TODO: Refactor this to use less parameters
    fn is_physical_device_suitable(
        instance: &ash::Instance,
        physical_device: ash::vk::PhysicalDevice,
        surface: &Surface,
    ) -> bool {
        // Get the supported surface formats
        let formats = unsafe {
            surface
                .surface()
                .get_physical_device_surface_formats(physical_device, surface.surface_khr())
                .expect("Failed to get physical device surface formats")
        };

        // Get the supported present modes
        let present_modes = unsafe {
            surface
                .surface()
                .get_physical_device_surface_present_modes(physical_device, surface.surface_khr())
                .expect("Failed to get physical device surface present modes")
        };

        let queue_family_index_set = QueueFamilyIndexSet::new(instance, physical_device, surface);
        let swapchain_adequate = !formats.is_empty() && !present_modes.is_empty();

        queue_family_index_set.is_some() && swapchain_adequate
    }
}
