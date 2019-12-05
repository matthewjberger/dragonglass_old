use ash::{
    extensions::khr::Surface,
    version::InstanceV1_0,
    vk::{self, SurfaceKHR},
};
use std::ffi::CStr;

pub struct PhysicalDevice {
    physical_device: ash::vk::PhysicalDevice,
    physical_device_memory_properties: ash::vk::PhysicalDeviceMemoryProperties,
}

impl PhysicalDevice {
    pub fn new(
        instance: &ash::Instance,
        surface: &ash::extensions::khr::Surface,
        surface_khr: ash::vk::SurfaceKHR,
    ) -> Self {
        let physical_device = Self::pick_physical_device(instance, &surface, surface_khr);

        // Get the memory properties of the physical device
        let physical_device_memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        PhysicalDevice {
            physical_device,
            physical_device_memory_properties,
        }
    }

    pub fn physical_device(&self) -> ash::vk::PhysicalDevice {
        self.physical_device
    }

    pub fn physical_device_memory_properties(&self) -> &ash::vk::PhysicalDeviceMemoryProperties {
        &self.physical_device_memory_properties
    }

    fn pick_physical_device(
        instance: &ash::Instance,
        surface: &Surface,
        surface_khr: SurfaceKHR,
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
                Self::is_physical_device_suitable(instance, *physical_device, surface, surface_khr)
            })
            .expect("Failed to find a suitable physical device");

        // Log the name of the physical device that was selected
        let props = unsafe { instance.get_physical_device_properties(physical_device) };
        log::debug!("Selected physical device: {:?}", unsafe {
            CStr::from_ptr(props.device_name.as_ptr())
        });

        physical_device
    }

    fn is_physical_device_suitable(
        instance: &ash::Instance,
        physical_device: ash::vk::PhysicalDevice,
        surface: &Surface,
        surface_khr: SurfaceKHR,
    ) -> bool {
        let (graphics_queue_family_index, present_queue_family_index) =
            Self::find_queue_family_indices(instance, physical_device, surface, surface_khr);

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

        let queue_families_supported =
            graphics_queue_family_index.is_some() && present_queue_family_index.is_some();
        let swapchain_adequate = !formats.is_empty() && !present_modes.is_empty();

        queue_families_supported && swapchain_adequate
    }

    // TODO: Refactor the queue family indices out to a struct
    pub fn find_queue_family_indices(
        instance: &ash::Instance,
        physical_device: ash::vk::PhysicalDevice,
        surface: &Surface,
        surface_khr: SurfaceKHR,
    ) -> (Option<u32>, Option<u32>) {
        let mut graphics_queue_family_index = None;
        let mut present_queue_family_index = None;

        let properties =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

        for (index, family) in properties.iter().filter(|f| f.queue_count > 0).enumerate() {
            let index = index as u32;

            // Check for a graphics queue
            if family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                && graphics_queue_family_index.is_none()
            {
                graphics_queue_family_index = Some(index);
            }

            // Check for a present queue
            let present_support = unsafe {
                surface.get_physical_device_surface_support(physical_device, index, surface_khr)
            };

            if present_support && present_queue_family_index.is_none() {
                present_queue_family_index = Some(index);
            }

            if graphics_queue_family_index.is_some() && present_queue_family_index.is_some() {
                break;
            }
        }

        (graphics_queue_family_index, present_queue_family_index)
    }
}
