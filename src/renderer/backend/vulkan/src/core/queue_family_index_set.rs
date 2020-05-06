use crate::core::Surface;
use ash::{version::InstanceV1_0, vk};

pub struct QueueFamilyIndexSet {
    graphics_queue_family_index: u32,
    present_queue_family_index: u32,
}

impl QueueFamilyIndexSet {
    pub fn new(
        instance: &ash::Instance,
        physical_device: ash::vk::PhysicalDevice,
        surface: &Surface,
    ) -> Option<Self> {
        // According to the Vulkan spec, the present queue
        // and graphics queue are not guaranteed to have the same index
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
                surface
                    .surface()
                    .get_physical_device_surface_support(
                        physical_device,
                        index,
                        surface.surface_khr(),
                    )
                    .expect("Failed to get physical device surface support!")
            };

            if present_support && present_queue_family_index.is_none() {
                present_queue_family_index = Some(index);
            }

            if graphics_queue_family_index.is_some() && present_queue_family_index.is_some() {
                break;
            }
        }

        if graphics_queue_family_index.is_none() || present_queue_family_index.is_none() {
            return None;
        }

        Some(QueueFamilyIndexSet {
            graphics_queue_family_index: graphics_queue_family_index
                .expect("Failed to get graphics queue family index!"),
            present_queue_family_index: present_queue_family_index
                .expect("Failed to get present queue family index!"),
        })
    }

    pub fn graphics_queue_family_index(&self) -> u32 {
        self.graphics_queue_family_index
    }

    pub fn present_queue_family_index(&self) -> u32 {
        self.present_queue_family_index
    }

    pub fn indices(&self) -> Vec<u32> {
        // The queue family indices need to be deduplicated because
        // Vulkan does not allow passing an array containing duplicated family
        // indices, and it is possible for the graphics queue family index
        // and present queue family index to be the same.
        let mut queue_family_indices = vec![
            self.graphics_queue_family_index,
            self.present_queue_family_index,
        ];
        queue_family_indices.dedup();
        queue_family_indices
    }
}
