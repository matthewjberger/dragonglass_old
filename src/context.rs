use ash::{
    extensions::{ext::DebugUtils, khr::Swapchain},
    version::{DeviceV1_0, InstanceV1_0},
    vk::{self, DebugUtilsMessengerEXT, SurfaceKHR},
};

use crate::{
    core::{Instance, PhysicalDevice, Surface},
    debug,
};

pub struct VulkanContext {
    physical_device: PhysicalDevice,
    logical_device: ash::Device,
    surface: Surface,
    debug_messenger: Option<(DebugUtils, DebugUtilsMessengerEXT)>,
    instance: Instance,
}

// TODO: Replace constructor return value with a result
impl VulkanContext {
    pub fn new(window: &winit::Window) -> Self {
        let instance = Instance::new().unwrap();
        let surface = Surface::new(&instance, window);
        let debug_messenger = debug::setup_debug_messenger(instance.entry(), instance.instance());
        let physical_device = PhysicalDevice::new(&instance.instance(), &surface).unwrap();
        let queue_family_indices = physical_device.queue_family_index_set().indices();

        // Build an array of DeviceQueueCreateInfo,
        // one for each different family index
        let queue_create_infos = queue_family_indices
            .iter()
            .map(|index| {
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(*index)
                    .queue_priorities(&[1.0f32])
                    .build()
            })
            .collect::<Vec<_>>();

        // Specify device extensions
        let device_extensions = [Swapchain::name().as_ptr()];

        // Get the features of the physical device
        let device_features = vk::PhysicalDeviceFeatures::builder().build();

        // Create the device creation info using the queue creation info and available features
        let mut device_create_info_builder = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&device_extensions)
            .enabled_features(&device_features);

        let layer_name_vec = Instance::required_layers();
        let layer_name_pointers = layer_name_vec.layer_name_pointers();
        if debug::ENABLE_VALIDATION_LAYERS {
            // Add the validation layers to the list of enabled layers if validation layers are enabled
            device_create_info_builder =
                device_create_info_builder.enabled_layer_names(&layer_name_pointers)
        }

        let device_create_info = device_create_info_builder.build();

        // Create the logical device using the physical device and device creation info
        let logical_device = unsafe {
            instance
                .instance()
                .create_device(physical_device.physical_device(), &device_create_info, None)
                .expect("Failed to create logical device.")
        };

        VulkanContext {
            instance,
            physical_device,
            logical_device,
            surface,
            debug_messenger,
        }
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
        &self.surface.surface()
    }

    pub fn surface_khr(&self) -> SurfaceKHR {
        self.surface.surface_khr()
    }

    pub fn physical_device_memory_properties(&self) -> &ash::vk::PhysicalDeviceMemoryProperties {
        &self.physical_device.physical_device_memory_properties()
    }

    pub fn logical_device(&self) -> &ash::Device {
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

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.logical_device.destroy_device(None);
            if let Some((debug_utils, messenger)) = &mut self.debug_messenger {
                debug_utils.destroy_debug_utils_messenger(*messenger, None);
            }
        }
    }
}
