use ash::{
    extensions::{ext::DebugUtils, khr::Surface},
    version::EntryV1_0,
    version::InstanceV1_0,
    vk::{self, DebugUtilsMessengerEXT, PhysicalDevice, SurfaceKHR},
    vk_make_version,
};
use std::ffi::{CStr, CString};

const APPLICATION_VERSION: u32 = vk_make_version!(1, 0, 0);
const API_VERSION: u32 = vk_make_version!(1, 0, 0);
const ENGINE_VERSION: u32 = vk_make_version!(1, 0, 0);
const ENGINE_NAME: &str = "Sepia Engine";

use crate::debug;
use crate::surface;

pub struct VulkanContext {
    _entry: ash::Entry,
    instance: ash::Instance,
    physical_device: PhysicalDevice,
    surface: Surface,
    surface_khr: SurfaceKHR,
    debug_messenger: Option<(DebugUtils, DebugUtilsMessengerEXT)>,
}

impl VulkanContext {
    pub fn new(window: &winit::Window) -> Self {
        // Load the Vulkan library
        let entry = ash::Entry::new().expect("Failed to create entry");

        let app_info = Self::build_application_creation_info();

        // Determine required extension names
        let instance_extensions = Self::required_instance_extension_names();

        // Create the instance creation info
        let mut instance_create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&instance_extensions);

        // Determine the required layer names
        let layer_names = debug::REQUIRED_LAYERS
            .iter()
            .map(|name| CString::new(*name).expect("Failed to build CString"))
            .collect::<Vec<_>>();

        // Determine required layer name pointers
        let layer_name_ptrs = layer_names
            .iter()
            .map(|name| name.as_ptr())
            .collect::<Vec<_>>();

        if debug::ENABLE_VALIDATION_LAYERS {
            // Check if the required validation layers are supported
            for required in debug::REQUIRED_LAYERS.iter() {
                let found = entry
                    .enumerate_instance_layer_properties()
                    .expect("Couldn't enumerate instance layer properties")
                    .iter()
                    .any(|layer| {
                        let name = unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) };
                        let name = name.to_str().expect("Failed to get layer name pointer");
                        required == &name
                    });

                if !found {
                    panic!("Validation layer not supported: {}", required);
                }
            }

            instance_create_info = instance_create_info.enabled_layer_names(&layer_name_ptrs);
        }

        let instance = unsafe {
            entry
                .create_instance(&instance_create_info, None)
                .expect("Failed to create instance")
        };

        // Create the window surface
        let surface = Surface::new(&entry, &instance);
        let surface_khr = unsafe {
            surface::create_surface(&entry, &instance, window)
                .expect("Failed to create window surface!")
        };

        let debug_messenger = debug::setup_debug_messenger(&entry, &instance);

        let physical_device = Self::pick_physical_device(&instance, &surface, surface_khr);

        VulkanContext {
            _entry: entry,
            instance,
            physical_device,
            surface,
            surface_khr,
            debug_messenger,
        }
    }

    fn build_application_creation_info() -> vk::ApplicationInfo {
        let app_name = CString::new("Vulkan Tutorial").expect("Failed to create CString");
        let engine_name = CString::new(ENGINE_NAME).expect("Failed to create CString");
        vk::ApplicationInfo::builder()
            .application_name(&app_name)
            .engine_name(&engine_name)
            .api_version(API_VERSION)
            .application_version(APPLICATION_VERSION)
            .engine_version(ENGINE_VERSION)
            .build()
    }

    fn required_instance_extension_names() -> Vec<*const i8> {
        let mut instance_extension_names = surface::surface_extension_names();
        if debug::ENABLE_VALIDATION_LAYERS {
            instance_extension_names.push(DebugUtils::name().as_ptr());
        }
        instance_extension_names
    }

    fn pick_physical_device(
        instance: &ash::Instance,
        surface: &Surface,
        surface_khr: SurfaceKHR,
    ) -> PhysicalDevice {
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
        physical_device: PhysicalDevice,
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

    fn find_queue_family_indices(
        instance: &ash::Instance,
        physical_device: PhysicalDevice,
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

    pub fn queue_family_indices(&self) -> (Option<u32>, Option<u32>) {
        Self::find_queue_family_indices(
            &self.instance,
            self.physical_device,
            &self.surface,
            self.surface_khr,
        )
    }

    pub fn instance(&self) -> &ash::Instance {
        &self.instance
    }

    pub fn physical_device(&self) -> PhysicalDevice {
        self.physical_device
    }

    pub fn surface(&self) -> &Surface {
        &self.surface
    }

    pub fn surface_khr(&self) -> SurfaceKHR {
        self.surface_khr
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.surface.destroy_surface(self.surface_khr, None);
            if let Some((debug_utils, messenger)) = &mut self.debug_messenger {
                debug_utils.destroy_debug_utils_messenger(*messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}
