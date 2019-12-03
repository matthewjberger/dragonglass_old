use ash::{
    extensions::{
        ext::DebugUtils,
        khr::{Surface, Swapchain},
    },
    version::{DeviceV1_0, EntryV1_0, InstanceV1_0},
    vk::{self, DebugUtilsMessengerEXT, PhysicalDevice, SurfaceKHR},
    vk_make_version,
};
use snafu::{ResultExt, Snafu};
use std::ffi::{CStr, CString};

use crate::debug::{self, LayerNameVec};
use crate::surface;

#[derive(Debug, Snafu)]
enum Error {
    #[snafu(display("Failed to create entry: {}", source))]
    EntryLoading { source: ash::LoadingError },

    #[snafu(display("Failed to create instance: {}", source))]
    InstanceCreation { source: ash::InstanceError },

    #[snafu(display("Failed to create a c-string from the application name: {}", source))]
    AppNameCreation { source: std::ffi::NulError },

    #[snafu(display("Failed to create a c-string from the engine name: {}", source))]
    EngineNameCreation { source: std::ffi::NulError },
}

type Result<T, E = Error> = std::result::Result<T, E>;

trait ApplicationDescription {
    const APPLICATION_NAME: &'static str;
    const APPLICATION_VERSION: u32;
    const API_VERSION: u32;
    const ENGINE_VERSION: u32;
    const ENGINE_NAME: &'static str;
}

pub struct Instance {
    entry: ash::Entry,
    instance: ash::Instance,
}

impl Instance {
    fn new() -> Result<Self> {
        let entry = ash::Entry::new().context(EntryLoading)?;
        Self::check_required_layers_supported(&entry);
        let app_info = Self::build_application_creation_info()?;
        let instance_extensions = Self::required_instance_extension_names();
        let layer_name_vec = Self::required_layers();
        let layer_name_pointers = layer_name_vec.layer_name_pointers();
        let instance_create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&instance_extensions)
            .enabled_layer_names(&layer_name_pointers);
        let instance = unsafe {
            entry
                .create_instance(&instance_create_info, None)
                .context(InstanceCreation)?
        };
        Ok(Instance { entry, instance })
    }

    pub fn entry(&self) -> &ash::Entry {
        &self.entry
    }

    pub fn instance(&self) -> &ash::Instance {
        &self.instance
    }

    fn build_application_creation_info() -> Result<vk::ApplicationInfo> {
        let app_name = CString::new(Instance::APPLICATION_NAME).context(AppNameCreation)?;
        let engine_name = CString::new(Instance::ENGINE_NAME).context(EngineNameCreation)?;
        let app_info = vk::ApplicationInfo::builder()
            .application_name(&app_name)
            .engine_name(&engine_name)
            .api_version(Instance::API_VERSION)
            .application_version(Instance::APPLICATION_VERSION)
            .engine_version(Instance::ENGINE_VERSION)
            .build();
        Ok(app_info)
    }

    fn required_instance_extension_names() -> Vec<*const i8> {
        let mut instance_extension_names = surface::surface_extension_names();
        if debug::ENABLE_VALIDATION_LAYERS {
            instance_extension_names.push(DebugUtils::name().as_ptr());
        }
        instance_extension_names
    }

    pub fn required_layers() -> LayerNameVec {
        let mut layer_name_vec = LayerNameVec::new();
        if debug::ENABLE_VALIDATION_LAYERS {
            layer_name_vec
                .layer_names
                .extend(debug::debug_layer_names().layer_names);
        }
        layer_name_vec
    }

    fn check_required_layers_supported(entry: &ash::Entry) {
        let layer_name_vec = Self::required_layers();
        for layer_name in layer_name_vec.layer_names.iter() {
            let all_layers_supported = entry
                .enumerate_instance_layer_properties()
                .expect("Couldn't enumerate instance layer properties")
                .iter()
                .any(|layer| {
                    let name = unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) };
                    let name = name.to_str().expect("Failed to get layer name pointer");
                    (*layer_name).name() == name
                });

            if !all_layers_supported {
                panic!("Validation layer not supported: {}", layer_name.name());
            }
        }
    }
}

impl ApplicationDescription for Instance {
    const APPLICATION_NAME: &'static str = "Vulkan Tutorial";
    const APPLICATION_VERSION: u32 = vk_make_version!(1, 0, 0);
    const API_VERSION: u32 = vk_make_version!(1, 0, 0);
    const ENGINE_VERSION: u32 = vk_make_version!(1, 0, 0);
    const ENGINE_NAME: &'static str = "Sepia Engine";
}

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}

pub struct VulkanContext {
    physical_device: PhysicalDevice,
    physical_device_memory_properties: ash::vk::PhysicalDeviceMemoryProperties,
    logical_device: ash::Device,
    surface: Surface,
    surface_khr: SurfaceKHR,
    debug_messenger: Option<(DebugUtils, DebugUtilsMessengerEXT)>,
    graphics_queue_family_index: u32,
    present_queue_family_index: u32,
    instance: Instance,
}

impl VulkanContext {
    pub fn new(window: &winit::Window) -> Self {
        let instance = Instance::new().unwrap();

        // Create the window surface
        let surface = Surface::new(instance.entry(), instance.instance());
        let surface_khr = unsafe {
            surface::create_surface(instance.entry(), instance.instance(), window)
                .expect("Failed to create window surface!")
        };

        let debug_messenger = debug::setup_debug_messenger(instance.entry(), instance.instance());

        let physical_device =
            Self::pick_physical_device(instance.instance(), &surface, surface_khr);

        // Get the memory properties of the physical device
        let physical_device_memory_properties = unsafe {
            instance
                .instance()
                .get_physical_device_memory_properties(physical_device)
        };

        let (graphics_queue_family_index, present_queue_family_index) =
            Self::find_queue_family_indices(
                instance.instance(),
                physical_device,
                &surface,
                surface_khr,
            );

        let (graphics_queue_family_index, present_queue_family_index) = (
            graphics_queue_family_index.expect("Failed to find a graphics queue family"),
            present_queue_family_index.expect("Failed to find a present queue family"),
        );

        // Need to dedup since the graphics family and presentation family
        // can have the same queue family index and
        // Vulkan does not allow passing an array containing duplicated family
        // indices.
        let mut queue_family_indices =
            vec![graphics_queue_family_index, present_queue_family_index];
        queue_family_indices.dedup();

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
                .create_device(physical_device, &device_create_info, None)
                .expect("Failed to create logical device.")
        };

        VulkanContext {
            instance,
            physical_device,
            physical_device_memory_properties,
            logical_device,
            surface,
            surface_khr,
            debug_messenger,
            graphics_queue_family_index,
            present_queue_family_index,
        }
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

    pub fn instance(&self) -> &ash::Instance {
        self.instance.instance()
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

    pub fn physical_device_memory_properties(&self) -> &ash::vk::PhysicalDeviceMemoryProperties {
        &self.physical_device_memory_properties
    }

    pub fn logical_device(&self) -> &ash::Device {
        &self.logical_device
    }

    pub fn graphics_queue_family_index(&self) -> u32 {
        self.graphics_queue_family_index
    }

    pub fn present_queue_family_index(&self) -> u32 {
        self.present_queue_family_index
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.surface.destroy_surface(self.surface_khr, None);
            self.logical_device.destroy_device(None);
            if let Some((debug_utils, messenger)) = &mut self.debug_messenger {
                debug_utils.destroy_debug_utils_messenger(*messenger, None);
            }
        }
    }
}
