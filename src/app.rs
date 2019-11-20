use crate::platform;
use ash::{
    extensions::ext::DebugUtils,
    version::{EntryV1_0, InstanceV1_0},
    vk::{
        self,
        Bool32, DebugUtilsMessageSeverityFlagsEXT, DebugUtilsMessageTypeFlagsEXT,
        DebugUtilsMessengerCallbackDataEXT,
        DebugUtilsMessengerEXT,
        PhysicalDevice
    },
    vk_make_version, Entry, Instance,
};
use std::{
    error::Error,
    ffi::CString,
    ffi::{CStr},
    os::raw::{c_void},
    result,
};
//use winit::{ControlFlow, Event, EventsLoop, VirtualKeyCode, WindowEvent};

#[cfg(debug_assertions)]
const ENABLE_VALIDATION_LAYERS: bool = true;

#[cfg(not(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = false;

const REQUIRED_LAYERS: [&'static str; 1] = ["VK_LAYER_LUNARG_standard_validation"];

const WINDOW_TITLE: &str = "Vulkan Tutorial";
const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;
const APPLICATION_VERSION: u32 = vk_make_version!(1, 0, 0);
const API_VERSION: u32 = vk_make_version!(1, 0, 0);
const ENGINE_VERSION: u32 = vk_make_version!(1, 0, 0);
const ENGINE_NAME: &str = "Sepia Engine";

type Result<T> = result::Result<T, Box<dyn Error>>;

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: DebugUtilsMessageSeverityFlagsEXT,
    message_types: DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const DebugUtilsMessengerCallbackDataEXT,
    p_user_data: *mut c_void,
) -> Bool32 {
    log::debug!("Validation layer: {:?}", CStr::from_ptr((*p_callback_data).p_message));
    vk::FALSE
}

pub struct VulkanApp {
    entry: Entry,
    pub instance: ash::Instance,
    debug_utils: Option<(DebugUtils, DebugUtilsMessengerEXT)>,
    _physical_device: vk::PhysicalDevice,
    // pub events_loop: EventsLoop,
    // pub _window: winit::Window,
}

impl VulkanApp {
    pub fn new() -> Result<Self> {
        log::debug!("Creating application");
        // let events_loop = EventsLoop::new();
        // let window = VulkanApp::init_window(&events_loop);

        let (entry, instance) = Self::create_instance()?;
        let debug_utils = Self::setup_debug_messenger(&entry, &instance)?;
        let physical_device = Self::pick_physical_device(&instance).expect("Couldn't pick a suitable physical device!");

        Ok(Self {
            entry,
            instance,
            debug_utils,
            _physical_device: physical_device,
            // events_loop,
            // _window: window,
        })
    }

    fn create_instance() -> Result<(Entry, ash::Instance)> {
        let entry = ash::Entry::new()?;

        let app_name = CString::new(WINDOW_TITLE)?;
        let engine_name = CString::new(ENGINE_NAME)?;

        let app_info = vk::ApplicationInfo::builder()
            .application_name(&app_name)
            .engine_name(&engine_name)
            .api_version(API_VERSION)
            .application_version(APPLICATION_VERSION)
            .engine_version(ENGINE_VERSION)
            .build();

        let mut extension_names = platform::required_extension_names();
        if ENABLE_VALIDATION_LAYERS {
            extension_names.push(DebugUtils::name().as_ptr());
        }

        let layer_names = REQUIRED_LAYERS
        .iter()
        .map(|name| CString::new(*name).expect("Failed to build CString"))
        .collect::<Vec<_>>();

        let layer_names_ptrs = layer_names.iter()
        .map(|name| name.as_ptr())
        .collect::<Vec<_>>();

        let mut instance_create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names);

        if ENABLE_VALIDATION_LAYERS {
            Self::check_validation_layer_support(&entry);
            instance_create_info = instance_create_info.enabled_layer_names(&layer_names_ptrs);
        }

        let instance = unsafe { entry.create_instance(&instance_create_info, None).unwrap() };
        Ok((entry, instance))
    }

    fn check_validation_layer_support(entry: &Entry) {
        for required in REQUIRED_LAYERS.iter() {
            let found = entry
            .enumerate_instance_layer_properties()
            .unwrap()
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
    }

    fn setup_debug_messenger(
        entry: &Entry,
        instance: &Instance,
    ) -> Result<Option<(DebugUtils, vk::DebugUtilsMessengerEXT)>> {
        if !ENABLE_VALIDATION_LAYERS {
            return Ok(None);
        }
        let create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .flags(vk::DebugUtilsMessengerCreateFlagsEXT::all())
            .pfn_user_callback(Some(vulkan_debug_callback))
            .build();
        let debug_utils = DebugUtils::new(entry, instance);
        let messenger =
            unsafe { debug_utils.create_debug_utils_messenger(&create_info, None)? };
        Ok(Some((debug_utils, messenger)))
    }

    fn pick_physical_device(instance: &Instance) -> Option<PhysicalDevice> {
        let devices = unsafe {
            instance.enumerate_physical_devices().expect("Couldn't get physical devices")
        };
        let device = devices
        .into_iter()
        .find(|device| Self::is_device_suitable(instance, *device));

        device
    }

    fn is_device_suitable(instance: &Instance, device: PhysicalDevice) -> bool {
        Self::find_queue_families(instance, device).is_some()
    }

    fn find_queue_families(instance: &Instance, device: PhysicalDevice) -> Option<usize> {
        let props = unsafe { instance.get_physical_device_queue_family_properties(device) };
        props
        .iter()
        .enumerate()
        .find(|(_, family)| {
            family.queue_count > 0 && family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
        })
        .map(|(index, _)| index)
    }

    // fn init_window(events_loop: &EventsLoop) -> winit::Window {
    //     winit::WindowBuilder::new()
    //         .with_title(WINDOW_TITLE)
    //         .with_dimensions((WINDOW_WIDTH, WINDOW_HEIGHT).into())
    //         .build(events_loop)
    //         .expect("Failed to create window.")
    // }

    pub fn run(&mut self) {
        log::info!("Running application");
        // self.events_loop.run_forever(|event| match event {
        //     Event::WindowEvent { event, .. } => match event {
        //         WindowEvent::KeyboardInput { input, .. } => {
        //             if let Some(VirtualKeyCode::Escape) = input.virtual_keycode {
        //                 ControlFlow::Break
        //             } else {
        //                 ControlFlow::Continue
        //             }
        //         }
        //         WindowEvent::CloseRequested => ControlFlow::Break,
        //         _ => ControlFlow::Continue,
        //     },
        //     _ => ControlFlow::Continue,
        // });
    }
}

impl Drop for VulkanApp {
    fn drop(&mut self) {
        log::debug!("Dropping application");
        unsafe {
            if let Some((debug_utils, messenger)) = self.debug_utils.take() {
                debug_utils.destroy_debug_utils_messenger(messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}
