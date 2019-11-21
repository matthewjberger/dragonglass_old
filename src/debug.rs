use crate::error::Result;
use ash::{
    extensions::ext::DebugUtils,
    version::EntryV1_0,
    vk::{
        self, Bool32, DebugUtilsMessageSeverityFlagsEXT, DebugUtilsMessageTypeFlagsEXT,
        DebugUtilsMessengerCallbackDataEXT,
    },
    Entry, Instance,
};
use std::{ffi::CStr, ffi::CString, os::raw::c_void};
//use winit::{ControlFlow, Event, EventsLoop, VirtualKeyCode, WindowEvent};

#[cfg(debug_assertions)]
pub const ENABLE_VALIDATION_LAYERS: bool = true;

#[cfg(not(debug_assertions))]
pub const ENABLE_VALIDATION_LAYERS: bool = false;

pub const REQUIRED_LAYERS: [&str; 1] = ["VK_LAYER_LUNARG_standard_validation"];

unsafe extern "system" fn vulkan_debug_callback(
    _: DebugUtilsMessageSeverityFlagsEXT,
    _: DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> Bool32 {
    log::debug!(
        "Validation layer: {:?}",
        CStr::from_ptr((*p_callback_data).p_message)
    );
    vk::FALSE
}

/// Get the pointers to the validation layer names
pub fn get_layer_names_and_pointers() -> (Vec<CString>, Vec<*const i8>) {
    let layer_names = REQUIRED_LAYERS
        .iter()
        .map(|name| CString::new(*name).expect("Failed to build CString"))
        .collect::<Vec<_>>();

    let layer_name_ptrs = layer_names
        .iter()
        .map(|name| name.as_ptr())
        .collect::<Vec<_>>();
    (layer_names, layer_name_ptrs)
}

/// Check if the required validation layers are supported
/// by the provided Vulkan instance
pub fn check_validation_layer_support(entry: &Entry) {
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

/// Setup the debug messenger callback if validation layers
/// are enabled
pub fn setup_debug_messenger(
    entry: &Entry,
    instance: &Instance,
) -> Result<Option<(DebugUtils, vk::DebugUtilsMessengerEXT)>> {
    if !ENABLE_VALIDATION_LAYERS {
        return Ok(None);
    }
    let create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .flags(vk::DebugUtilsMessengerCreateFlagsEXT::all())
        .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
        .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
        .pfn_user_callback(Some(vulkan_debug_callback))
        .build();
    let debug_utils = DebugUtils::new(entry, instance);
    let messenger = unsafe { debug_utils.create_debug_utils_messenger(&create_info, None)? };
    Ok(Some((debug_utils, messenger)))
}
