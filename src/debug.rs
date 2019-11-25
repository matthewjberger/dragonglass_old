use ash::{
    extensions::ext::DebugUtils,
    version::{EntryV1_0, InstanceV1_0},
    vk::{
        self, Bool32, DebugUtilsMessageSeverityFlagsEXT, DebugUtilsMessageTypeFlagsEXT,
        DebugUtilsMessengerCallbackDataEXT, DebugUtilsMessengerEXT,
    },
};
use std::{ffi::CStr, os::raw::c_void};

// Enable validation layers only in debug mode
#[cfg(debug_assertions)]
pub const ENABLE_VALIDATION_LAYERS: bool = true;

#[cfg(not(debug_assertions))]
pub const ENABLE_VALIDATION_LAYERS: bool = false;

pub const REQUIRED_LAYERS: [&str; 1] = ["VK_LAYER_LUNARG_standard_validation"];

// Setup the callback for the debug utils extension
pub unsafe extern "system" fn vulkan_debug_callback(
    flags: DebugUtilsMessageSeverityFlagsEXT,
    type_flags: DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> Bool32 {
    let type_flag = if type_flags == DebugUtilsMessageTypeFlagsEXT::GENERAL {
        "General"
    } else if type_flags == DebugUtilsMessageTypeFlagsEXT::PERFORMANCE {
        "Performance"
    } else if type_flags == DebugUtilsMessageTypeFlagsEXT::VALIDATION {
        "Validation"
    } else {
        unreachable!()
    };

    let message = format!(
        "[{}] {:?}",
        type_flag,
        CStr::from_ptr((*p_callback_data).p_message)
    );

    if flags == DebugUtilsMessageSeverityFlagsEXT::ERROR {
        log::error!("{}", message);
    } else if flags == DebugUtilsMessageSeverityFlagsEXT::INFO {
        log::info!("{}", message);
    } else if flags == DebugUtilsMessageSeverityFlagsEXT::WARNING {
        log::warn!("{}", message);
    } else if flags == DebugUtilsMessageSeverityFlagsEXT::VERBOSE {
        log::trace!("{}", message);
    }
    vk::FALSE
}

pub fn setup_debug_messenger<E: EntryV1_0, I: InstanceV1_0>(
    entry: &E,
    instance: &I,
) -> Option<(DebugUtils, DebugUtilsMessengerEXT)> {
    if ENABLE_VALIDATION_LAYERS {
        let debug_utils = DebugUtils::new(entry, instance);
        let create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .flags(vk::DebugUtilsMessengerCreateFlagsEXT::all())
            .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
            .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
            .pfn_user_callback(Some(vulkan_debug_callback))
            .build();
        let messenger = unsafe {
            debug_utils
                .create_debug_utils_messenger(&create_info, None)
                .expect("Failed to create debug utils messenger")
        };
        Some((debug_utils, messenger))
    } else {
        None
    }
}
