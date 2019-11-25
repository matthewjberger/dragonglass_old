use ash::vk::{
    self, Bool32, DebugUtilsMessageSeverityFlagsEXT, DebugUtilsMessageTypeFlagsEXT,
    DebugUtilsMessengerCallbackDataEXT,
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
