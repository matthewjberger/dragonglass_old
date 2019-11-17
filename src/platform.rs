use ash::extensions::{DebugReport, Surface};

#[cfg(target_os = "windows")]
use ash::extensions::Win32Surface;

#[cfg(target_os = "linux")]
use ash::extensions::XlibSurface;

#[cfg(target_os = "windows")]
pub fn required_extension_names() -> Vec<*const i8> {
    vec![
        Surface::name().as_ptr(),
        Win32Surface::name().as_ptr(),
        DebugReport::name().as_ptr(),
    ]
}

#[cfg(target_os = "linux")]
pub fn required_extension_names() -> Vec<*const i8> {
    vec![
        Surface::name().as_ptr(),
        XlibSurface::name().as_ptr(),
        DebugReport::name().as_ptr(),
    ]
}
