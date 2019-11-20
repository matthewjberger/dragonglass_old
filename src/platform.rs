use ash::extensions::khr::Surface;

#[cfg(target_os = "windows")]
use ash::extensions::khr::Win32Surface;

#[cfg(target_os = "linux")]
use ash::extensions::khr::XlibSurface;

#[cfg(target_os = "windows")]
pub fn required_extension_names() -> Vec<*const i8> {
    vec![Surface::name().as_ptr(), Win32Surface::name().as_ptr()]
}

#[cfg(target_os = "linux")]
pub fn required_extension_names() -> Vec<*const i8> {
    vec![Surface::name().as_ptr(), XlibSurface::name().as_ptr()]
}
