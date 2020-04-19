use crate::core::Instance;
use ash::{
    extensions::khr::Surface as AshSurface,
    version::{EntryV1_0, InstanceV1_0},
    vk,
    vk::SurfaceKHR,
};
use winit::window::Window;

pub struct Surface {
    surface: AshSurface,
    surface_khr: SurfaceKHR,
}

impl Surface {
    pub fn new(instance: &Instance, window: &Window) -> Self {
        let surface = AshSurface::new(instance.entry(), instance.instance());
        let surface_khr = unsafe {
            create_surface(instance.entry(), instance.instance(), window)
                .expect("Failed to create window surface!")
        };

        Surface {
            surface,
            surface_khr,
        }
    }

    pub fn surface(&self) -> &AshSurface {
        &self.surface
    }

    pub fn surface_khr(&self) -> ash::vk::SurfaceKHR {
        self.surface_khr
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe {
            self.surface.destroy_surface(self.surface_khr, None);
        }
    }
}

#[cfg(target_os = "windows")]
pub fn surface_extension_names() -> Vec<*const i8> {
    use ash::extensions::khr::Win32Surface;
    vec![
        ash::extensions::khr::Surface::name().as_ptr(),
        Win32Surface::name().as_ptr(),
    ]
}

#[cfg(target_os = "linux")]
pub fn surface_extension_names() -> Vec<*const i8> {
    use ash::extensions::khr::XlibSurface;
    vec![
        ash::extensions::khr::Surface::name().as_ptr(),
        XlibSurface::name().as_ptr(),
    ]
}

#[cfg(target_os = "windows")]
unsafe fn create_surface<E: EntryV1_0, I: InstanceV1_0>(
    entry: &E,
    instance: &I,
    window: &Window,
) -> Result<vk::SurfaceKHR, vk::Result> {
    use ash::extensions::khr::Win32Surface;
    use raw_window_handle::{RawWindowHandle, HasRawWindowHandle};

    let handle = 
    match window.raw_window_handle() {
        RawWindowHandle::Windows(handle) => handle,
        _ => panic!("Failed to get windows handle!")
    };

    let win32_create_info = vk::Win32SurfaceCreateInfoKHR::builder()
    .hwnd(handle.hwnd)
    .hinstance(handle.hinstance)
    .build();
    let win32_surface_loader = Win32Surface::new(entry, instance);
    win32_surface_loader.create_win32_surface(&win32_create_info, None)
}

#[cfg(target_os = "linux")]
unsafe fn create_surface<E: EntryV1_0, I: InstanceV1_0>(
    entry: &E,
    instance: &I,
    window: &Window,
) -> Result<vk::SurfaceKHR, vk::Result> {
    use ash::extensions::khr::XlibSurface;
    use winit::platform::unix::WindowExtUnix;

    let x11_display = window.xlib_display().expect("Failed to get xlib display!");
    let x11_window = window.xlib_window().expect("Failed to get xlib window!");
    let x11_create_info = vk::XlibSurfaceCreateInfoKHR::builder()
        .window(x11_window)
        .dpy(x11_display as *mut vk::Display);

    let xlib_surface_loader = XlibSurface::new(entry, instance);
    xlib_surface_loader.create_xlib_surface(&x11_create_info, None)
}
