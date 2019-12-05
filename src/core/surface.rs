use ash::{extensions::khr::Surface as AshSurface, vk::SurfaceKHR};

use crate::{core::Instance, surface};

pub struct Surface {
    surface: AshSurface,
    surface_khr: SurfaceKHR,
}

impl Surface {
    pub fn new(instance: &Instance, window: &winit::Window) -> Self {
        let surface = AshSurface::new(instance.entry(), instance.instance());
        let surface_khr = unsafe {
            surface::create_surface(instance.entry(), instance.instance(), window)
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
