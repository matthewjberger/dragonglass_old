use crate::platform::*;
use ash::{
    version::{EntryV1_0, InstanceV1_0},
    vk, vk_make_version,
};
use std::ffi::CString;
use winit::{ControlFlow, Event, EventsLoop, VirtualKeyCode, WindowEvent};

const WINDOW_TITLE: &str = "Vulkan Tutorial";
const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;
const APPLICATION_VERSION: u32 = vk_make_version!(1, 0, 0);
const API_VERSION: u32 = vk_make_version!(1, 0, 0);
const ENGINE_VERSION: u32 = vk_make_version!(1, 0, 0);

pub struct VulkanApp {
    pub events_loop: EventsLoop,
    pub _window: winit::Window,
    pub instance: ash::Instance,
}

impl VulkanApp {
    pub fn new() -> VulkanApp {
        let events_loop = EventsLoop::new();
        let window = VulkanApp::init_window(&events_loop);

        let instance = VulkanApp::create_instance();

        VulkanApp {
            events_loop,
            _window: window,
            instance,
        }
    }

    fn init_window(events_loop: &EventsLoop) -> winit::Window {
        winit::WindowBuilder::new()
            .with_title(WINDOW_TITLE)
            .with_dimensions((WINDOW_WIDTH, WINDOW_HEIGHT).into())
            .build(events_loop)
            .expect("Failed to create window.")
    }

    pub fn run(&mut self) {
        self.events_loop.run_forever(|event| match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::KeyboardInput { input, .. } => {
                    if let Some(VirtualKeyCode::Escape) = input.virtual_keycode {
                        ControlFlow::Break
                    } else {
                        ControlFlow::Continue
                    }
                }
                WindowEvent::CloseRequested => ControlFlow::Break,
                _ => ControlFlow::Continue,
            },
            _ => ControlFlow::Continue,
        });
    }

    fn create_instance() -> ash::Instance {
        let entry = ash::Entry::new().unwrap();

        let app_name = CString::new(WINDOW_TITLE).unwrap();
        let engine_name = CString::new("Vulkan Engine").unwrap();

        let app_info = vk::ApplicationInfo::builder()
            .application_name(&app_name)
            .engine_name(&engine_name)
            .api_version(API_VERSION)
            .application_version(APPLICATION_VERSION)
            .engine_version(ENGINE_VERSION)
            .build();

        let extension_names = required_extension_names();

        let create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .flags(vk::InstanceCreateFlags::empty())
            .enabled_extension_names(&extension_names)
            .build();

        let instance: ash::Instance = unsafe {
            entry
                .create_instance(&create_info, None)
                .expect("Failed to create instance!")
        };

        instance
    }
}

impl Drop for VulkanApp {
    fn drop(&mut self) {
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}
