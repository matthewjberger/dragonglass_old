use ash::{
    extensions::{
        ext::DebugUtils,
        khr::{Surface, Swapchain},
    },
    version::{DeviceV1_0, EntryV1_0, InstanceV1_0},
    vk, vk_make_version,
};
use std::{
    ffi::{CStr, CString},
    mem,
};
use winit::{dpi::LogicalSize, Event, EventsLoop, VirtualKeyCode, Window, WindowEvent};

pub struct App {
    pub event_loop: EventsLoop,
    pub window: Window,
}

impl App {
    pub fn new(width: u32, height: u32, title: &str) -> Self {
        // Initialize the window
        let mut event_loop = EventsLoop::new();
        let window = winit::WindowBuilder::new()
            .with_title(title)
            .with_dimensions((width, height).into())
            .build(&event_loop)
            .expect("Failed to create window.");

        App { event_loop, window }
    }

    pub fn run<F>(&mut self, mut handler: F)
    where
        F: FnMut(),
    {
        log::debug!("Running application.");
        let mut should_stop = false;
        loop {
            self.event_loop.poll_events(|event| match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                }
                | Event::WindowEvent {
                    event:
                        WindowEvent::KeyboardInput {
                            input:
                                winit::KeyboardInput {
                                    virtual_keycode: Some(VirtualKeyCode::Escape),
                                    ..
                                },
                            ..
                        },
                    ..
                } => should_stop = true,
                Event::WindowEvent {
                    event:
                        WindowEvent::Resized(LogicalSize {
                            width: _width,
                            height: _height,
                        }),
                    ..
                } => {
                    // TODO: Handle resizing by recreating the swapchain
                }
                _ => {}
            });

            if should_stop {
                break;
            }

            handler();
        }
    }
}
