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
use winit::{dpi::LogicalSize, Event, EventsLoop, VirtualKeyCode, WindowEvent};

struct App;

impl App {
    fn new() -> Self {}
}
