use std::collections::HashMap;
use winit::{ElementState, VirtualKeyCode};

pub type KeyMap = HashMap<VirtualKeyCode, ElementState>;

#[derive(Default)]
pub struct Input {
    pub keystates: KeyMap,
}

impl Input {
    pub fn is_key_pressed(&self, keycode: VirtualKeyCode) -> bool {
        self.keystates.contains_key(&keycode) && self.keystates[&keycode] == ElementState::Pressed
    }
}
