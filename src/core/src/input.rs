use nalgebra_glm as glm;
use std::collections::HashMap;
use winit::event::{ElementState, VirtualKeyCode};

pub type KeyMap = HashMap<VirtualKeyCode, ElementState>;

#[derive(Default)]
pub struct Input {
    pub keystates: KeyMap,
    pub mouse: Mouse,
}

pub struct Mouse {
    pub is_left_clicked: bool,
    pub is_right_clicked: bool,
    pub position: glm::Vec2,
    pub position_delta: glm::Vec2,
    pub offset_from_center: glm::Vec2,
    pub wheel_delta: f32,
}

impl Default for Mouse {
    fn default() -> Self {
        Self {
            is_left_clicked: false,
            is_right_clicked: false,
            position: glm::vec2(0.0, 0.0),
            position_delta: glm::vec2(0.0, 0.0),
            offset_from_center: glm::vec2(0.0, 0.0),
            wheel_delta: 0.0,
        }
    }
}

impl Input {
    pub fn is_key_pressed(&self, keycode: VirtualKeyCode) -> bool {
        self.keystates.contains_key(&keycode) && self.keystates[&keycode] == ElementState::Pressed
    }
}
