use crate::render::Renderer;
use specs::prelude::*;
use std::{collections::HashMap, time::Instant};
use winit::{
    dpi::LogicalSize, ElementState, Event, EventsLoop, VirtualKeyCode, Window, WindowEvent,
};

#[derive(Default)]
struct ExitRequested(bool);

#[derive(Default)]
struct Input(InputState);

type KeyMap = HashMap<VirtualKeyCode, ElementState>;

#[derive(Default)]
struct InputState {
    keystates: KeyMap,
}

// Create a system that does something if a certain key is pressed
fn is_key_pressed(keystates: &KeyMap, keycode: VirtualKeyCode) -> bool {
    keystates.contains_key(&keycode) && keystates[&keycode] == ElementState::Pressed
}

#[derive(Default)]
struct EventSystem;

impl<'a> System<'a> for EventSystem {
    type SystemData = Read<'a, Input>;

    fn run(&mut self, input_data: Self::SystemData) {
        let input = &input_data.0;
        if is_key_pressed(&input.keystates, VirtualKeyCode::Space) {
            // TODO: Do something with the spacebar
        }
    }

    fn setup(&mut self, world: &mut World) {
        Self::SystemData::setup(world);
    }
}

pub struct App {
    event_loop: EventsLoop,
    window: Window,
    should_exit: bool,
}

impl App {
    pub fn new(width: u32, height: u32, title: &str) -> Self {
        log::debug!("Initializing application.");
        let event_loop = EventsLoop::new();
        let window = winit::WindowBuilder::new()
            .with_title(title)
            .with_dimensions((width, height).into())
            .build(&event_loop)
            .expect("Failed to create window.");

        App {
            event_loop,
            window,
            should_exit: false,
        }
    }

    pub fn run(&mut self) {
        log::debug!("Running application.");

        let mut renderer = Renderer::new(&self.window);

        let mut world = World::new();
        let mut dispatcher = DispatcherBuilder::new()
            .with(EventSystem, "event_system", &[])
            .build();
        dispatcher.setup(&mut world);

        let start_time = Instant::now();
        loop {
            self.process_events(&mut world);

            dispatcher.dispatch(&world);

            if self.should_exit {
                break;
            }

            renderer.step(start_time);
        }

        renderer.wait_idle();
    }

    fn process_events(&mut self, world: &mut World) {
        let mut should_exit = false;
        self.event_loop.poll_events(|event| match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                should_exit = true;
            }
            Event::WindowEvent {
                event:
                    WindowEvent::KeyboardInput {
                        input:
                            winit::KeyboardInput {
                                virtual_keycode: Some(keycode),
                                state,
                                ..
                            },
                        ..
                    },
                ..
            } => {
                let input_state = &mut world.write_resource::<Input>().0;
                if keycode == VirtualKeyCode::Escape {
                    should_exit = true;
                }
                *input_state.keystates.entry(keycode).or_insert(state) = state;
            }
            Event::WindowEvent {
                event:
                    WindowEvent::Resized(LogicalSize {
                        width: _width,
                        height: _height,
                    }),
                ..
            } => {
                // TODO: Handle resizing
            }
            _ => {}
        });
        self.should_exit = should_exit;
    }
}
