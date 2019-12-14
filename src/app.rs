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
    _window: Window, // Needs to live as long the event loop
    renderer: Renderer,
    should_exit: bool,
    dimensions: Option<[u32; 2]>,
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
        let renderer = Renderer::new(&window, [width, height]);

        App {
            event_loop,
            _window: window,
            renderer,
            should_exit: false,
            dimensions: Some([width, height]),
        }
    }

    pub fn run(&mut self) {
        log::debug!("Running application.");

        let mut world = World::new();
        // world.insert(InputState::default());
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

            self.renderer.step(self.dimensions, start_time);
        }

        self.renderer.wait_idle();
    }

    fn process_events(&mut self, world: &mut World) {
        let mut dimensions: Option<[u32; 2]> = None;
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
                event: WindowEvent::Resized(LogicalSize { width, height }),
                ..
            } => {
                dimensions = Some([width as u32, height as u32]);
            }
            _ => {}
        });
        self.dimensions = dimensions;
        self.should_exit = should_exit;
        self.block_while_minimized(world);
    }

    fn block_while_minimized(&mut self, world: &mut World) {
        if let Some(dimensions) = self.dimensions {
            let is_minimized = dimensions[0] == 0 || dimensions[1] == 0;
            if is_minimized {
                loop {
                    self.process_events(world);
                    let is_minimized = dimensions[0] == 0 || dimensions[1] == 0;
                    if !is_minimized {
                        break;
                    }
                }
            }
        }
    }
}
