use crate::model::GltfAsset;
use crate::render::renderer::{
    MeshComponent, RenderSystem, Renderer, StartTime, TransformComponent, TransformationSystem,
};
use nalgebra_glm as glm;
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
        let start_time = Instant::now();

        // Load the models at this time
        // and pass them to the renderer to prepare
        // vertex buffers
        let gltf_asset = GltfAsset::from_file("assets/models/Duck/Duck.gltf");
        // let gltf_asset2 = GltfAsset::from_file("assets/models/Duck/Duck.gltf");
        let renderer = Renderer::new(&self.window, &gltf_asset);

        let mut world = World::new();
        let mut dispatcher = DispatcherBuilder::new()
            .with(EventSystem, "event_system", &[])
            .with(TransformationSystem, "transformation_system", &[])
            .with_thread_local(RenderSystem)
            .build();
        dispatcher.setup(&mut world);

        // Resource fetching will panic without these
        // because of the WriteExpect and ReadExpect lookups
        // on the render system
        world.insert(StartTime(start_time));
        world.insert(renderer);

        // Add an entity that uses the gltf asset
        // that the renderer prepared data buffers for
        world
            .create_entity()
            .with(MeshComponent { mesh: gltf_asset })
            .with(TransformComponent {
                rotate: glm::rotate(
                    &glm::Mat4::identity(),
                    90_f32.to_radians(),
                    &glm::vec3(0.0, 1.0, 1.0),
                ),
                ..Default::default()
            })
            .build();

        loop {
            self.process_events(&mut world);

            if self.should_exit {
                break;
            }

            dispatcher.dispatch(&world);
        }

        (*world.read_resource::<Renderer>()).wait_idle();
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
