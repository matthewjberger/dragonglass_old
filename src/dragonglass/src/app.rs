use dragonglass_backend_vulkan::render::{
    component::{GltfAssetComponent, TransformComponent},
    renderer::Renderer,
    system::{PrepareRendererSystem, RenderSystem, TransformationSystem},
};
use nalgebra_glm as glm;
use specs::prelude::*;
use std::collections::HashMap;
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

        let renderer = Renderer::new(&self.window);
        let mut world = World::new();

        // Resource fetching will panic without this
        // because of the WriteExpect and ReadExpect lookups
        // on the render system
        world.insert(renderer);

        // Register the render preparation system
        // and its components
        let mut render_preparation_dispatcher = DispatcherBuilder::new()
            .with_thread_local(PrepareRendererSystem)
            .build();
        render_preparation_dispatcher.setup(&mut world);

        // Register the main systems and their components
        let mut system_dispatcher = DispatcherBuilder::new()
            .with(EventSystem, "event_system", &[])
            .with(TransformationSystem, "transformation_system", &[])
            .with_thread_local(RenderSystem)
            .build();
        system_dispatcher.setup(&mut world);

        // Add renderable entities
        let scale = 600.0;
        world
            .create_entity()
            .with(GltfAssetComponent {
                asset_name: "examples/assets/models/FlightHelmet/gltf/FlightHelmet.gltf"
                    .to_string(),
            })
            .with(TransformComponent {
                scale: glm::scale(&glm::Mat4::identity(), &glm::vec3(scale, scale, scale)),
                ..Default::default()
            })
            .build();

        // Prepare the renderer
        // by creating vertex buffers, index buffers, and command buffers
        render_preparation_dispatcher.dispatch(&world);

        loop {
            self.process_events(&mut world);

            if self.should_exit {
                break;
            }

            system_dispatcher.dispatch(&world);
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
