use dragonglass_backend_vulkan::render::{
    component::{GltfAssetComponent, TransformComponent},
    GltfPipeline,
    renderer::{Renderer},
    system::render_system,
};
use legion::prelude::*;
use nalgebra_glm as glm;
use std::collections::HashMap;
use winit::{
    dpi::LogicalSize, ElementState, Event, EventsLoop, VirtualKeyCode, Window, WindowEvent,
};

#[derive(Default)]
struct ExitRequested(bool);

type KeyMap = HashMap<VirtualKeyCode, ElementState>;

#[derive(Default)]
struct Input {
    keystates: KeyMap,
}

// Create a system that does something if a certain key is pressed
fn is_key_pressed(keystates: &KeyMap, keycode: VirtualKeyCode) -> bool {
    keystates.contains_key(&keycode) && keystates[&keycode] == ElementState::Pressed
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

        let mut world = World::new();

        let renderer = Renderer::new(&self.window);
        world.resources.insert(renderer);

        let input = Input::default();
        world.resources.insert(input);

        // Register the render preparation system and its components
        let prepare_renderer_system = SystemBuilder::new("prepare_renderer")
            .write_resource::<Renderer>()
            .with_query(<Read<GltfAssetComponent>>::query())
            .build(|_, mut world, mut renderer, query| {
                let asset_names = query.iter(&mut world).map(|asset| asset.asset_name.to_string()).collect::<Vec<_>>();
                let pipeline_gltf = GltfPipeline::new(&mut renderer, &asset_names);
                renderer.pipeline_gltf = Some(pipeline_gltf);
            });
        let mut prepare_schedule = Schedule::builder()
            .add_system(prepare_renderer_system)
            .build();

        // Setup game loop systems
        let event_system = SystemBuilder::new("event")
            .write_resource::<Input>()
            .with_query(<Read<TransformComponent>>::query())
            .build(|_, _, input, _| {
                if is_key_pressed(&input.keystates, VirtualKeyCode::Space) {
                    // TODO: Do something with the spacebar
                }
            });

        let transformation_system = SystemBuilder::new("transformation")
            .with_query(<Write<TransformComponent>>::query())
            .build(|_, mut world, _, query| {
                for mut transform in query.iter(&mut world) {
                    transform.rotate = glm::rotate(
                        &transform.rotate,
                        0.1_f32.to_radians(),
                        &glm::vec3(0.0, 1.0, 0.0),
                    );
                }
            });

        let mut schedule = Schedule::builder()
            .add_system(event_system)
            .flush()
            .add_system(transformation_system)
            // More game simulation systems can go here
            .flush()
            .add_thread_local(render_system())
            .build();

        // Add renderable entities
        world.insert(
            (),
            vec![(
                GltfAssetComponent {
                    //asset_name: "examples/assets/models/FlightHelmet/glTF/FlightHelmet.gltf"
                    asset_name: "examples/assets/models/Buggy.glb"
                        .to_string(),
                },
                TransformComponent::default(),
            )],
        );

        prepare_schedule.execute(&mut world);

        loop {
            self.process_events(&mut world);

            if self.should_exit {
                break;
            }

            schedule.execute(&mut world);
        }

        let renderer = world
            .resources
            .get::<Renderer>()
            .expect("Failed to get renderer resource!");
        renderer.wait_idle();
    }

    fn process_events(&mut self, world: &mut World) {
        let mut input = world
            .resources
            .get_mut::<Input>()
            .expect("Failed to get input resource!");
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
                if keycode == VirtualKeyCode::Escape {
                    should_exit = true;
                }
                *input.keystates.entry(keycode).or_insert(state) = state;
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
