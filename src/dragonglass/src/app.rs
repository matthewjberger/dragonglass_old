use dragonglass_backend_vulkan::render::{renderer::Renderer, system::render_system, GltfPipeline};
use dragonglass_core::{
    camera::{fps_camera_key_system, fps_camera_mouse_system, Camera, CameraViewMatrix},
    components::{AssetName, Transform},
    input::Input,
};
use legion::prelude::*;
use nalgebra_glm as glm;
use winit::{
    dpi::{LogicalPosition, LogicalSize},
    ElementState, Event, EventsLoop, MouseButton, VirtualKeyCode, Window, WindowEvent,
};

#[derive(Default)]
struct ExitRequested(bool);

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
        window
            .grab_cursor(true)
            .expect("Failed to set cursor grabbing on window!");

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

        world
            .resources
            .insert(CameraViewMatrix(glm::Mat4::identity()));

        // Register the render preparation system and its components
        let prepare_renderer_system = SystemBuilder::new("prepare_renderer")
            .write_resource::<Renderer>()
            .with_query(<Read<AssetName>>::query())
            .build(|_, mut world, mut renderer, query| {
                let asset_names = query
                    .iter(&mut world)
                    .map(|asset_name| asset_name.0.to_string())
                    .collect::<Vec<_>>();
                let pipeline_gltf = GltfPipeline::new(&mut renderer, &asset_names);
                renderer.pipeline_gltf = Some(pipeline_gltf);
            });
        let mut prepare_schedule = Schedule::builder()
            .add_system(prepare_renderer_system)
            .build();

        let mut schedule = Schedule::builder()
            .add_system(fps_camera_mouse_system())
            .add_system(fps_camera_key_system())
            .flush()
            // More game simulation systems can go here
            .add_thread_local(render_system())
            .build();

        world.insert((), vec![(Camera::default(),)]);
        world.insert(
            (),
            vec![(
                AssetName("examples/assets/models/Sponza/Sponza.gltf".to_string()),
                Transform::default(),
            )],
        );

        prepare_schedule.execute(&mut world);

        loop {
            self.process_events(&mut world);

            if self.should_exit {
                break;
            }

            let window_size = self
                .window
                .get_inner_size()
                .expect("Failed to get window inner size!");
            self.window
                .set_cursor_position(LogicalPosition::new(
                    window_size.width / 2.0,
                    window_size.height / 2.0,
                ))
                .expect("Failed to set cursor position!");
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
        let window_size = self
            .window
            .get_inner_size()
            .expect("Failed to get window inner size!");

        self.event_loop.poll_events(|event| match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    should_exit = true;
                }
                WindowEvent::KeyboardInput {
                    input:
                        winit::KeyboardInput {
                            virtual_keycode: Some(keycode),
                            state,
                            ..
                        },
                    ..
                } => {
                    if keycode == VirtualKeyCode::Escape {
                        should_exit = true;
                    }
                    *input.keystates.entry(keycode).or_insert(state) = state;
                }
                WindowEvent::Resized(LogicalSize {
                    width: _width,
                    height: _height,
                }) => {
                    // TODO: Handle resizing
                }
                WindowEvent::MouseInput { button, state, .. } => {
                    let clicked = state == ElementState::Pressed;
                    match button {
                        MouseButton::Left => input.mouse.is_left_clicked = clicked,
                        MouseButton::Right => input.mouse.is_right_clicked = clicked,
                        _ => {}
                    }
                }
                WindowEvent::CursorMoved { position, .. } => {
                    input.mouse.position = glm::vec2(position.x as _, position.y as _);
                    input.mouse.offset_from_center = glm::vec2(
                        ((window_size.width / 2.0) - position.x) as _,
                        ((window_size.height / 2.0) - position.y) as _,
                    );
                }
                _ => {}
            },
            _ => {}
        });
        self.should_exit = should_exit;
    }
}
