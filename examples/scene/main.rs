use dragonglass_backend_vulkan::{
    render::Renderer,
    systems::render::{animation_system, prepare_renderer_system, reload_system, render_system},
};
use dragonglass_core::{
    camera::{orbital_camera_mouse_system, Camera, CameraState},
    components::{AssetName, Transform},
    input::Input,
    AnimationState, AppState, DeltaTime,
};
use legion::prelude::*;
use nalgebra_glm as glm;
use std::time::Instant;
use winit::{
    dpi::{LogicalSize, PhysicalSize},
    event::{
        ElementState, Event, KeyboardInput, MouseButton, MouseScrollDelta, VirtualKeyCode,
        WindowEvent,
    },
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

fn main() {
    let system = System::new(1920, 1080, "Dragonglass - Vulkan Rendering");
    system.run();
}

pub struct System {
    event_loop: EventLoop<()>,
    window: Window,
    world: World,
}

impl System {
    pub fn new(width: u32, height: u32, title: &str) -> Self {
        env_logger::init();
        let event_loop = EventLoop::new();
        let window = WindowBuilder::new()
            .with_title(title)
            .with_inner_size(LogicalSize::new(width, height))
            .build(&event_loop)
            .expect("Failed to create window.");

        log::debug!("Running application.");

        // stow_cursor();

        let mut world = World::new();

        let renderer = Renderer::new(&window);
        world.resources.insert(renderer);

        let input = Input::default();
        world.resources.insert(input);

        world.resources.insert(CameraState::default());

        world.resources.insert(DeltaTime(0 as _));

        world.resources.insert(AppState::default());

        // Register the render preparation system and its components
        let mut prepare_schedule = Schedule::builder()
            .add_system(prepare_renderer_system())
            .build();

        let camera = Camera {
            speed: 5.0,
            ..Default::default()
        };
        world.insert((), vec![(camera,)]);
        world.insert(
            (),
            vec![(
                // AssetName("examples/assets/models/Sponza/Sponza.gltf".to_string()),
                // AssetName("examples/assets/models/BoxAnimated.glb".to_string()),
                AssetName("examples/assets/models/DamagedHelmet.glb".to_string()),
                AnimationState { time: 0.0 },
                Transform::default(),
            )],
        );

        prepare_schedule.execute(&mut world);

        Self {
            event_loop,
            window,
            world,
        }
    }

    pub fn run(self) {
        let System {
            event_loop,
            window,
            mut world,
        } = self;
        let mut last_frame = Instant::now();

        let mut schedule = Schedule::builder()
            .add_system(orbital_camera_mouse_system())
            .add_system(animation_system())
            .add_system(reload_system())
            .flush()
            // More game simulation systems can go here
            .add_thread_local(render_system())
            .build();

        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Poll;
            let window_size = window.inner_size();
            let mut cursor_moved = false;

            {
                let mut input = world
                    .resources
                    .get_mut::<Input>()
                    .expect("Failed to get input resource!");
                input.mouse.wheel_delta = 0.0;
            }

            if let Event::WindowEvent { event, .. } = event {
                match event {
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                virtual_keycode: Some(keycode),
                                state,
                                ..
                            },
                        ..
                    } => {
                        if keycode == VirtualKeyCode::Escape {
                            *control_flow = ControlFlow::Exit;
                        }
                        let mut input = world
                            .resources
                            .get_mut::<Input>()
                            .expect("Failed to get input resource!");
                        *input.keystates.entry(keycode).or_insert(state) = state;
                    }
                    WindowEvent::Resized(PhysicalSize { width, height }) => {
                        let mut app_state = world
                            .resources
                            .get_mut::<AppState>()
                            .expect("Failed to get input resource!");
                        app_state.window.width = width as u32;
                        app_state.window.height = height as u32;
                    }
                    WindowEvent::MouseInput { button, state, .. } => {
                        let mut input = world
                            .resources
                            .get_mut::<Input>()
                            .expect("Failed to get input resource!");
                        let clicked = state == ElementState::Pressed;
                        match button {
                            MouseButton::Left => input.mouse.is_left_clicked = clicked,
                            MouseButton::Right => input.mouse.is_right_clicked = clicked,
                            _ => {}
                        }
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        let mut input = world
                            .resources
                            .get_mut::<Input>()
                            .expect("Failed to get input resource!");
                        let last_position = input.mouse.position;
                        let current_position = glm::vec2(position.x as _, position.y as _);
                        input.mouse.position = current_position;
                        input.mouse.position_delta = current_position - last_position;
                        input.mouse.offset_from_center = glm::vec2(
                            ((window_size.width as f64 / 2.0) - position.x) as _,
                            ((window_size.height as f64 / 2.0) - position.y) as _,
                        );
                        cursor_moved = true;
                    }
                    WindowEvent::MouseWheel {
                        delta: MouseScrollDelta::LineDelta(_, v_lines),
                        ..
                    } => {
                        let mut input = world
                            .resources
                            .get_mut::<Input>()
                            .expect("Failed to get input resource!");
                        input.mouse.wheel_delta = v_lines;
                    }
                    WindowEvent::DroppedFile(file_pathbuf) => {
                        println!("Received file: {:?}", file_pathbuf);
                    }
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                    _ => {}
                }

                {
                    let mut input = world
                        .resources
                        .get_mut::<Input>()
                        .expect("Failed to get input resource!");

                    if !cursor_moved {
                        input.mouse.position_delta = glm::vec2(0.0, 0.0);
                    }
                }

                schedule.execute(&mut world);

                let delta_time =
                    (Instant::now().duration_since(last_frame).as_millis() as f64) / 1000_f64;
                last_frame = Instant::now();
                world
                    .resources
                    .get_mut::<DeltaTime>()
                    .expect("Failed to get delta time resource!")
                    .0 = delta_time;
            }
        });
    }
}
