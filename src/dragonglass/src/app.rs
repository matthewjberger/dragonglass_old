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
    dpi::{LogicalPosition, LogicalSize},
    ElementState, Event, EventsLoop, MouseButton, MouseScrollDelta, VirtualKeyCode, Window,
    WindowEvent,
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

        App {
            event_loop,
            window,
            should_exit: false,
        }
    }

    #[allow(dead_code)]
    fn stow_cursor(&self) {
        self.window
            .grab_cursor(true)
            .expect("Failed to set cursor grabbing on window!");
        self.window.hide_cursor(true);
    }

    #[allow(dead_code)]
    fn set_fullscreen(&self) {
        let monitor_id = self.window.get_current_monitor();
        self.window.set_fullscreen(Some(monitor_id));
    }

    #[allow(dead_code)]
    fn center_cursor(&mut self) {
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
    }

    pub fn run(&mut self) {
        log::debug!("Running application.");

        // self.stow_cursor();

        let mut world = World::new();

        let renderer = Renderer::new(&self.window);
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

        let mut schedule = Schedule::builder()
            .add_system(orbital_camera_mouse_system())
            .add_system(animation_system())
            .add_system(reload_system())
            .flush()
            // More game simulation systems can go here
            .add_thread_local(render_system())
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
                // AssetName("examples/assets/models/DamagedHelmet.glb".to_string()),
                AssetName("examples/assets/models/MetalRoughSpheres.glb".to_string()),
                AnimationState { time: 0.0 },
                Transform::default(),
            )],
        );

        prepare_schedule.execute(&mut world);

        let mut last_frame = Instant::now();
        loop {
            self.process_events(&mut world);

            if self.should_exit {
                break;
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

        let renderer = world
            .resources
            .get::<Renderer>()
            .expect("Failed to get renderer resource!");
        renderer.context.logical_device().wait_idle();
    }

    fn process_events(&mut self, world: &mut World) {
        let mut input = world
            .resources
            .get_mut::<Input>()
            .expect("Failed to get input resource!");
        let mut app_state = world
            .resources
            .get_mut::<AppState>()
            .expect("Failed to get input resource!");
        let mut should_exit = false;
        let window_size = self
            .window
            .get_inner_size()
            .expect("Failed to get window inner size!");
        let mut cursor_moved = false;
        input.mouse.wheel_delta = 0.0;

        self.event_loop.poll_events(|event| {
            if let Event::WindowEvent { event, .. } = event {
                match event {
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
                    WindowEvent::Resized(LogicalSize { width, height }) => {
                        app_state.window.width = width as u32;
                        app_state.window.height = height as u32;
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
                        let last_position = input.mouse.position;
                        let current_position = glm::vec2(position.x as _, position.y as _);
                        input.mouse.position = current_position;
                        input.mouse.position_delta = current_position - last_position;
                        input.mouse.offset_from_center = glm::vec2(
                            ((window_size.width / 2.0) - position.x) as _,
                            ((window_size.height / 2.0) - position.y) as _,
                        );
                        cursor_moved = true;
                    }
                    WindowEvent::MouseWheel {
                        delta: MouseScrollDelta::LineDelta(_, v_lines),
                        ..
                    } => {
                        input.mouse.wheel_delta = v_lines;
                    }
                    WindowEvent::DroppedFile(file_pathbuf) => {
                        println!("Received file: {:?}", file_pathbuf);
                    }
                    _ => {}
                }
            }
        });
        self.should_exit = should_exit;
        if !cursor_moved {
            input.mouse.position_delta = glm::vec2(0.0, 0.0);
        }
    }
}
