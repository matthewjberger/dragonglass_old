use dragonglass_backend_vulkan::{
    render::Renderer,
    systems::render::{animation_system, prepare_renderer, reload_system, render_system},
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
    dpi::PhysicalSize,
    event::{
        ElementState, Event, KeyboardInput, MouseButton, MouseScrollDelta, VirtualKeyCode,
        WindowEvent,
    },
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

#[derive(Default)]
struct ExitRequested(bool);

fn main() {
    env_logger::init();
    let (width, height, title) = (1920, 1080, "Dragonglass - Vulkan Rendering");

    log::debug!("Initializing application.");
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title(title)
        .with_inner_size(PhysicalSize::new(width, height))
        .build(&event_loop)
        .expect("Failed to create window.");

    log::debug!("Running application.");

    let mut world = World::new();

    let mut renderer = Renderer::new(&window);

    let input = Input::default();
    world.resources.insert(input);

    world.resources.insert(CameraState::default());

    world.resources.insert(DeltaTime(0 as _));

    let mut app_state = AppState::default();
    let window_size = window.inner_size();
    app_state.window.width = window_size.width as u32;
    app_state.window.height = window_size.height as u32;
    world.resources.insert(app_state);

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
            // AssetName("examples/assets/models/MetalRoughSpheres.glb".to_string()),
            // AssetName("examples/assets/models/RiggedSimple.glb".to_string()),
            // AssetName("examples/assets/models/BrainStem.glb".to_string()),
            AssetName("examples/assets/models/VC.glb".to_string()),
            AnimationState { time: 0.0 },
            Transform::default(),
        )],
    );

    world.insert(
        (),
        vec![(
            AssetName("examples/assets/models/BoxAnimated.glb".to_string()),
            AnimationState { time: 0.0 },
            Transform {
                translate: glm::translate(&glm::Mat4::identity(), &glm::vec3(6.0, 0.0, 0.0)),
                ..Default::default()
            },
        )],
    );

    world.insert(
        (),
        vec![(
            AssetName("examples/assets/models/DamagedHelmet.glb".to_string()),
            Transform {
                translate: glm::translate(&glm::Mat4::identity(), &glm::vec3(-6.0, 0.0, 0.0)),
                ..Default::default()
            },
        )],
    );

    // world.insert(
    //     (),
    //     vec![(
    //         AssetName("examples/assets/models/Sponza/Sponza.gltf".to_string()),
    //         AnimationState { time: 0.0 },
    //         Transform {
    //             translate: glm::translate(&glm::Mat4::identity(), &glm::vec3(0.0, -6.0, 0.0)),
    //             scale: glm::scale(&glm::Mat4::identity(), &glm::vec3(6.0, 6.0, 6.0)),
    //             ..Default::default()
    //         },
    //     )],
    // );

    prepare_renderer(&mut renderer, &mut world);
    world.resources.insert(renderer);

    let mut last_frame = Instant::now();
    let mut cursor_moved = false;
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::NewEvents { .. } => {
                let delta_time =
                    (Instant::now().duration_since(last_frame).as_micros() as f64) / 1_000_000_f64;
                last_frame = Instant::now();
                world
                    .resources
                    .get_mut::<DeltaTime>()
                    .expect("Failed to get delta time resource!")
                    .0 = delta_time;
            }
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            virtual_keycode: Some(keycode),
                            state,
                            ..
                        },
                    ..
                } => {
                    let mut input = world
                        .resources
                        .get_mut::<Input>()
                        .expect("Failed to get input resource!");
                    if keycode == VirtualKeyCode::Escape {
                        *control_flow = ControlFlow::Exit;
                    }
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
                        ((window_size.width as f32 / 2.0) - position.x as f32) as _,
                        ((window_size.height as f32 / 2.0) - position.y as f32) as _,
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
                _ => {}
            },
            Event::MainEventsCleared => {
                let mut input = world
                    .resources
                    .get_mut::<Input>()
                    .expect("Failed to get input resource!");

                if !cursor_moved {
                    input.mouse.position_delta = glm::vec2(0.0, 0.0);
                }
                cursor_moved = false;
                window.request_redraw();
            }
            Event::RedrawRequested(_) => {
                schedule.execute(&mut world);
            }
            Event::RedrawEventsCleared => {
                let mut input = world
                    .resources
                    .get_mut::<Input>()
                    .expect("Failed to get input resource!");
                input.mouse.wheel_delta = 0.0;
            }
            _ => (),
        }
    });
}
