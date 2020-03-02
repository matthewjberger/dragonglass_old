use crate::{input::Input, DeltaTime};
use legion::prelude::*;
use nalgebra_glm as glm;
use winit::VirtualKeyCode;

pub struct CameraViewMatrix(pub glm::Mat4);

pub enum CameraDirection {
    Forward,
    Backward,
    Left,
    Right,
}

pub struct Camera {
    pub position: glm::Vec3,
    pub right: glm::Vec3,
    pub front: glm::Vec3,
    pub up: glm::Vec3,
    pub world_up: glm::Vec3,
    pub speed: f32,
    pub sensitivity: f32,
    pub yaw_degrees: f32,
    pub pitch_degrees: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            position: glm::vec3(0.0, 0.0, 10.0),
            right: glm::vec3(0.0, 0.0, 0.0),
            front: glm::vec3(0.0, 0.0, -1.0),
            up: glm::vec3(0.0, 0.0, 0.0),
            world_up: glm::vec3(0.0, 1.0, 0.0),
            speed: 20.0,
            sensitivity: 0.05,
            yaw_degrees: -90.0,
            pitch_degrees: 0.0,
        }
    }
}

pub fn fps_camera_key_system() -> Box<dyn Schedulable> {
    SystemBuilder::new("fps_camera_key")
        .read_resource::<Input>()
        .read_resource::<DeltaTime>()
        .with_query(<Write<Camera>>::query())
        .build(move |_, mut world, (input, delta_time), query| {
            for mut camera in query.iter(&mut world) {
                let velocity = camera.speed * delta_time.0 as f32;

                let x_delta = camera.right * velocity;
                let y_delta = camera.front * velocity;

                if input.is_key_pressed(VirtualKeyCode::W) {
                    camera.position += y_delta;
                }

                if input.is_key_pressed(VirtualKeyCode::A) {
                    camera.position -= x_delta;
                }

                if input.is_key_pressed(VirtualKeyCode::S) {
                    camera.position -= y_delta;
                }

                if input.is_key_pressed(VirtualKeyCode::D) {
                    camera.position += x_delta;
                }
            }
        })
}

pub fn fps_camera_mouse_system() -> Box<dyn Schedulable> {
    let pitch_threshold = 89.0;
    SystemBuilder::new("fps_camera_mouse")
        .read_resource::<Input>()
        .write_resource::<CameraViewMatrix>()
        .with_query(<Write<Camera>>::query())
        .build(move |_, mut world, (input, camera_view_matrix), query| {
            // TODO: Support multiple cameras
            let mut camera = &mut query.iter(&mut world).collect::<Vec<_>>()[0];

            let (x_offset, y_offset) = (
                (input.mouse.offset_from_center.x as i32) as f32 * camera.sensitivity,
                (input.mouse.offset_from_center.y as i32) as f32 * camera.sensitivity,
            );

            camera.yaw_degrees -= x_offset;
            camera.pitch_degrees += y_offset;

            if camera.pitch_degrees > pitch_threshold {
                camera.pitch_degrees = pitch_threshold
            } else if camera.pitch_degrees < -pitch_threshold {
                camera.pitch_degrees = -pitch_threshold
            }

            calculate_vectors(&mut camera);

            let target = camera.position + camera.front;
            camera_view_matrix.0 = glm::look_at(&camera.position, &target, &camera.up);
        })
}

pub fn calculate_vectors(camera: &mut Camera) {
    let pitch_radians = camera.pitch_degrees.to_radians();
    let yaw_radians = camera.yaw_degrees.to_radians();
    camera.front = glm::vec3(
        pitch_radians.cos() * yaw_radians.cos(),
        pitch_radians.sin(),
        yaw_radians.sin() * pitch_radians.cos(),
    )
    .normalize();
    camera.right = camera.front.cross(&camera.world_up).normalize();
    camera.up = camera.right.cross(&camera.front).normalize();
}
