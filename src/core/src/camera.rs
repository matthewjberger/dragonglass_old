use crate::{input::Input, DeltaTime};
use legion::prelude::*;
use nalgebra_glm as glm;
use winit::VirtualKeyCode;

pub struct CameraState {
    pub view: glm::Mat4,
    pub position: glm::Vec3,
}

impl Default for CameraState {
    fn default() -> Self {
        CameraState {
            view: glm::Mat4::identity(),
            position: glm::Vec3::identity(),
        }
    }
}

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

impl Camera {
    pub fn look_at(&mut self, target: &glm::Vec3) {
        self.front = (target - self.position).normalize();
        self.pitch_degrees = self.front.y.asin().to_degrees();
        self.yaw_degrees = (self.front.x / self.front.y.asin().cos())
            .acos()
            .to_degrees();
        self.calculate_vectors();
    }

    pub fn calculate_vectors(&mut self) {
        let pitch_radians = self.pitch_degrees.to_radians();
        let yaw_radians = self.yaw_degrees.to_radians();
        self.front = glm::vec3(
            pitch_radians.cos() * yaw_radians.cos(),
            pitch_radians.sin(),
            yaw_radians.sin() * pitch_radians.cos(),
        )
        .normalize();
        self.right = self.front.cross(&self.world_up).normalize();
        self.up = self.right.cross(&self.front).normalize();
    }
}
pub fn fps_camera_key_system() -> Box<dyn Schedulable> {
    SystemBuilder::new("fps_camera_key")
        .read_resource::<Input>()
        .read_resource::<DeltaTime>()
        .write_resource::<CameraState>()
        .with_query(<Write<Camera>>::query())
        .build(
            move |_, mut world, (input, delta_time, camera_state), query| {
                let cameras = &mut query.iter(&mut world).collect::<Vec<_>>();
                let camera = &mut cameras[0];

                let velocity = (camera.speed * delta_time.0 as f32) + 0.02;

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

                camera_state.position = camera.position;
            },
        )
}

pub fn fps_camera_mouse_system() -> Box<dyn Schedulable> {
    let pitch_threshold = 89.0;
    SystemBuilder::new("fps_camera_mouse")
        .read_resource::<Input>()
        .write_resource::<CameraState>()
        .with_query(<Write<Camera>>::query())
        .build(move |_, mut world, (input, camera_state), query| {
            // TODO: Support multiple cameras
            let camera = &mut query.iter(&mut world).collect::<Vec<_>>()[0];

            let (x_offset, y_offset) = (
                (input.mouse.offset_from_center.x as i32) as f32 * camera.sensitivity,
                (input.mouse.offset_from_center.y as i32) as f32 * camera.sensitivity,
            );

            camera.yaw_degrees -= x_offset;
            camera.pitch_degrees -= y_offset;

            if camera.pitch_degrees > pitch_threshold {
                camera.pitch_degrees = pitch_threshold
            } else if camera.pitch_degrees < -pitch_threshold {
                camera.pitch_degrees = -pitch_threshold
            }

            camera.calculate_vectors();

            let target = camera.position + camera.front;
            camera_state.view = glm::look_at(&camera.position, &target, &camera.up);
            camera_state.position = camera.position;
        })
}
