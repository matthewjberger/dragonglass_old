use crate::render::component::TransformComponent;
use nalgebra_glm as glm;
use specs::prelude::*;

pub struct TransformationSystem;

impl<'a> System<'a> for TransformationSystem {
    type SystemData = WriteStorage<'a, TransformComponent>;

    fn run(&mut self, data: Self::SystemData) {
        let mut transforms = data;
        for transform in (&mut transforms).join() {
            transform.rotate = glm::rotate(
                &transform.rotate,
                0.1_f32.to_radians(),
                &glm::vec3(0.0, 1.0, 0.0),
            );
        }
    }
}
