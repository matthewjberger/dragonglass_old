use nalgebra_glm as glm;
use specs::{prelude::*, Component};

// TODO: Rename MeshComponent to something more generic. (RenderComponent?)
#[derive(Component, Debug)]
#[storage(VecStorage)]
pub struct MeshComponent {
    pub mesh_name: String, // TODO: Make this a tag rather than a full path
}

#[derive(Component, Debug)]
#[storage(VecStorage)]
pub struct TransformComponent {
    pub translate: glm::Mat4,
    pub rotate: glm::Mat4,
    pub scale: glm::Mat4,
}

impl Default for TransformComponent {
    fn default() -> Self {
        Self {
            translate: glm::Mat4::identity(),
            rotate: glm::Mat4::identity(),
            scale: glm::Mat4::identity(),
        }
    }
}
