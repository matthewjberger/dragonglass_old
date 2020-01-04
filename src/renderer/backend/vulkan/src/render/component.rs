use crate::resource::Buffer;
use nalgebra_glm as glm;
use specs::{prelude::*, Component};

// TODO: Move non-vulkan specific components out to a separate crate
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

#[derive(Component)]
#[storage(VecStorage)]
pub struct PrimitiveSetComponent {
    pub primitives: Vec<Primitive>,
}

// TODO: Move this to another place as it isn't a component
pub struct Primitive {
    pub number_of_indices: u32,
    pub uniform_buffers: Vec<Buffer>,
    pub descriptor_sets: Vec<ash::vk::DescriptorSet>,
    pub material_index: Option<usize>,
    pub vertex_buffer_offset: i32,
    pub index_buffer_offset: u32,
    pub asset_index: usize,
}
