pub mod gltf;
use crate::resource::{Buffer, CommandPool};
use ash::vk;

pub struct Model {
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
}

impl Model {
    pub fn new(command_pool: &CommandPool, vertices: &[f32], indices: &[u32]) -> Self {
        let vertex_buffer =
            command_pool.create_device_local_buffer(vk::BufferUsageFlags::VERTEX_BUFFER, &vertices);

        let index_buffer =
            command_pool.create_device_local_buffer(vk::BufferUsageFlags::INDEX_BUFFER, &indices);

        Self {
            vertex_buffer,
            index_buffer,
        }
    }
}
