pub mod gltf;
use crate::resource::{Buffer, CommandPool};
use ash::vk;

pub struct ModelBuffers {
    pub vertex_buffer: Buffer,
    pub index_buffer: Option<Buffer>,
}

impl ModelBuffers {
    pub fn new(command_pool: &CommandPool, vertices: &[f32], indices: Option<&[u32]>) -> Self {
        let vertex_buffer =
            command_pool.create_device_local_buffer(vk::BufferUsageFlags::VERTEX_BUFFER, &vertices);

        let index_buffer = if let Some(indices) = indices {
            Some(
                command_pool
                    .create_device_local_buffer(vk::BufferUsageFlags::INDEX_BUFFER, &indices),
            )
        } else {
            None
        };

        Self {
            vertex_buffer,
            index_buffer,
        }
    }
}
