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
            Self::create_buffer(command_pool, &vertices, vk::BufferUsageFlags::VERTEX_BUFFER);

        let index_buffer = if let Some(indices) = indices {
            let index_buffer =
                Self::create_buffer(command_pool, &indices, vk::BufferUsageFlags::INDEX_BUFFER);
            Some(index_buffer)
        } else {
            None
        };

        Self {
            vertex_buffer,
            index_buffer,
        }
    }

    fn create_buffer<T: Copy>(
        command_pool: &CommandPool,
        data: &[T],
        usage_flags: vk::BufferUsageFlags,
    ) -> Buffer {
        let region = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: (data.len() * std::mem::size_of::<T>()) as ash::vk::DeviceSize,
        };
        command_pool.create_device_local_buffer(usage_flags, &data, &[region])
    }
}
