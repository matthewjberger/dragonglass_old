use crate::{resource::Buffer, VulkanContext};
use ash::{version::DeviceV1_0, vk};
use std::{mem, sync::Arc};

pub fn create_device_local_buffer<A, T: Copy>(
    context: Arc<VulkanContext>,
    command_pool: ash::vk::CommandPool,
    graphics_queue: vk::Queue,
    usage_flags: vk::BufferUsageFlags,
    vertices: &[T],
) -> Buffer {
    let buffer_size = (vertices.len() * mem::size_of::<T>() as usize) as ash::vk::DeviceSize;

    let staging_buffer = Buffer::new(
        context.clone(),
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    );

    unsafe {
        // Map the entire buffer
        let data_pointer = context
            .logical_device()
            .map_memory(
                staging_buffer.memory(),
                0,
                buffer_size as _,
                vk::MemoryMapFlags::empty(),
            )
            .unwrap();

        // Upload aligned staging data to the mapped buffer
        let mut align = ash::util::Align::new(
            data_pointer,
            mem::align_of::<A>() as _,
            staging_buffer.memory_requirements().size as _,
        );
        align.copy_from_slice(&vertices);

        // Unmap the buffer memory
        context
            .logical_device()
            .unmap_memory(staging_buffer.memory());
    }

    let vertex_buffer = Buffer::new(
        context.clone(),
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_DST | usage_flags,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    );

    copy_buffer(
        context.logical_device(),
        command_pool,
        graphics_queue,
        staging_buffer.buffer(),
        vertex_buffer.buffer(),
        buffer_size,
    );

    vertex_buffer
}

pub fn execute_command_once<F: FnOnce(vk::CommandBuffer)>(
    logical_device: &ash::Device,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    executor: F,
) {
    // Allocate a command buffer using the command pool
    let command_buffer = {
        let allocation_info = vk::CommandBufferAllocateInfo::builder()
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(command_pool)
            .command_buffer_count(1)
            .build();

        unsafe {
            logical_device
                .allocate_command_buffers(&allocation_info)
                .unwrap()[0]
        }
    };
    let command_buffers = [command_buffer];

    // Begin recording
    let begin_info = vk::CommandBufferBeginInfo::builder()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
        .build();
    unsafe {
        logical_device
            .begin_command_buffer(command_buffer, &begin_info)
            .unwrap()
    };

    executor(command_buffer);

    // End command buffer recording
    unsafe { logical_device.end_command_buffer(command_buffer).unwrap() };

    // Build the submission info
    let submit_info = vk::SubmitInfo::builder()
        .command_buffers(&command_buffers)
        .build();
    let submit_info_arr = [submit_info];

    unsafe {
        // Submit the command buffer
        logical_device
            .queue_submit(queue, &submit_info_arr, vk::Fence::null())
            .unwrap();

        // Wait for the command buffer to be executed
        logical_device.queue_wait_idle(queue).unwrap();

        // Free the command buffer
        logical_device.free_command_buffers(command_pool, &command_buffers);
    };
}

pub fn copy_buffer(
    logical_device: &ash::Device,
    command_pool: vk::CommandPool,
    transfer_queue: vk::Queue,
    source: vk::Buffer,
    destination: vk::Buffer,
    buffer_size: vk::DeviceSize,
) {
    execute_command_once(
        logical_device,
        command_pool,
        transfer_queue,
        |command_buffer| {
            // Define the region for the buffer copy
            let region = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: buffer_size as _,
            };
            let regions = [region];

            // Copy the bytes of the staging buffer to the vertex buffer
            unsafe {
                logical_device.cmd_copy_buffer(command_buffer, source, destination, &regions)
            };
        },
    );
}
