use ash::{version::DeviceV1_0, vk};
use std::mem;

pub fn determine_memory_type(
    buffer_memory_requirements: vk::MemoryRequirements,
    physical_device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
    required_properties: vk::MemoryPropertyFlags,
) -> u32 {
    // Determine the buffer's memory type
    let mut memory_type = 0;
    let mut found_memory_type = false;
    for index in 0..physical_device_memory_properties.memory_type_count {
        if buffer_memory_requirements.memory_type_bits & (1 << index) != 0
            && physical_device_memory_properties.memory_types[index as usize]
                .property_flags
                .contains(required_properties)
        {
            memory_type = index;
            found_memory_type = true;
        }
    }
    if !found_memory_type {
        panic!("Failed to find suitable memory type.")
    }
    memory_type
}

pub fn create_buffer(
    logical_device: &ash::Device,
    size: ash::vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    physical_device_memory_properties: &ash::vk::PhysicalDeviceMemoryProperties,
    required_properties: vk::MemoryPropertyFlags,
) -> (vk::Buffer, vk::DeviceMemory, vk::DeviceSize) {
    // Build the staging buffer creation info
    let buffer_info = vk::BufferCreateInfo::builder()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .build();

    // Create the staging buffer
    let buffer = unsafe { logical_device.create_buffer(&buffer_info, None).unwrap() };

    // Get the buffer's memory requirements
    let buffer_memory_requirements =
        unsafe { logical_device.get_buffer_memory_requirements(buffer) };

    let memory_type = determine_memory_type(
        buffer_memory_requirements,
        physical_device_memory_properties,
        required_properties,
    );

    // Create the staging buffer allocation info
    let buffer_allocation_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(buffer_memory_requirements.size)
        .memory_type_index(memory_type)
        .build();

    // Allocate memory for the buffer
    let buffer_memory = unsafe {
        logical_device
            .allocate_memory(&buffer_allocation_info, None)
            .unwrap()
    };

    unsafe {
        // Bind the buffer memory for mapping
        logical_device
            .bind_buffer_memory(buffer, buffer_memory, 0)
            .unwrap();
    }

    (buffer, buffer_memory, buffer_memory_requirements.size)
}

pub fn create_device_local_buffer<A, T: Copy>(
    logical_device: &ash::Device,
    physical_device_memory_properties: &ash::vk::PhysicalDeviceMemoryProperties,
    command_pool: ash::vk::CommandPool,
    graphics_queue: vk::Queue,
    usage_flags: vk::BufferUsageFlags,
    vertices: &[T],
) -> (vk::Buffer, vk::DeviceMemory) {
    let buffer_size = (vertices.len() * mem::size_of::<T>() as usize) as ash::vk::DeviceSize;

    let (staging_buffer, staging_buffer_memory, staging_memory_size) = create_buffer(
        &logical_device,
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        physical_device_memory_properties,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    );

    unsafe {
        // Map the entire buffer
        let data_pointer = logical_device
            .map_memory(
                staging_buffer_memory,
                0,
                buffer_size as _,
                vk::MemoryMapFlags::empty(),
            )
            .unwrap();

        // Upload aligned staging data to the mapped buffer
        let mut align = ash::util::Align::new(
            data_pointer,
            mem::align_of::<A>() as _,
            staging_memory_size as _,
        );
        align.copy_from_slice(&vertices);

        // Unmap the buffer memory
        logical_device.unmap_memory(staging_buffer_memory);
    }

    let (vertex_buffer, vertex_buffer_memory, _) = create_buffer(
        &logical_device,
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_DST | usage_flags,
        physical_device_memory_properties,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    );

    copy_buffer(
        logical_device,
        command_pool,
        graphics_queue,
        staging_buffer,
        vertex_buffer,
        buffer_size,
    );

    unsafe {
        // Free the staging buffer
        logical_device.destroy_buffer(staging_buffer, None);

        // Free the staging buffer memory
        logical_device.free_memory(staging_buffer_memory, None)
    };

    (vertex_buffer, vertex_buffer_memory)
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
