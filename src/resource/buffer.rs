use crate::VulkanContext;
use ash::{version::DeviceV1_0, vk};
use std::sync::Arc;

// TODO: Add snafu errors

pub struct Buffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    memory_requirements: vk::MemoryRequirements,
    context: Arc<VulkanContext>,
}

impl Buffer {
    // TODO: Refactor this to use less parameters and be shorter
    pub fn new(
        context: Arc<VulkanContext>,
        size: ash::vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        required_properties: vk::MemoryPropertyFlags,
    ) -> Self {
        // Build the staging buffer creation info
        let buffer_info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .build();

        // Create the staging buffer
        let buffer = unsafe {
            context
                .logical_device()
                .logical_device()
                .create_buffer(&buffer_info, None)
                .unwrap()
        };

        // Get the buffer's memory requirements
        let memory_requirements = unsafe {
            context
                .logical_device()
                .logical_device()
                .get_buffer_memory_requirements(buffer)
        };

        let memory_type = Self::determine_memory_type(
            memory_requirements,
            context.physical_device_memory_properties(),
            required_properties,
        );

        // Create the staging buffer allocation info
        let buffer_allocation_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(memory_requirements.size)
            .memory_type_index(memory_type)
            .build();

        // Allocate memory for the buffer
        let memory = unsafe {
            context
                .logical_device()
                .logical_device()
                .allocate_memory(&buffer_allocation_info, None)
                .unwrap()
        };

        unsafe {
            // Bind the buffer memory for mapping
            context
                .logical_device()
                .logical_device()
                .bind_buffer_memory(buffer, memory, 0)
                .unwrap();
        }

        Buffer {
            buffer,
            memory,
            memory_requirements,
            context,
        }
    }

    fn determine_memory_type(
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

    // TODO: Refactor this to use less parameters
    // Upload to the entire buffer
    pub fn upload_to_entire_buffer<A, T: Copy>(&self, data: &[T]) {
        let data_pointer = self.map_entire_buffer();
        unsafe {
            // Upload aligned staging data to the mapped buffer
            let mut align = ash::util::Align::new(
                data_pointer,
                std::mem::align_of::<A>() as _,
                self.memory_requirements.size as _,
            );
            align.copy_from_slice(&data);
        }
        self.unmap();
    }

    fn map_entire_buffer(&self) -> *mut std::ffi::c_void {
        self.map(
            0,
            self.memory_requirements.size as _,
            vk::MemoryMapFlags::empty(),
        )
    }

    fn map(
        &self,
        offset: vk::DeviceSize,
        size: vk::DeviceSize,
        flags: vk::MemoryMapFlags,
    ) -> *mut std::ffi::c_void {
        unsafe {
            self.context
                .logical_device()
                .logical_device()
                .map_memory(self.memory, offset, size, flags)
                .unwrap()
        }
    }

    fn unmap(&self) {
        unsafe {
            self.context
                .logical_device()
                .logical_device()
                .unmap_memory(self.memory());
        }
    }

    pub fn buffer(&self) -> vk::Buffer {
        self.buffer
    }

    pub fn memory(&self) -> vk::DeviceMemory {
        self.memory
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.context
                .logical_device()
                .logical_device()
                .destroy_buffer(self.buffer, None);
            self.context
                .logical_device()
                .logical_device()
                .free_memory(self.memory, None);
        }
    }
}

// TODO: Refactor this to use less parameters
pub fn create_device_local_buffer<A, T: Copy>(
    context: Arc<VulkanContext>,
    command_pool: ash::vk::CommandPool,
    graphics_queue: vk::Queue,
    usage_flags: vk::BufferUsageFlags,
    vertices: &[T],
) -> Buffer {
    let buffer_size = (vertices.len() * std::mem::size_of::<T>() as usize) as ash::vk::DeviceSize;

    let staging_buffer = Buffer::new(
        context.clone(),
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    );

    staging_buffer.upload_to_entire_buffer::<A, _>(&vertices);

    let vertex_buffer = Buffer::new(
        context.clone(),
        buffer_size,
        vk::BufferUsageFlags::TRANSFER_DST | usage_flags,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    );

    copy_buffer(
        context.logical_device().logical_device(),
        command_pool,
        graphics_queue,
        staging_buffer.buffer(),
        vertex_buffer.buffer(),
        buffer_size,
    );

    vertex_buffer
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

// TODO: Move this to a separate command module
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
