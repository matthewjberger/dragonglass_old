use crate::core::VulkanContext;
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
                .expect("Failed to create buffer!")
        };

        // Get the buffer's memory requirements
        let memory_requirements = unsafe {
            context
                .logical_device()
                .logical_device()
                .get_buffer_memory_requirements(buffer)
        };

        let memory_type = Self::determine_memory_type_index(
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
                .expect("Failed to allocate memory!")
        };

        unsafe {
            // Bind the buffer memory for mapping
            context
                .logical_device()
                .logical_device()
                .bind_buffer_memory(buffer, memory, 0)
                .expect("Failed to bind buffer memory!");
        }

        Buffer {
            buffer,
            memory,
            memory_requirements,
            context,
        }
    }

    pub fn determine_memory_type_index(
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

    pub fn upload_to_buffer<T: Copy>(&self, data: &[T], offset: usize) {
        let data_pointer = self.map(
            offset as _,
            (data.len() * std::mem::size_of::<T>()) as vk::DeviceSize,
            vk::MemoryMapFlags::empty(),
        );
        unsafe {
            // Upload aligned staging data to the mapped buffer
            let mut align = ash::util::Align::new(
                data_pointer,
                std::mem::align_of::<T>() as _,
                self.memory_requirements.size as _,
            );
            align.copy_from_slice(data);
        }
        self.unmap();
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
                .expect("Failed to map memory!")
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