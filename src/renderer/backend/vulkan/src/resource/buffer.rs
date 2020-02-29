use crate::core::VulkanContext;
use ash::vk;
use std::sync::Arc;

// TODO: Add snafu errors

pub struct Buffer {
    buffer: vk::Buffer,
    allocation: vk_mem::Allocation,
    allocation_info: vk_mem::AllocationInfo,
    context: Arc<VulkanContext>,
}

impl Buffer {
    pub fn new(
        context: Arc<VulkanContext>,
        allocation_create_info: &vk_mem::AllocationCreateInfo,
        buffer_create_info: &vk::BufferCreateInfo,
    ) -> Self {
        let (buffer, allocation, allocation_info) = context
            .allocator()
            .create_buffer(&buffer_create_info, &allocation_create_info)
            .expect("Failed to create buffer!");

        Self {
            buffer,
            allocation,
            allocation_info,
            context,
        }
    }

    pub fn new_mapped_basic(
        context: Arc<VulkanContext>,
        size: vk::DeviceSize,
        buffer_usage: vk::BufferUsageFlags,
        memory_usage: vk_mem::MemoryUsage,
    ) -> Self {
        let allocation_create_info = vk_mem::AllocationCreateInfo {
            usage: memory_usage,
            ..Default::default()
        };

        let buffer_create_info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(buffer_usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .build();

        Buffer::new(context, &allocation_create_info, &buffer_create_info)
    }

    pub fn upload_to_buffer<T: Copy>(&self, data: &[T], offset: usize, alignment: vk::DeviceSize) {
        let data_pointer = self.map_memory().expect("Failed to map memory!");
        unsafe {
            let mut align = ash::util::Align::new(
                data_pointer.add(offset) as _,
                alignment,
                self.allocation_info.get_size() as _,
            );
            align.copy_from_slice(data);
        }
    }

    pub fn map_memory(&self) -> vk_mem::error::Result<*mut u8> {
        self.context.allocator().map_memory(&self.allocation)
    }

    pub fn unmap_memory(&self) -> vk_mem::error::Result<()> {
        self.context.allocator().unmap_memory(&self.allocation)
    }

    pub fn flush(&self, offset: usize, size: usize) -> vk_mem::error::Result<()> {
        self.context
            .allocator()
            .flush_allocation(&self.allocation, offset, size)
    }

    pub fn buffer(&self) -> vk::Buffer {
        self.buffer
    }

    pub fn allocation(&self) -> &vk_mem::Allocation {
        &self.allocation
    }

    pub fn allocation_info(&self) -> &vk_mem::AllocationInfo {
        &self.allocation_info
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        self.context
            .allocator()
            .destroy_buffer(self.buffer, &self.allocation)
            .expect("Failed to destroy buffer!");
    }
}
