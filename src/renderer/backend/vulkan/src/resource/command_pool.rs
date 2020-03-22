// TODO: Make a type alias for the current device version (DeviceV1_0)
use crate::{core::VulkanContext, resource::Buffer, sync::CurrentFrameSynchronization};
use ash::{version::DeviceV1_0, vk};
use std::sync::Arc;

// TODO: Add snafu errors

pub struct CommandPool {
    pool: vk::CommandPool,
    context: Arc<VulkanContext>,
    command_buffers: Vec<vk::CommandBuffer>,
}

impl CommandPool {
    pub fn new(context: Arc<VulkanContext>, flags: vk::CommandPoolCreateFlags) -> Self {
        let command_pool_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(context.graphics_queue_family_index())
            .flags(flags)
            .build();

        let pool = unsafe {
            context
                .logical_device()
                .logical_device()
                .create_command_pool(&command_pool_info, None)
                .expect("Failed to create a command pool!")
        };

        CommandPool {
            pool,
            context,
            command_buffers: Vec::new(),
        }
    }

    pub fn pool(&self) -> vk::CommandPool {
        self.pool
    }

    pub fn command_buffers(&self) -> &[vk::CommandBuffer] {
        &self.command_buffers
    }

    pub fn allocate_command_buffers(&mut self, size: vk::DeviceSize) {
        let allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(self.pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(size as _)
            .build();

        self.command_buffers = unsafe {
            self.context
                .logical_device()
                .logical_device()
                .allocate_command_buffers(&allocate_info)
                .expect("Failed to allocate command buffers!")
        };
    }

    pub fn clear_command_buffers(&mut self) {
        if !self.command_buffers.is_empty() {
            unsafe {
                self.context
                    .logical_device()
                    .logical_device()
                    .free_command_buffers(self.pool, &self.command_buffers);
            }
        }
        self.command_buffers.clear();
    }

    pub fn create_staging_buffer<T: Copy>(&self, data: &[T]) -> Buffer {
        let buffer_size = (data.len() * std::mem::size_of::<T>()) as ash::vk::DeviceSize;

        let staging_buffer = Buffer::new_mapped_basic(
            self.context.clone(),
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk_mem::MemoryUsage::CpuToGpu,
        );
        staging_buffer.upload_to_buffer(&data, 0, std::mem::align_of::<T>() as _);
        staging_buffer
    }

    pub fn create_device_local_buffer<T: Copy>(
        &self,
        usage_flags: vk::BufferUsageFlags,
        data: &[T],
        regions: &[vk::BufferCopy],
    ) -> Buffer {
        let staging_buffer = self.create_staging_buffer(&data);

        let device_local_buffer = Buffer::new_mapped_basic(
            self.context.clone(),
            staging_buffer.allocation_info().get_size() as _,
            vk::BufferUsageFlags::TRANSFER_DST | usage_flags,
            vk_mem::MemoryUsage::GpuOnly,
        );

        self.copy_buffer_to_buffer(
            self.context.graphics_queue(),
            staging_buffer.buffer(),
            device_local_buffer.buffer(),
            &regions,
        );

        device_local_buffer
    }

    // TODO: refactor this to use less parameters
    pub fn submit_command_buffer(
        &self,
        index: usize,
        queue: vk::Queue,
        wait_stages: &[vk::PipelineStageFlags],
        current_frame_synchronization: &CurrentFrameSynchronization,
    ) {
        let image_available_semaphores = [current_frame_synchronization.image_available()];
        let render_finished_semaphores = [current_frame_synchronization.render_finished()];
        // TODO: Add error handling, index may be invalid
        let command_buffers_to_use = [self.command_buffers()[index]];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(&image_available_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers_to_use)
            .signal_semaphores(&render_finished_semaphores)
            .build();
        let submit_info_arr = [submit_info];
        unsafe {
            self.context
                .logical_device()
                .logical_device()
                .queue_submit(
                    queue,
                    &submit_info_arr,
                    current_frame_synchronization.in_flight(),
                )
                .expect("Failed to submit command buffer to queue!")
        }
    }

    pub fn copy_buffer_to_buffer(
        &self,
        transfer_queue: vk::Queue,
        source: vk::Buffer,
        destination: vk::Buffer,
        regions: &[vk::BufferCopy],
    ) {
        self.execute_command_once(transfer_queue, |command_buffer| {
            unsafe {
                self.context
                    .logical_device()
                    .logical_device()
                    .cmd_copy_buffer(command_buffer, source, destination, &regions)
            };
        });
    }

    pub fn copy_buffer_to_image(
        &self,
        transition_queue: vk::Queue,
        buffer: vk::Buffer,
        image: vk::Image,
        regions: &[vk::BufferImageCopy],
    ) {
        self.execute_command_once(transition_queue, |command_buffer| unsafe {
            self.context
                .logical_device()
                .logical_device()
                .cmd_copy_buffer_to_image(
                    command_buffer,
                    buffer,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    regions,
                )
        });
    }

    // TODO: Refactor this to be smaller. Functionality can probably be reused
    // in generic command buffer submission method
    pub fn execute_command_once<F: FnOnce(vk::CommandBuffer)>(
        &self,
        queue: vk::Queue,
        executor: F,
    ) {
        // Allocate a command buffer using the command pool
        let command_buffer = {
            let allocation_info = vk::CommandBufferAllocateInfo::builder()
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_pool(self.pool)
                .command_buffer_count(1)
                .build();

            unsafe {
                self.context
                    .logical_device()
                    .logical_device()
                    .allocate_command_buffers(&allocation_info)
                    .expect("Failed to allocate command buffers")[0]
            }
        };
        let command_buffers = [command_buffer];

        // Begin recording
        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
            .build();
        let logical_device = self.context.logical_device().logical_device();

        unsafe {
            logical_device
                .begin_command_buffer(command_buffer, &begin_info)
                .expect("Failed to begin command buffer!")
        };

        executor(command_buffer);

        // End command buffer recording
        unsafe {
            logical_device
                .end_command_buffer(command_buffer)
                .expect("Failed to end command buffer!")
        };

        // Build the submission info
        let submit_info = vk::SubmitInfo::builder()
            .command_buffers(&command_buffers)
            .build();
        let submit_info_arr = [submit_info];

        unsafe {
            // Submit the command buffer
            logical_device
                .queue_submit(queue, &submit_info_arr, vk::Fence::null())
                .expect("Failed to submit command buffer to queue!");

            // Wait for the command buffer to be executed
            logical_device
                .queue_wait_idle(queue)
                .expect("Failed to wait for command buffer to be executed!");

            // Free the command buffer
            logical_device.free_command_buffers(self.pool(), &command_buffers);
        };
    }

    // TODO: Move this to the texture module
    pub fn transition_image_layout(
        &self,
        barriers: &[vk::ImageMemoryBarrier],
        src_stage_mask: vk::PipelineStageFlags,
        dst_stage_mask: vk::PipelineStageFlags,
    ) {
        self.execute_command_once(self.context.graphics_queue(), |command_buffer| {
            unsafe {
                self.context
                    .logical_device()
                    .logical_device()
                    .cmd_pipeline_barrier(
                        command_buffer,
                        src_stage_mask,
                        dst_stage_mask,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &barriers,
                    )
            };
        });
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        self.clear_command_buffers();
        unsafe {
            self.context
                .logical_device()
                .logical_device()
                .destroy_command_pool(self.pool, None);
        }
    }
}
