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

    pub fn create_device_local_buffer<T: Copy>(
        &self,
        graphics_queue: vk::Queue,
        usage_flags: vk::BufferUsageFlags,
        vertices: &[T],
    ) -> Buffer {
        let buffer_size = (vertices.len() * std::mem::size_of::<T>()) as ash::vk::DeviceSize;

        let staging_buffer = Buffer::new(
            self.context.clone(),
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        staging_buffer.upload_to_buffer(&vertices, 0, std::mem::align_of::<T>() as _, false);

        let vertex_buffer = Buffer::new(
            self.context.clone(),
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_DST | usage_flags,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        self.copy_buffer_to_buffer(
            graphics_queue,
            staging_buffer.buffer(),
            vertex_buffer.buffer(),
            buffer_size,
        );

        vertex_buffer
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
        size: vk::DeviceSize,
    ) {
        self.execute_command_once(transfer_queue, |command_buffer| {
            // Define the region for the buffer copy
            let region = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size,
            };
            let regions = [region];

            // Copy the bytes of the staging buffer to the vertex buffer
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
        width: u32,
        height: u32,
    ) {
        self.execute_command_once(transition_queue, |command_buffer| {
            let region = vk::BufferImageCopy::builder()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })
                .build();
            let regions = [region];
            unsafe {
                self.context
                    .logical_device()
                    .logical_device()
                    .cmd_copy_buffer_to_image(
                        command_buffer,
                        buffer,
                        image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &regions,
                    )
            }
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

    pub fn transition_image_layout(
        &self,
        transition_queue: vk::Queue,
        image: vk::Image,
        format: vk::Format,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    ) {
        self.execute_command_once(transition_queue, |command_buffer| {
            let (src_access_mask, dst_access_mask, src_stage, dst_stage) =
                match (old_layout, new_layout) {
                    (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
                        vk::AccessFlags::empty(),
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::TRANSFER,
                    ),
                    (
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    ) => (
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::AccessFlags::SHADER_READ,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                    ),
                    (
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    ) => (
                        vk::AccessFlags::empty(),
                        vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                    ),
                    _ => panic!(
                        "Unsupported layout transition({:?} => {:?})",
                        old_layout, new_layout
                    ),
                };

            let aspect_mask = if new_layout == vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL {
                let mut mask = vk::ImageAspectFlags::DEPTH;
                if Self::has_stencil_component(format) {
                    mask |= vk::ImageAspectFlags::STENCIL;
                }
                mask
            } else {
                vk::ImageAspectFlags::COLOR
            };

            let barrier = vk::ImageMemoryBarrier::builder()
                .old_layout(old_layout)
                .new_layout(new_layout)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .src_access_mask(src_access_mask)
                .dst_access_mask(dst_access_mask)
                .build();
            let barriers = [barrier];

            unsafe {
                self.context
                    .logical_device()
                    .logical_device()
                    .cmd_pipeline_barrier(
                        command_buffer,
                        src_stage,
                        dst_stage,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &barriers,
                    )
            };
        });
    }

    // TODO: Move this to a more specific component
    pub fn has_stencil_component(format: vk::Format) -> bool {
        format == vk::Format::D32_SFLOAT_S8_UINT || format == vk::Format::D24_UNORM_S8_UINT
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
