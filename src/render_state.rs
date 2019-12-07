use crate::{
    context::VulkanContext,
    core::{Swapchain, SwapchainProperties},
    render::{Framebuffer, GraphicsPipeline, RenderPass},
    resource::{Buffer, CommandPool, DescriptorPool, DescriptorSetLayout},
    vertex::Vertex,
};
use ash::{version::DeviceV1_0, vk};
use nalgebra_glm as glm;
use std::{mem, sync::Arc, time::Instant};

#[derive(Debug, Clone, Copy)]
pub struct UniformBufferObject {
    pub model: glm::Mat4,
    pub view: glm::Mat4,
    pub projection: glm::Mat4,
}

impl UniformBufferObject {
    fn get_descriptor_set_layout_bindings() -> [vk::DescriptorSetLayoutBinding; 1] {
        let ubo_layout_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .build();
        [ubo_layout_binding]
    }
}

pub struct RenderState {
    context: Arc<VulkanContext>,
    pub command_pool: CommandPool,
    pub descriptor_pool: DescriptorPool,
    pub descriptor_set_layout: DescriptorSetLayout,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub framebuffers: Vec<Framebuffer>,
    pub graphics_queue: vk::Queue,
    pub index_buffer: Buffer,
    pub number_of_indices: u32,
    pub pipeline: GraphicsPipeline,
    pub present_queue: vk::Queue,
    pub render_pass: RenderPass,
    pub swapchain: Swapchain,
    pub transient_command_pool: CommandPool,
    pub uniform_buffers: Vec<Buffer>,
    pub vertex_buffer: Buffer,
}

impl RenderState {
    pub fn new(context: Arc<VulkanContext>, vertices: &[Vertex], indices: &[u16]) -> Self {
        unsafe { context.logical_device().device_wait_idle().unwrap() };

        let graphics_queue = unsafe {
            context
                .logical_device()
                .get_device_queue(context.graphics_queue_family_index(), 0)
        };

        let present_queue = unsafe {
            context
                .logical_device()
                .get_device_queue(context.present_queue_family_index(), 0)
        };

        let swapchain = Swapchain::new(context.clone());
        let render_pass = RenderPass::new(context.clone(), swapchain.properties());

        let bindings = UniformBufferObject::get_descriptor_set_layout_bindings();
        let descriptor_set_layout = DescriptorSetLayout::new(context.clone(), &bindings);

        let pipeline = GraphicsPipeline::new(
            context.clone(),
            swapchain.properties(),
            render_pass.render_pass(),
            descriptor_set_layout.layout(),
        );

        // Create one framebuffer for each image in the swapchain
        let framebuffers = swapchain
            .image_views()
            .iter()
            .map(|view| [view.view()])
            .map(|attachments| {
                Framebuffer::new(
                    context.clone(),
                    swapchain.properties(),
                    render_pass.render_pass(),
                    &attachments,
                )
            })
            .collect::<Vec<_>>();

        let number_of_images = swapchain.images().len();
        let descriptor_pool = DescriptorPool::new(context.clone(), number_of_images as _);

        let command_pool = CommandPool::new(context.clone(), vk::CommandPoolCreateFlags::empty());
        let transient_command_pool =
            CommandPool::new(context.clone(), vk::CommandPoolCreateFlags::TRANSIENT);

        let vertex_buffer = crate::resource::buffer::create_device_local_buffer::<u32, _>(
            context.clone(),
            transient_command_pool.pool(),
            graphics_queue,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            &vertices,
        );

        let index_buffer = crate::resource::buffer::create_device_local_buffer::<u16, _>(
            context.clone(),
            transient_command_pool.pool(),
            graphics_queue,
            vk::BufferUsageFlags::INDEX_BUFFER,
            &indices,
        );

        let size = mem::size_of::<UniformBufferObject>() as vk::DeviceSize;
        let uniform_buffers = (0..swapchain.images().len())
            .map(|_| {
                Buffer::new(
                    context.clone(),
                    size,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
            })
            .collect::<Vec<_>>();

        let descriptor_sets = descriptor_pool
            .allocate_descriptor_sets(descriptor_set_layout.layout(), uniform_buffers.len() as _);

        descriptor_pool.update_descriptor_sets(
            &descriptor_sets,
            vk::DescriptorType::UNIFORM_BUFFER,
            &uniform_buffers,
            mem::size_of::<UniformBufferObject>() as vk::DeviceSize,
        );

        let mut vulkan_swapchain = RenderState {
            command_pool,
            context,
            descriptor_pool,
            descriptor_set_layout,
            descriptor_sets,
            framebuffers,
            graphics_queue,
            index_buffer,
            number_of_indices: indices.len() as _,
            pipeline,
            present_queue,
            render_pass,
            swapchain,
            transient_command_pool,
            uniform_buffers,
            vertex_buffer,
        };

        vulkan_swapchain.create_command_buffers();
        vulkan_swapchain
    }

    fn create_command_buffers(&mut self) {
        // Allocate one command buffer per swapchain image
        self.command_pool
            .allocate_command_buffers(self.framebuffers.len() as _);
        self.command_pool
            .command_buffers()
            .iter()
            .enumerate()
            .for_each(|(index, buffer)| {
                let command_buffer = buffer;
                let framebuffer = self.framebuffers[index].framebuffer();
                self.record_render_pass(framebuffer, *command_buffer, || unsafe {
                    self.play_render_commands(
                        &self.descriptor_sets,
                        self.number_of_indices,
                        *command_buffer,
                        index,
                    );
                });
            });
    }

    fn record_render_pass<F>(
        &self,
        framebuffer: vk::Framebuffer,
        command_buffer: vk::CommandBuffer,
        mut render_action: F,
    ) where
        F: FnMut(),
    {
        // Begin the command buffer
        let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE)
            .build();
        unsafe {
            self.context
                .logical_device()
                .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                .unwrap()
        };

        let clear_values = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        }];

        let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.render_pass.render_pass())
            .framebuffer(framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.swapchain.properties().extent,
            })
            .clear_values(&clear_values)
            .build();

        unsafe {
            self.context.logical_device().cmd_begin_render_pass(
                command_buffer,
                &render_pass_begin_info,
                vk::SubpassContents::INLINE,
            );

            // Bind pipeline
            self.context.logical_device().cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.pipeline(),
            );
        }

        render_action();

        unsafe {
            // End render pass
            self.context
                .logical_device()
                .cmd_end_render_pass(command_buffer);

            // End command buffer
            self.context
                .logical_device()
                .end_command_buffer(command_buffer)
                .unwrap();
        }
    }

    unsafe fn play_render_commands(
        &self,
        descriptor_sets: &[vk::DescriptorSet],
        number_of_indices: u32,
        command_buffer: vk::CommandBuffer,
        image_index: usize,
    ) {
        // Bind pipeline
        self.context.logical_device().cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline.pipeline(),
        );

        // Bind vertex buffer
        let offsets = [0];
        let vertex_buffers = [self.vertex_buffer.buffer()];
        self.context.logical_device().cmd_bind_vertex_buffers(
            command_buffer,
            0,
            &vertex_buffers,
            &offsets,
        );

        // Bind index buffer
        self.context.logical_device().cmd_bind_index_buffer(
            command_buffer,
            self.index_buffer.buffer(),
            0,
            vk::IndexType::UINT16,
        );

        // Bind descriptor sets
        let null = [];
        self.context.logical_device().cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline.layout(),
            0,
            &descriptor_sets[image_index..=image_index],
            &null,
        );

        // Draw
        self.context.logical_device().cmd_draw_indexed(
            command_buffer,
            number_of_indices,
            1,
            0,
            0,
            0,
        );
    }

    pub fn update_uniform_buffers(
        &self,
        current_image: u32,
        swapchain_properties: &SwapchainProperties,
        start_time: Instant,
    ) {
        let elapsed_time = start_time.elapsed();
        let elapsed_time =
            elapsed_time.as_secs() as f32 + (elapsed_time.subsec_millis() as f32) / 1000_f32;

        let ubo = UniformBufferObject {
            model: glm::rotate(
                &glm::Mat4::identity(),
                (elapsed_time * 90.0).to_radians(),
                &glm::vec3(0.0, 1.0, 0.0),
            ),
            view: glm::look_at(
                &glm::vec3(0.0, 0.0, 2.0),
                &glm::vec3(0.0, 0.0, 0.0),
                &glm::vec3(0.0, 1.0, 0.0),
            ), // TODO: Make Z the up axis
            projection: glm::perspective(
                swapchain_properties.aspect_ratio(),
                90_f32.to_radians(),
                0.1_f32,
                1000_f32,
            ),
        };

        let ubos = [ubo];
        let buffer = &self.uniform_buffers[current_image as usize];
        buffer.upload_to_entire_buffer::<u32, _>(&ubos);
    }
}
