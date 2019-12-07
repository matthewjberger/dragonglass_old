use crate::{
    buffer::{create_buffer, create_device_local_buffer},
    context::VulkanContext,
    core::{Swapchain, SwapchainProperties},
    render::{Framebuffer, GraphicsPipeline, RenderPass},
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

pub struct VulkanSwapchain {
    context: Arc<VulkanContext>,
    pub command_buffers: Vec<vk::CommandBuffer>,
    pub command_pool: vk::CommandPool,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub framebuffers: Vec<Framebuffer>,
    pub graphics_queue: vk::Queue,
    pub index_buffer: vk::Buffer,
    pub index_buffer_memory: vk::DeviceMemory,
    pub number_of_indices: u32,
    pub pipeline: GraphicsPipeline,
    pub present_queue: vk::Queue,
    pub render_pass: RenderPass,
    pub swapchain: Swapchain,
    pub transient_command_pool: vk::CommandPool,
    pub uniform_buffer_memory_list: Vec<vk::DeviceMemory>,
    pub uniform_buffers: Vec<vk::Buffer>,
    pub vertex_buffer: vk::Buffer,
    pub vertex_buffer_memory: vk::DeviceMemory,
}

impl VulkanSwapchain {
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

        let descriptor_set_layout = create_descriptor_set_layout(context.logical_device());
        let pipeline = GraphicsPipeline::new(
            context.clone(),
            swapchain.properties(),
            render_pass.render_pass(),
            descriptor_set_layout,
        );

        // Create one framebuffer for each image in the swapchain
        let framebuffers = swapchain
            .image_views()
            .iter()
            .map(|view| [*view])
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
        let descriptor_pool =
            create_descriptor_pool(context.logical_device(), number_of_images as _);

        let command_pool = create_command_pool(&context, vk::CommandPoolCreateFlags::empty());
        let transient_command_pool =
            create_command_pool(&context, vk::CommandPoolCreateFlags::TRANSIENT);

        let (vertex_buffer, vertex_buffer_memory) = create_device_local_buffer::<u32, _>(
            context.logical_device(),
            context.physical_device_memory_properties(),
            transient_command_pool,
            graphics_queue,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            &vertices,
        );

        let (index_buffer, index_buffer_memory) = create_device_local_buffer::<u16, _>(
            context.logical_device(),
            context.physical_device_memory_properties(),
            transient_command_pool,
            graphics_queue,
            vk::BufferUsageFlags::INDEX_BUFFER,
            &indices,
        );

        let (uniform_buffers, uniform_buffer_memory_list) = create_uniform_buffers(
            context.logical_device(),
            context.physical_device_memory_properties(),
            swapchain.images().len(),
        );

        let descriptor_sets = create_descriptor_sets(
            context.logical_device(),
            descriptor_pool,
            descriptor_set_layout,
            &uniform_buffers,
        );

        let mut vulkan_swapchain = VulkanSwapchain {
            command_buffers: Vec::new(),
            command_pool,
            context,
            descriptor_pool,
            descriptor_set_layout,
            descriptor_sets,
            framebuffers,
            graphics_queue,
            index_buffer,
            index_buffer_memory,
            number_of_indices: indices.len() as _,
            pipeline,
            present_queue,
            render_pass,
            swapchain,
            transient_command_pool,
            uniform_buffer_memory_list,
            uniform_buffers,
            vertex_buffer,
            vertex_buffer_memory,
        };

        vulkan_swapchain.command_buffers = vulkan_swapchain.create_command_buffers();
        vulkan_swapchain
    }

    fn cleanup_swapchain(&self) {
        let logical_device = self.context.logical_device();
        unsafe {
            logical_device.free_command_buffers(self.command_pool, &self.command_buffers);
        }
    }

    fn create_command_buffers(&self) -> Vec<ash::vk::CommandBuffer> {
        // Build the command buffer allocation info
        let allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(self.framebuffers.len() as _)
            .build();

        // Allocate one command buffer per swapchain image
        let command_buffers = unsafe {
            self.context
                .logical_device()
                .allocate_command_buffers(&allocate_info)
                .unwrap()
        };

        command_buffers
            .iter()
            .enumerate()
            .for_each(|(index, buffer)| {
                let command_buffer = *buffer;
                let framebuffer = self.framebuffers[index].framebuffer();
                self.record_render_pass(framebuffer, command_buffer, || unsafe {
                    self.play_render_commands(
                        &self.descriptor_sets,
                        self.number_of_indices,
                        command_buffer,
                        index,
                    );
                });
            });

        command_buffers
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
        let vertex_buffers = [self.vertex_buffer];
        self.context.logical_device().cmd_bind_vertex_buffers(
            command_buffer,
            0,
            &vertex_buffers,
            &offsets,
        );

        // Bind index buffer
        self.context.logical_device().cmd_bind_index_buffer(
            command_buffer,
            self.index_buffer,
            0,
            vk::IndexType::UINT16,
        );

        // Bind descriptor sets
        let null = [];
        self.context.logical_device().cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline.pipeline_layout(),
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

        let aspect_ratio =
            swapchain_properties.extent.width as f32 / swapchain_properties.extent.height as f32;
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
            projection: glm::perspective(aspect_ratio, 90_f32.to_radians(), 0.1_f32, 1000_f32),
        };

        let ubos = [ubo];

        let buffer_memory = self.uniform_buffer_memory_list[current_image as usize];
        let buffer_memory_size = mem::size_of::<UniformBufferObject>() as vk::DeviceSize;

        unsafe {
            let data_pointer = self
                .context
                .logical_device()
                .map_memory(
                    buffer_memory,
                    0,
                    buffer_memory_size,
                    vk::MemoryMapFlags::empty(),
                )
                .unwrap();
            let mut align = ash::util::Align::new(
                data_pointer,
                mem::align_of::<f32>() as _,
                buffer_memory_size,
            );
            align.copy_from_slice(&ubos);
            self.context.logical_device().unmap_memory(buffer_memory);
        }
    }
}

impl Drop for VulkanSwapchain {
    fn drop(&mut self) {
        self.cleanup_swapchain();
        let logical_device = self.context.logical_device();
        unsafe {
            logical_device.destroy_descriptor_pool(self.descriptor_pool, None);
            logical_device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);

            self.uniform_buffer_memory_list
                .iter()
                .for_each(|m| logical_device.free_memory(*m, None));
            self.uniform_buffers
                .iter()
                .for_each(|b| logical_device.destroy_buffer(*b, None));

            logical_device.destroy_buffer(self.vertex_buffer, None);
            logical_device.free_memory(self.vertex_buffer_memory, None);

            logical_device.destroy_buffer(self.index_buffer, None);
            logical_device.free_memory(self.index_buffer_memory, None);

            logical_device.destroy_command_pool(self.command_pool, None);
            logical_device.destroy_command_pool(self.transient_command_pool, None);
        }
    }
}

fn create_descriptor_set_layout(logical_device: &ash::Device) -> vk::DescriptorSetLayout {
    let bindings = UniformBufferObject::get_descriptor_set_layout_bindings();
    let layout_info = vk::DescriptorSetLayoutCreateInfo::builder()
        .bindings(&bindings)
        .build();

    unsafe {
        logical_device
            .create_descriptor_set_layout(&layout_info, None)
            .unwrap()
    }
}

fn create_descriptor_pool(logical_device: &ash::Device, size: u32) -> vk::DescriptorPool {
    let pool_size = vk::DescriptorPoolSize {
        ty: vk::DescriptorType::UNIFORM_BUFFER,
        descriptor_count: size,
    };
    let pool_sizes = [pool_size];

    let pool_info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(&pool_sizes)
        .max_sets(size)
        .build();

    unsafe {
        logical_device
            .create_descriptor_pool(&pool_info, None)
            .unwrap()
    }
}

fn create_command_pool(
    context: &VulkanContext,
    flags: vk::CommandPoolCreateFlags,
) -> vk::CommandPool {
    let command_pool_info = vk::CommandPoolCreateInfo::builder()
        .queue_family_index(context.graphics_queue_family_index())
        .flags(flags)
        .build();

    unsafe {
        context
            .logical_device()
            .create_command_pool(&command_pool_info, None)
            .unwrap()
    }
}

fn create_descriptor_sets(
    logical_device: &ash::Device,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    uniform_buffers: &[vk::Buffer],
) -> Vec<vk::DescriptorSet> {
    let layouts = (0..uniform_buffers.len())
        .map(|_| layout)
        .collect::<Vec<_>>();
    let allocation_info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(pool)
        .set_layouts(&layouts)
        .build();
    let descriptor_sets = unsafe {
        logical_device
            .allocate_descriptor_sets(&allocation_info)
            .unwrap()
    };

    descriptor_sets
        .iter()
        .zip(uniform_buffers.iter())
        .for_each(|(set, buffer)| {
            let buffer_info = vk::DescriptorBufferInfo::builder()
                .buffer(*buffer)
                .offset(0)
                .range(mem::size_of::<UniformBufferObject>() as vk::DeviceSize)
                .build();
            let buffer_infos = [buffer_info];

            let descriptor_write = vk::WriteDescriptorSet::builder()
                .dst_set(*set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&buffer_infos)
                .build();
            let descriptor_writes = [descriptor_write];
            let null = [];

            unsafe { logical_device.update_descriptor_sets(&descriptor_writes, &null) }
        });

    descriptor_sets
}

fn create_uniform_buffers(
    logical_device: &ash::Device,
    physical_device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
    count: usize,
) -> (Vec<vk::Buffer>, Vec<vk::DeviceMemory>) {
    let size = mem::size_of::<UniformBufferObject>() as vk::DeviceSize;
    let mut buffers = Vec::new();
    let mut memory_list = Vec::new();

    for _ in 0..count {
        let (buffer, memory, _) = create_buffer(
            logical_device,
            size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            physical_device_memory_properties,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );
        buffers.push(buffer);
        memory_list.push(memory);
    }

    (buffers, memory_list)
}
