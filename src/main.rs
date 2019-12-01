use ash::{version::DeviceV1_0, vk};
use nalgebra_glm as glm;
use std::{mem, time::Instant};

mod app;
mod buffer;
mod context;
mod debug;
mod shader;
mod surface;
mod swapchain;
mod vertex;

use app::App;
use buffer::{create_buffer, create_device_local_buffer};
use context::VulkanContext;
use std::sync::Arc;
use swapchain::{SwapchainProperties, UniformBufferObject, VulkanSwapchain};
use vertex::Vertex;

// The maximum number of frames that can be rendered simultaneously
pub const MAX_FRAMES_IN_FLIGHT: u32 = 2;

fn main() {
    env_logger::init();

    let vertices: [Vertex; 4] = [
        Vertex::new([-0.5, -0.5], [1.0, 0.0, 0.0]),
        Vertex::new([0.5, -0.5], [0.0, 1.0, 0.0]),
        Vertex::new([0.5, 0.5], [0.0, 0.0, 1.0]),
        Vertex::new([-0.5, 0.5], [1.0, 1.0, 1.0]),
    ];

    let indices: [u16; 6] = [0, 1, 2, 2, 3, 0];

    let mut app = App::new(800, 600, "Vulkan Tutorial");
    let context = Arc::new(VulkanContext::new(app.window()));

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

    // TODO: create vulkanswapchain
    let vulkan_swapchain = VulkanSwapchain::new(context.clone());

    let command_pool = create_command_pool(
        context.logical_device(),
        context.graphics_queue_family_index(),
        vk::CommandPoolCreateFlags::empty(),
    );

    let transient_command_pool = create_command_pool(
        context.logical_device(),
        context.graphics_queue_family_index(),
        vk::CommandPoolCreateFlags::TRANSIENT,
    );

    let (vertex_buffer, vertex_buffer_memory) = create_device_local_buffer::<u32, _>(
        context.logical_device(),
        context.physical_device_memory_properties(),
        transient_command_pool,
        graphics_queue,
        vk::BufferUsageFlags::VERTEX_BUFFER,
        &vertices,
    );
    let vertex_buffers = [vertex_buffer];

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
        vulkan_swapchain.images.len(),
    );

    let descriptor_sets = create_descriptor_sets(
        context.logical_device(),
        vulkan_swapchain.descriptor_pool,
        vulkan_swapchain.descriptor_set_layout,
        &uniform_buffers,
    );

    let command_buffers = create_command_buffers(
        context.logical_device(),
        command_pool,
        &vulkan_swapchain.framebuffers,
        vulkan_swapchain.render_pass,
        vulkan_swapchain.pipeline,
        vulkan_swapchain.pipeline_layout,
        &descriptor_sets,
        &vulkan_swapchain.swapchain_properties,
        &vertex_buffers,
        index_buffer,
        indices.len() as u32,
    );

    let (image_available_semaphores, render_finished_semaphores, in_flight_fences) =
        create_semaphores_and_fences(context.logical_device());

    let mut current_frame = 0;
    let start_time = Instant::now();

    let swapchain_khr_arr = [vulkan_swapchain.swapchain_khr];

    app.run(|| {
        let image_available_semaphore = image_available_semaphores[current_frame];
        let image_available_semaphores = [image_available_semaphore];

        let render_finished_semaphore = render_finished_semaphores[current_frame];
        let render_finished_semaphores = [render_finished_semaphore];

        let in_flight_fence = in_flight_fences[current_frame];
        let in_flight_fences = [in_flight_fence];

        unsafe {
            context
                .logical_device()
                .wait_for_fences(&in_flight_fences, true, std::u64::MAX)
                .unwrap();
            context
                .logical_device()
                .reset_fences(&in_flight_fences)
                .unwrap();
        }

        // Acqure the next image from the swapchain
        let image_index = unsafe {
            vulkan_swapchain
                .swapchain
                .acquire_next_image(
                    vulkan_swapchain.swapchain_khr,
                    std::u64::MAX,
                    image_available_semaphore,
                    vk::Fence::null(),
                )
                .unwrap()
                .0
        };
        let image_indices = [image_index];

        update_uniform_buffers(
            context.logical_device(),
            image_index,
            vulkan_swapchain.swapchain_properties,
            &uniform_buffer_memory_list,
            start_time,
        );

        // Submit the command buffer
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers_to_use = [command_buffers[image_index as usize]];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(&image_available_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers_to_use)
            .signal_semaphores(&render_finished_semaphores)
            .build();
        let submit_info_arr = [submit_info];

        unsafe {
            context
                .logical_device()
                .queue_submit(graphics_queue, &submit_info_arr, in_flight_fence)
                .unwrap()
        };

        // Present the rendered image
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&render_finished_semaphores)
            .swapchains(&swapchain_khr_arr)
            .image_indices(&image_indices)
            .build();

        unsafe {
            vulkan_swapchain
                .swapchain
                .queue_present(present_queue, &present_info)
                .unwrap()
        };

        current_frame += (1 + current_frame) % MAX_FRAMES_IN_FLIGHT as usize;
    });

    unsafe { context.logical_device().device_wait_idle().unwrap() };

    // Free objects
    log::debug!("Dropping application");
    unsafe {
        in_flight_fences
            .iter()
            .for_each(|fence| context.logical_device().destroy_fence(*fence, None));
        render_finished_semaphores
            .iter()
            .for_each(|semaphore| context.logical_device().destroy_semaphore(*semaphore, None));
        image_available_semaphores
            .iter()
            .for_each(|semaphore| context.logical_device().destroy_semaphore(*semaphore, None));
        uniform_buffer_memory_list
            .iter()
            .for_each(|m| context.logical_device().free_memory(*m, None));
        uniform_buffers
            .iter()
            .for_each(|b| context.logical_device().destroy_buffer(*b, None));
        context.logical_device().destroy_buffer(vertex_buffer, None);
        context
            .logical_device()
            .free_memory(vertex_buffer_memory, None);
        context.logical_device().destroy_buffer(index_buffer, None);
        context
            .logical_device()
            .free_memory(index_buffer_memory, None);
        context
            .logical_device()
            .destroy_command_pool(command_pool, None);
        context
            .logical_device()
            .destroy_command_pool(transient_command_pool, None);
    }
}

fn create_command_pool(
    logical_device: &ash::Device,
    graphics_queue_family_index: u32,
    flags: vk::CommandPoolCreateFlags,
) -> vk::CommandPool {
    let command_pool_info = vk::CommandPoolCreateInfo::builder()
        .queue_family_index(graphics_queue_family_index)
        .flags(flags)
        .build();

    unsafe {
        logical_device
            .create_command_pool(&command_pool_info, None)
            .unwrap()
    }
}

fn create_command_buffers(
    logical_device: &ash::Device,
    command_pool: ash::vk::CommandPool,
    framebuffers: &[ash::vk::Framebuffer],
    render_pass: vk::RenderPass,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_sets: &[vk::DescriptorSet],
    swapchain_properties: &SwapchainProperties,
    vertex_buffers: &[vk::Buffer],
    index_buffer: vk::Buffer,
    number_of_indices: u32,
) -> Vec<ash::vk::CommandBuffer> {
    // Build the command buffer allocation info
    let allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(framebuffers.len() as _)
        .build();

    // Allocate one command buffer per swapchain image
    let command_buffers = unsafe {
        logical_device
            .allocate_command_buffers(&allocate_info)
            .unwrap()
    };

    command_buffers
        .iter()
        .enumerate()
        .for_each(|(index, buffer)| {
            let command_buffer = *buffer;
            let framebuffer = framebuffers[index];

            // Begin the command buffer
            {
                let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE)
                    .build();
                unsafe {
                    logical_device
                        .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                        .unwrap()
                };
            }

            // Begin the render pass
            {
                let clear_values = [vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                }];

                let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                    .render_pass(render_pass)
                    .framebuffer(framebuffer)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: swapchain_properties.extent,
                    })
                    .clear_values(&clear_values)
                    .build();

                unsafe {
                    logical_device.cmd_begin_render_pass(
                        command_buffer,
                        &render_pass_begin_info,
                        vk::SubpassContents::INLINE,
                    )
                };
            }

            // Bind pipeline
            unsafe {
                logical_device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline,
                );

                // Bind vertex buffer
                let offsets = [0];
                logical_device.cmd_bind_vertex_buffers(
                    command_buffer,
                    0,
                    &vertex_buffers,
                    &offsets,
                );

                // Bind index buffer
                logical_device.cmd_bind_index_buffer(
                    command_buffer,
                    index_buffer,
                    0,
                    vk::IndexType::UINT16,
                );

                // Bind descriptor sets
                let null = [];
                logical_device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline_layout,
                    0,
                    &descriptor_sets[index..=index],
                    &null,
                );

                // Draw
                logical_device.cmd_draw_indexed(command_buffer, number_of_indices, 1, 0, 0, 0);

                // End render pass
                logical_device.cmd_end_render_pass(command_buffer);

                // End command buffer
                logical_device.end_command_buffer(command_buffer).unwrap();
            };
        });

    command_buffers
}

fn create_semaphores_and_fences(
    logical_device: &ash::Device,
) -> (Vec<vk::Semaphore>, Vec<vk::Semaphore>, Vec<vk::Fence>) {
    let mut image_available_semaphores = Vec::new();
    let mut render_finished_semaphores = Vec::new();
    let mut in_flight_fences = Vec::new();
    for _ in 0..MAX_FRAMES_IN_FLIGHT {
        let image_available_semaphore = {
            let semaphore_info = vk::SemaphoreCreateInfo::builder().build();
            unsafe {
                logical_device
                    .create_semaphore(&semaphore_info, None)
                    .unwrap()
            }
        };
        image_available_semaphores.push(image_available_semaphore);

        let render_finished_semaphore = {
            let semaphore_info = vk::SemaphoreCreateInfo::builder().build();
            unsafe {
                logical_device
                    .create_semaphore(&semaphore_info, None)
                    .unwrap()
            }
        };
        render_finished_semaphores.push(render_finished_semaphore);

        let in_flight_fence = {
            let fence_info = vk::FenceCreateInfo::builder()
                .flags(vk::FenceCreateFlags::SIGNALED)
                .build();
            unsafe { logical_device.create_fence(&fence_info, None).unwrap() }
        };
        in_flight_fences.push(in_flight_fence);
    }

    (
        image_available_semaphores,
        render_finished_semaphores,
        in_flight_fences,
    )
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

fn update_uniform_buffers(
    logical_device: &ash::Device,
    current_image: u32,
    swapchain_properties: SwapchainProperties,
    buffer_memory_list: &[vk::DeviceMemory],
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

    let buffer_memory = buffer_memory_list[current_image as usize];
    let buffer_memory_size = mem::size_of::<UniformBufferObject>() as vk::DeviceSize;

    unsafe {
        let data_pointer = logical_device
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
        logical_device.unmap_memory(buffer_memory);
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
