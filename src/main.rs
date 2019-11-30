use ash::{
    extensions::khr::{Surface, Swapchain},
    version::DeviceV1_0,
    vk,
};
use nalgebra_glm as glm;
use std::{ffi::CString, mem};

mod app;
mod context;
mod debug;
mod shader;
mod surface;
mod swapchain;
mod vertex;

use app::App;
use context::VulkanContext;
use swapchain::{SwapchainProperties, SwapchainSupportDetails};
use vertex::Vertex;

// The maximum number of frames that can be rendered simultaneously
pub const MAX_FRAMES_IN_FLIGHT: u32 = 2;

#[derive(Debug, Clone, Copy)]
struct UniformBufferObject {
    model: glm::Mat4,
    view: glm::Mat4,
    projection: glm::Mat4,
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
    let context = VulkanContext::new(app.window());

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

    let (swapchain, swapchain_khr, swapchain_properties, images) = create_swapchain(
        context.instance(),
        context.physical_device(),
        context.logical_device(),
        &context.surface(),
        context.surface_khr(),
        context.graphics_queue_family_index(),
        context.present_queue_family_index(),
    );
    let swapchain_khr_arr = [swapchain_khr];
    let image_views = create_image_views(context.logical_device(), &swapchain_properties, &images);
    let render_pass = create_render_pass(context.logical_device(), &swapchain_properties);
    let descriptor_set_layout = create_descriptor_set_layout(context.logical_device());
    let (pipeline, pipeline_layout) = create_pipeline(
        context.logical_device(),
        &swapchain_properties,
        render_pass,
        descriptor_set_layout,
    );
    let framebuffers = create_framebuffers(
        context.logical_device(),
        image_views.as_slice(),
        &swapchain_properties,
        render_pass,
    );

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
        command_pool,
        graphics_queue,
        vk::BufferUsageFlags::VERTEX_BUFFER,
        &vertices,
    );
    let vertex_buffers = [vertex_buffer];

    let (index_buffer, index_buffer_memory) = create_device_local_buffer::<u16, _>(
        context.logical_device(),
        context.physical_device_memory_properties(),
        command_pool,
        graphics_queue,
        vk::BufferUsageFlags::INDEX_BUFFER,
        &indices,
    );

    // TODO: Create a uniform buffer

    let number_of_images = images.len();
    let (uniform_buffers, uniform_buffer_memory_list) = create_uniform_buffers(
        context.logical_device(),
        context.physical_device_memory_properties(),
        number_of_images,
    );
    let descriptor_pool = create_descriptor_pool(context.logical_device(), number_of_images as _);
    let descriptor_sets = create_descriptor_sets(
        context.logical_device(),
        descriptor_pool,
        descriptor_set_layout,
        &uniform_buffers,
    );

    let command_buffers = create_command_buffers(
        context.logical_device(),
        command_pool,
        &framebuffers,
        render_pass,
        pipeline,
        pipeline_layout,
        &descriptor_sets,
        &swapchain_properties,
        &vertex_buffers,
        index_buffer,
        indices.len() as u32,
    );

    let (image_available_semaphores, render_finished_semaphores, in_flight_fences) =
        create_semaphores_and_fences(context.logical_device());

    let mut current_frame = 0;

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
            swapchain
                .acquire_next_image(
                    swapchain_khr,
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
            swapchain_properties,
            &uniform_buffer_memory_list,
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
            swapchain
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
        framebuffers
            .iter()
            .for_each(|f| context.logical_device().destroy_framebuffer(*f, None));
        context.logical_device().destroy_pipeline(pipeline, None);
        context
            .logical_device()
            .destroy_pipeline_layout(pipeline_layout, None);
        context
            .logical_device()
            .destroy_render_pass(render_pass, None);
        image_views
            .iter()
            .for_each(|v| context.logical_device().destroy_image_view(*v, None));
        swapchain.destroy_swapchain(swapchain_khr, None);
        context
            .logical_device()
            .destroy_descriptor_pool(descriptor_pool, None);
        context
            .logical_device()
            .destroy_descriptor_set_layout(descriptor_set_layout, None);
        context.logical_device().destroy_device(None);
    }
}

fn create_render_pass(
    logical_device: &ash::Device,
    swapchain_properties: &SwapchainProperties,
) -> vk::RenderPass {
    let attachment_description = vk::AttachmentDescription::builder()
        .format(swapchain_properties.format.format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .build();
    let attachment_descriptions = [attachment_description];

    let attachment_reference = vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .build();
    let attachment_references = [attachment_reference];

    let subpass_description = vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&attachment_references)
        .build();
    let subpass_descriptions = [subpass_description];

    let subpass_dependency = vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        )
        .build();
    let subpass_dependencies = [subpass_dependency];

    let render_pass_info = vk::RenderPassCreateInfo::builder()
        .attachments(&attachment_descriptions)
        .subpasses(&subpass_descriptions)
        .dependencies(&subpass_dependencies)
        .build();

    unsafe {
        logical_device
            .create_render_pass(&render_pass_info, None)
            .unwrap()
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

fn create_pipeline(
    logical_device: &ash::Device,
    swapchain_properties: &SwapchainProperties,
    render_pass: vk::RenderPass,
    descriptor_set_layout: vk::DescriptorSetLayout,
) -> (vk::Pipeline, vk::PipelineLayout) {
    // Define the entry point for shaders
    let entry_point_name = &CString::new("main").unwrap();

    // Create the vertex shader module
    let vertex_shader_module =
        shader::create_shader_from_file("shaders/shader.vert.spv", logical_device);

    let vertex_shader_state_info = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vertex_shader_module)
        .name(entry_point_name)
        .build();

    // Create the fragment shader module
    let fragment_shader_module =
        shader::create_shader_from_file("shaders/shader.frag.spv", logical_device);

    let fragment_shader_state_info = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::FRAGMENT)
        .module(fragment_shader_module)
        .name(entry_point_name)
        .build();

    let shader_state_info = [vertex_shader_state_info, fragment_shader_state_info];

    let descriptions = [Vertex::get_binding_description()];
    let attributes = Vertex::get_attribute_descriptions();

    // Build vertex input creation info
    let vertex_input_create_info = vk::PipelineVertexInputStateCreateInfo::builder()
        .vertex_binding_descriptions(&descriptions)
        .vertex_attribute_descriptions(&attributes)
        .build();

    // Build input assembly creation info
    let input_assembly_create_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false)
        .build();

    // Create a viewport
    let viewport = vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: swapchain_properties.extent.width as _,
        height: swapchain_properties.extent.height as _,
        min_depth: 0.0,
        max_depth: 1.0,
    };
    let viewports = [viewport];

    // Create a stencil
    let scissor = vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: swapchain_properties.extent,
    };
    let scissors = [scissor];

    // Build the viewport info using the viewport and stencil
    let viewport_create_info = vk::PipelineViewportStateCreateInfo::builder()
        .viewports(&viewports)
        .scissors(&scissors)
        .build();

    // Build the rasterizer info
    let rasterizer_create_info = vk::PipelineRasterizationStateCreateInfo::builder()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::CLOCKWISE)
        .depth_bias_enable(false)
        .depth_bias_constant_factor(0.0)
        .depth_bias_clamp(0.0)
        .depth_bias_slope_factor(0.0)
        .build();

    // Create the multisampling info for the pipline
    let multisampling_create_info = vk::PipelineMultisampleStateCreateInfo::builder()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlags::TYPE_1)
        .min_sample_shading(1.0)
        // .sample_mask()
        .alpha_to_coverage_enable(false)
        .alpha_to_one_enable(false)
        .build();

    // Create the color blend attachment
    let color_blend_attachment = vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::all())
        .blend_enable(false)
        .src_color_blend_factor(vk::BlendFactor::ONE)
        .dst_color_blend_factor(vk::BlendFactor::ZERO)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD)
        .build();
    let color_blend_attachments = [color_blend_attachment];

    // Build the color blending info using the color blend attachment
    let color_blending_info = vk::PipelineColorBlendStateCreateInfo::builder()
        .logic_op_enable(false)
        .logic_op(vk::LogicOp::COPY)
        .attachments(&color_blend_attachments)
        .blend_constants([0.0, 0.0, 0.0, 0.0])
        .build();

    // Build the pipeline layout info
    let descriptor_set_layouts = [descriptor_set_layout];
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(&descriptor_set_layouts) // needed for uniforms in shaders
        // .push_constant_ranges()
        .build();

    // Create the pipeline layout
    let pipeline_layout = unsafe {
        logical_device
            .create_pipeline_layout(&pipeline_layout_info, None)
            .unwrap()
    };

    // Create the pipeline info
    let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
        .stages(&shader_state_info)
        .vertex_input_state(&vertex_input_create_info)
        .input_assembly_state(&input_assembly_create_info)
        .viewport_state(&viewport_create_info)
        .rasterization_state(&rasterizer_create_info)
        .multisample_state(&multisampling_create_info)
        //.depth_stencil_state() // not using depth/stencil tests
        .color_blend_state(&color_blending_info)
        //.dynamic_state // no dynamic states
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0)
        .build();

    let pipeline_info_arr = [pipeline_info];

    // Create the pipeline
    let pipeline = unsafe {
        logical_device
            .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_info_arr, None)
            .unwrap()[0]
    };

    // Delete shader modules
    unsafe {
        logical_device.destroy_shader_module(vertex_shader_module, None);
        logical_device.destroy_shader_module(fragment_shader_module, None);
    };

    (pipeline, pipeline_layout)
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

fn create_framebuffers(
    logical_device: &ash::Device,
    image_views: &[vk::ImageView],
    swapchain_properties: &SwapchainProperties,
    render_pass: vk::RenderPass,
) -> Vec<vk::Framebuffer> {
    // Create one framebuffer for each image in the swapchain
    image_views
        .iter()
        .map(|view| [*view])
        .map(|attachments| {
            let framebuffer_info = vk::FramebufferCreateInfo::builder()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(swapchain_properties.extent.width)
                .height(swapchain_properties.extent.height)
                .layers(1)
                .build();
            unsafe {
                logical_device
                    .create_framebuffer(&framebuffer_info, None)
                    .unwrap()
            }
        })
        .collect::<Vec<_>>()
}

fn create_swapchain(
    instance: &ash::Instance,
    physical_device: ash::vk::PhysicalDevice,
    logical_device: &ash::Device,
    surface: &Surface,
    surface_khr: ash::vk::SurfaceKHR,
    graphics_queue_family_index: u32,
    present_queue_family_index: u32,
) -> (
    Swapchain,
    vk::SwapchainKHR,
    SwapchainProperties,
    Vec<vk::Image>,
) {
    let swapchain_support_details =
        SwapchainSupportDetails::new(physical_device, surface, surface_khr);
    let capabilities = &swapchain_support_details.capabilities;

    let swapchain_properties = swapchain_support_details.suitable_properties([800, 600]);
    let surface_format = swapchain_properties.format;
    let present_mode = swapchain_properties.present_mode;
    let extent = swapchain_properties.extent;

    // Choose the number of images to use in the swapchain
    let image_count = {
        let max = capabilities.max_image_count;
        let mut preferred = capabilities.min_image_count + 1;
        if max > 0 && preferred > max {
            preferred = max;
        }
        preferred
    };

    // Build the swapchain creation info
    let swapchain_create_info = {
        let mut builder = vk::SwapchainCreateInfoKHR::builder()
            .surface(surface_khr)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT);

        let mut queue_family_indices =
            vec![graphics_queue_family_index, present_queue_family_index];
        queue_family_indices.dedup();

        builder = if graphics_queue_family_index != present_queue_family_index {
            builder
                .image_sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&queue_family_indices)
        } else {
            builder.image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        };

        builder
            .pre_transform(capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .build()
    };

    // Create the swapchain using the swapchain creation info
    let swapchain = Swapchain::new(instance, logical_device);
    let swapchain_khr = unsafe {
        swapchain
            .create_swapchain(&swapchain_create_info, None)
            .unwrap()
    };

    log::debug!(
        "Creating swapchain.\n\tFormat: {:?}\n\tColorSpace: {:?}\n\tPresentMode: {:?}\n\tExtent: {:?}\n\tImageCount: {}",
        surface_format.format,
        surface_format.color_space,
        present_mode,
        extent,
        image_count
    );

    // Get the swapchain images
    let images = unsafe { swapchain.get_swapchain_images(swapchain_khr).unwrap() };

    (swapchain, swapchain_khr, swapchain_properties, images)
}

fn create_image_views(
    logical_device: &ash::Device,
    swapchain_properties: &SwapchainProperties,
    images: &[vk::Image],
) -> Vec<vk::ImageView> {
    images
        .iter()
        .map(|image| {
            let create_info = vk::ImageViewCreateInfo::builder()
                .image(*image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(swapchain_properties.format.format)
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::IDENTITY,
                    b: vk::ComponentSwizzle::IDENTITY,
                    a: vk::ComponentSwizzle::IDENTITY,
                })
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .build();

            unsafe {
                logical_device
                    .create_image_view(&create_info, None)
                    .unwrap()
            }
        })
        .collect::<Vec<_>>()
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

fn create_buffer(
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

fn create_device_local_buffer<A, T: Copy>(
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

fn execute_command_once<F: FnOnce(vk::CommandBuffer)>(
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

fn copy_buffer(
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
) {
    let aspect_ratio =
        swapchain_properties.extent.width as f32 / swapchain_properties.extent.height as f32;
    let ubo = UniformBufferObject {
        model: glm::Mat4::identity(),
        view: glm::look_at(
            &glm::vec3(2.0, 2.0, 2.0),
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
