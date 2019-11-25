use ash::{
    extensions::khr::Swapchain,
    version::{DeviceV1_0, InstanceV1_0},
    vk, Device,
};
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

fn main() {
    env_logger::init();

    let vertices: [Vertex; 3] = [
        Vertex::new([0.0, -0.5], [1.0, 0.0, 0.0]),
        Vertex::new([0.5, 0.5], [0.0, 1.0, 0.0]),
        Vertex::new([-0.5, 0.5], [0.0, 0.0, 1.0]),
    ];

    let mut app = App::new(800, 600, "Vulkan Tutorial");
    let context = VulkanContext::new(app.window());
    let instance = context.instance();
    let physical_device = context.physical_device();
    let surface = context.surface();
    let surface_khr = context.surface_khr();

    // Get the memory properties of the physical device
    let memory_properties =
        unsafe { instance.get_physical_device_memory_properties(physical_device) };

    let (graphics_family_index, present_family_index) = context.queue_family_indices();

    // Need to dedup since the graphics family and presentation family
    // can have the same queue family index and
    // Vulkan does not allow passing an array containing duplicated family
    // indices.
    let mut queue_family_indices = vec![
        graphics_family_index.expect("Failed to find a graphics queue"),
        present_family_index.expect("Failed to find a present queue"),
    ];
    queue_family_indices.dedup();

    // Build an array of DeviceQueueCreateInfo,
    // one for each different family index
    let queue_create_infos = queue_family_indices
        .iter()
        .map(|index| {
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(*index)
                .queue_priorities(&[1.0f32])
                .build()
        })
        .collect::<Vec<_>>();

    // Specify device extensions
    let device_extensions = [Swapchain::name().as_ptr()];

    // Get the features of the physical device
    let device_features = vk::PhysicalDeviceFeatures::builder().build();

    // Create the device creation info using the queue creation info and available features
    let mut device_create_info_builder = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_create_infos)
        .enabled_extension_names(&device_extensions)
        .enabled_features(&device_features);

    ////// Delete this
    // Determine the required layer names
    let layer_names = debug::REQUIRED_LAYERS
        .iter()
        .map(|name| CString::new(*name).expect("Failed to build CString"))
        .collect::<Vec<_>>();

    // Determine required layer name pointers
    let layer_name_ptrs = layer_names
        .iter()
        .map(|name| name.as_ptr())
        .collect::<Vec<_>>();
    //////

    if debug::ENABLE_VALIDATION_LAYERS {
        // Add the validation layers to the list of enabled layers if validation layers are enabled
        device_create_info_builder =
            device_create_info_builder.enabled_layer_names(layer_name_ptrs.as_slice())
    }

    let device_create_info = device_create_info_builder.build();

    // Create the logical device using the physical device and device creation info
    let logical_device = unsafe {
        instance
            .create_device(physical_device, &device_create_info, None)
            .expect("Failed to create logical device.")
    };

    // Retrieve the graphics and present queues from the logical device using the graphics queue family index
    let graphics_queue =
        unsafe { logical_device.get_device_queue(graphics_family_index.unwrap(), 0) };
    let present_queue =
        unsafe { logical_device.get_device_queue(present_family_index.unwrap(), 0) };

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

    log::debug!(
        "Creating swapchain.\n\tFormat: {:?}\n\tColorSpace: {:?}\n\tPresentMode: {:?}\n\tExtent: {:?}\n\tImageCount: {}",
        surface_format.format,
        surface_format.color_space,
        present_mode,
        extent,
        image_count
    );

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

        builder = match (graphics_family_index, present_family_index) {
            (Some(graphics), Some(present)) if graphics != present => builder
                .image_sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&queue_family_indices),
            (Some(_), Some(_)) => builder.image_sharing_mode(vk::SharingMode::EXCLUSIVE),
            _ => panic!(),
        };

        builder
            .pre_transform(capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .build()
    };

    // Create the swapchain using the swapchain creation info
    let swapchain = Swapchain::new(instance, &logical_device);
    let swapchain_khr = unsafe {
        swapchain
            .create_swapchain(&swapchain_create_info, None)
            .unwrap()
    };
    let swapchain_khr_arr = [swapchain_khr];

    // Get the swapchain images
    let images = unsafe { swapchain.get_swapchain_images(swapchain_khr).unwrap() };

    // Create the swapchain image views
    let image_views = images
        .into_iter()
        .map(|image| {
            let create_info = vk::ImageViewCreateInfo::builder()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(surface_format.format)
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
        .collect::<Vec<_>>();

    let render_pass = create_render_pass(&logical_device, &swapchain_properties);

    let (pipeline, pipeline_layout) =
        create_pipeline(&logical_device, &swapchain_properties, render_pass);

    // Create one framebuffer for each image in the swapchain
    let framebuffers = create_framebuffers(
        &logical_device,
        image_views.as_slice(),
        &swapchain_properties,
        render_pass,
    );

    // Create the command pool
    let command_pool = create_command_pool(
        &logical_device,
        graphics_family_index.unwrap(),
        vk::CommandPoolCreateFlags::empty(),
    );

    // Build the transient command pool info
    let transient_command_pool = create_command_pool(
        &logical_device,
        graphics_family_index.unwrap(),
        vk::CommandPoolCreateFlags::TRANSIENT,
    );

    let buffer_size = vertices.len() * mem::size_of::<Vertex>() as usize;

    //--- BEGIN STAGING BUFFER

    // Build the staging buffer creation info
    let staging_buffer_info = vk::BufferCreateInfo::builder()
        .size(buffer_size as _)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .build();

    // Create the staging buffer
    let staging_buffer = unsafe {
        logical_device
            .create_buffer(&staging_buffer_info, None)
            .unwrap()
    };

    // Get the buffer's memory requirements
    let staging_buffer_memory_requirements =
        unsafe { logical_device.get_buffer_memory_requirements(staging_buffer) };

    // Specify the required memory properties
    let required_properties =
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

    // Determine the buffer's memory type
    let mut memory_type = 0;
    let mut found_memory_type = false;
    for index in 0..memory_properties.memory_type_count {
        if staging_buffer_memory_requirements.memory_type_bits & (1 << index) != 0
            && memory_properties.memory_types[index as usize]
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

    // Create the staging buffer allocation info
    let staging_buffer_allocation_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(staging_buffer_memory_requirements.size)
        .memory_type_index(memory_type)
        .build();

    // Allocate memory for the buffer
    let staging_buffer_memory = unsafe {
        logical_device
            .allocate_memory(&staging_buffer_allocation_info, None)
            .unwrap()
    };

    unsafe {
        // Bind the buffer memory for mapping
        logical_device
            .bind_buffer_memory(staging_buffer, staging_buffer_memory, 0)
            .unwrap();

        // Map the entire buffer buffer
        let data_pointer = logical_device
            .map_memory(
                staging_buffer_memory,
                0,
                staging_buffer_info.size,
                vk::MemoryMapFlags::empty(),
            )
            .unwrap();

        // Upload aligned staging data to the mapped buffer
        let mut align = ash::util::Align::new(
            data_pointer,
            mem::align_of::<u32>() as _,
            staging_buffer_memory_requirements.size,
        );
        align.copy_from_slice(&vertices);

        // Unmap the buffer memory
        logical_device.unmap_memory(staging_buffer_memory);
    }

    //--- END STAGING BUFFER

    //--- BEGIN VERTEX BUFFER

    // Build the vertex buffer creation info
    let vertex_buffer_info = vk::BufferCreateInfo::builder()
        .size(buffer_size as _)
        .usage(vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .build();

    // Create the vertex buffer
    let vertex_buffer = unsafe {
        logical_device
            .create_buffer(&vertex_buffer_info, None)
            .unwrap()
    };
    let vertex_buffers = [vertex_buffer];

    // Get the buffer's memory requirements
    let memory_requirements =
        unsafe { logical_device.get_buffer_memory_requirements(vertex_buffer) };

    // Specify the required memory properties
    let required_properties = vk::MemoryPropertyFlags::DEVICE_LOCAL;

    // Determine the buffer's memory type
    let mut memory_type = 0;
    let mut found_memory_type = false;
    for index in 0..memory_properties.memory_type_count {
        if memory_requirements.memory_type_bits & (1 << index) != 0
            && memory_properties.memory_types[index as usize]
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

    // Create the vertex buffer allocation info
    let vertex_buffer_allocation_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(memory_requirements.size)
        .memory_type_index(memory_type)
        .build();

    // Allocate memory for the buffer
    let vertex_buffer_memory = unsafe {
        logical_device
            .allocate_memory(&vertex_buffer_allocation_info, None)
            .unwrap()
    };

    // Bind the buffer memory for mapping
    unsafe {
        logical_device
            .bind_buffer_memory(vertex_buffer, vertex_buffer_memory, 0)
            .unwrap()
    }

    {
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

        // Define the region for the buffer copy
        let region = vk::BufferCopy {
            src_offset: 0,
            dst_offset: 0,
            size: buffer_size as _,
        };
        let regions = [region];

        // Copy the bytes of the staging buffer to the vertex buffer
        unsafe {
            logical_device.cmd_copy_buffer(command_buffer, staging_buffer, vertex_buffer, &regions)
        };

        // End command buffer recording
        unsafe { logical_device.end_command_buffer(command_buffer).unwrap() };

        // Build the submission info
        let submit_info = vk::SubmitInfo::builder()
            .command_buffers(&command_buffers)
            .build();
        let submit_info_arr = [submit_info];

        // Submit the command buffer
        unsafe {
            logical_device
                .queue_submit(graphics_queue, &submit_info_arr, vk::Fence::null())
                .unwrap()
        };

        // Wait for the command buffer to be executed
        unsafe { logical_device.queue_wait_idle(graphics_queue).unwrap() };

        // Free the command buffer
        unsafe { logical_device.free_command_buffers(command_pool, &command_buffers) };
    }

    // Free the staging buffer
    unsafe { logical_device.destroy_buffer(staging_buffer, None) };

    // Free the staging buffer memory
    unsafe { logical_device.free_memory(staging_buffer_memory, None) };

    //--- END VERTEX BUFFER

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
        .zip(framebuffers.iter())
        .for_each(|(buffer, framebuffer)| {
            let command_buffer = *buffer;

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
                    .framebuffer(*framebuffer)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent,
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
                )
            };

            // Bind vertex buffer
            unsafe {
                logical_device.cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &[0])
            };

            // Draw
            unsafe { logical_device.cmd_draw(command_buffer, 3, 1, 0, 0) };

            // End render pass
            unsafe { logical_device.cmd_end_render_pass(command_buffer) };

            // End command buffer
            unsafe { logical_device.end_command_buffer(command_buffer).unwrap() };
        });

    // Create semaphores
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

    let mut current_frame = 0;

    app.run(|| {
        let image_available_semaphore = image_available_semaphores[current_frame];
        let image_available_semaphores = [image_available_semaphore];

        let render_finished_semaphore = render_finished_semaphores[current_frame];
        let render_finished_semaphores = [render_finished_semaphore];

        let in_flight_fence = in_flight_fences[current_frame];
        let in_flight_fences = [in_flight_fence];

        unsafe {
            logical_device
                .wait_for_fences(&in_flight_fences, true, std::u64::MAX)
                .unwrap();
            logical_device.reset_fences(&in_flight_fences).unwrap();
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
            logical_device
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

    unsafe { logical_device.device_wait_idle().unwrap() };

    // Free objects
    log::debug!("Dropping application");
    unsafe {
        in_flight_fences
            .iter()
            .for_each(|fence| logical_device.destroy_fence(*fence, None));
        render_finished_semaphores
            .iter()
            .for_each(|semaphore| logical_device.destroy_semaphore(*semaphore, None));
        image_available_semaphores
            .iter()
            .for_each(|semaphore| logical_device.destroy_semaphore(*semaphore, None));
        logical_device.destroy_buffer(vertex_buffer, None);
        logical_device.free_memory(vertex_buffer_memory, None);
        logical_device.destroy_command_pool(command_pool, None);
        logical_device.destroy_command_pool(transient_command_pool, None);
        framebuffers
            .iter()
            .for_each(|f| logical_device.destroy_framebuffer(*f, None));
        logical_device.destroy_pipeline(pipeline, None);
        logical_device.destroy_pipeline_layout(pipeline_layout, None);
        logical_device.destroy_render_pass(render_pass, None);
        image_views
            .iter()
            .for_each(|v| logical_device.destroy_image_view(*v, None));
        swapchain.destroy_swapchain(swapchain_khr, None);
        logical_device.destroy_device(None);
    }
}

fn create_render_pass(
    logical_device: &Device,
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

fn create_pipeline(
    logical_device: &Device,
    swapchain_properties: &SwapchainProperties,
    render_pass: vk::RenderPass,
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
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder()
        // .set_layouts() // needed for uniforms in shaders
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
    logical_device: &Device,
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
    logical_device: &Device,
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
