use ash::{
    extensions::{
        ext::DebugUtils,
        khr::{Surface, Swapchain},
    },
    version::{DeviceV1_0, EntryV1_0, InstanceV1_0},
    vk, vk_make_version,
};
use std::{
    ffi::{CStr, CString},
    mem,
};
use winit::{dpi::LogicalSize, Event, EventsLoop, VirtualKeyCode, WindowEvent};

mod debug;
mod surface;
mod vertex;

use vertex::Vertex;

// The maximum number of frames that can be rendered simultaneously
pub const MAX_FRAMES_IN_FLIGHT: u32 = 2;

const WINDOW_TITLE: &str = "Vulkan Tutorial";
const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;
const APPLICATION_VERSION: u32 = vk_make_version!(1, 0, 0);
const API_VERSION: u32 = vk_make_version!(1, 0, 0);
const ENGINE_VERSION: u32 = vk_make_version!(1, 0, 0);
const ENGINE_NAME: &str = "Sepia Engine";

fn main() {
    env_logger::init();

    let vertices: [Vertex; 3] = [
        Vertex::new([0.0, -0.5], [1.0, 0.0, 0.0]),
        Vertex::new([0.5, 0.5], [0.0, 1.0, 0.0]),
        Vertex::new([-0.5, 0.5], [0.0, 0.0, 1.0]),
    ];

    // Load the Vulkan library
    let entry = ash::Entry::new().expect("Failed to create entry");

    // Create the application info
    let app_name = CString::new(WINDOW_TITLE).expect("Failed to create CString");
    let engine_name = CString::new(ENGINE_NAME).expect("Failed to create CString");
    let app_info = vk::ApplicationInfo::builder()
        .application_name(&app_name)
        .engine_name(&engine_name)
        .api_version(API_VERSION)
        .application_version(APPLICATION_VERSION)
        .engine_version(ENGINE_VERSION)
        .build();

    // Determine required extension names
    let mut instance_extension_names = surface::surface_extension_names();

    // If validation layers are enabled
    // add the Debug Utils extension name to the list of required extension names
    if debug::ENABLE_VALIDATION_LAYERS {
        instance_extension_names.push(DebugUtils::name().as_ptr());
    }
    // Create the instance creation info
    let mut instance_create_info = vk::InstanceCreateInfo::builder()
        .application_info(&app_info)
        .enabled_extension_names(&instance_extension_names);

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

    if debug::ENABLE_VALIDATION_LAYERS {
        // Check if the required validation layers are supported
        for required in debug::REQUIRED_LAYERS.iter() {
            let found = entry
                .enumerate_instance_layer_properties()
                .expect("Couldn't enumerate instance layer properties")
                .iter()
                .any(|layer| {
                    let name = unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) };
                    let name = name.to_str().expect("Failed to get layer name pointer");
                    required == &name
                });

            if !found {
                panic!("Validation layer not supported: {}", required);
            }
        }

        instance_create_info = instance_create_info.enabled_layer_names(&layer_name_ptrs);
    }

    // Create a Vulkan instance
    let instance = unsafe {
        entry
            .create_instance(&instance_create_info, None)
            .expect("Failed to create instance")
    };

    // Initialize the window
    let mut event_loop = EventsLoop::new();
    let window = winit::WindowBuilder::new()
        .with_title(WINDOW_TITLE)
        .with_dimensions((WINDOW_WIDTH, WINDOW_HEIGHT).into())
        .build(&event_loop)
        .expect("Failed to create window.");

    // Create the window surface
    let surface = Surface::new(&entry, &instance);
    let surface_khr = unsafe {
        surface::create_surface(&entry, &instance, &window)
            .expect("Failed to create window surface!")
    };

    // Setup the debug messenger
    let debug_messenger = debug::setup_debug_messenger(&entry, &instance);

    // Pick a physical device
    let devices = unsafe {
        instance
            .enumerate_physical_devices()
            .expect("Couldn't get physical devices")
    };

    // Pick the first suitable physical device
    let physical_device = devices
        .into_iter()
        .find(|device| {
            // Check if device is suitable
            let mut graphics = None;
            let mut present = None;
            let properties =
                unsafe { instance.get_physical_device_queue_family_properties(*device) };
            for (index, family) in properties.iter().filter(|f| f.queue_count > 0).enumerate() {
                let index = index as u32;
                if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) && graphics.is_none() {
                    graphics = Some(index);
                }

                let present_support = unsafe {
                    surface.get_physical_device_surface_support(*device, index, surface_khr)
                };

                if present_support && present.is_none() {
                    present = Some(index);
                }

                if graphics.is_some() && present.is_some() {
                    break;
                }
            }

            // Get the supported surface formats
            let formats = unsafe {
                surface
                    .get_physical_device_surface_formats(*device, surface_khr)
                    .expect("Failed to get physical device surface formats")
            };

            // Get the supported present modes
            let present_modes = unsafe {
                surface
                    .get_physical_device_surface_present_modes(*device, surface_khr)
                    .expect("Failed to get physical device surface present modes")
            };

            let queue_families_supported = graphics.is_some() && present.is_some();
            let swapchain_adequate = !formats.is_empty() && !present_modes.is_empty();
            queue_families_supported && swapchain_adequate
        })
        .expect("Failed to create logical device");

    // Log the name of the physical device that was selected
    let props = unsafe { instance.get_physical_device_properties(physical_device) };
    log::debug!("Selected physical device: {:?}", unsafe {
        CStr::from_ptr(props.device_name.as_ptr())
    });

    // Get the memory properties of the physical device
    let memory_properties =
        unsafe { instance.get_physical_device_memory_properties(physical_device) };

    // Find the index of the graphics and present queue families
    let (graphics_family_index, present_family_index) = {
        let mut graphics = None;
        let mut present = None;
        let properties =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        for (index, family) in properties.iter().filter(|f| f.queue_count > 0).enumerate() {
            let index = index as u32;
            if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) && graphics.is_none() {
                graphics = Some(index);
            }

            let present_support = unsafe {
                surface.get_physical_device_surface_support(physical_device, index, surface_khr)
            };

            if present_support && present.is_none() {
                present = Some(index);
            }

            if graphics.is_some() && present.is_some() {
                break;
            }
        }
        (graphics, present)
    };

    // Need to dedup since the graphics family and presentation family
    // can have the same queue family index and
    // Vulkan does not allow passing an array containing duplicated family
    // indices.
    let mut queue_family_indices = vec![
        graphics_family_index.expect("Failed to find a graphics queue"),
        present_family_index.expect("Failed to find a present queue"),
    ];
    queue_family_indices.dedup();

    // Create the device queue creation info
    let queue_priorities = [1.0f32];

    // Build an array of DeviceQueueCreateInfo,
    // one for each different family index
    let queue_create_infos = queue_family_indices
        .iter()
        .map(|index| {
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(*index)
                .queue_priorities(&queue_priorities)
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

    if debug::ENABLE_VALIDATION_LAYERS {
        // Add the validation layers to the list of enabled layers if validation layers are enabled
        device_create_info_builder =
            device_create_info_builder.enabled_layer_names(&layer_name_ptrs)
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

    // Get the surface capabilities
    let capabilities = unsafe {
        surface
            .get_physical_device_surface_capabilities(physical_device, surface_khr)
            .expect("Failed to get physical device surface capabilities")
    };

    // Get the supported surface formats
    let formats = unsafe {
        surface
            .get_physical_device_surface_formats(physical_device, surface_khr)
            .expect("Failed to get physical device surface formats")
    };

    // Get the supported present modes
    let present_modes = unsafe {
        surface
            .get_physical_device_surface_present_modes(physical_device, surface_khr)
            .expect("Failed to get physical device surface present modes")
    };

    // Specify a default format and color space
    let (default_format, default_color_space) = (
        vk::Format::B8G8R8A8_UNORM,
        vk::ColorSpaceKHR::SRGB_NONLINEAR,
    );

    // Choose the default format if available or choose the first available format
    let surface_format = if formats.len() == 1 && formats[0].format == vk::Format::UNDEFINED {
        // If only one format is available
        // but it is undefined, assign a default
        vk::SurfaceFormatKHR {
            format: default_format,
            color_space: default_color_space,
        }
    } else {
        *formats
            .iter()
            .find(|format| {
                format.format == default_format
                    && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or_else(|| formats.first().expect("Failed to get first surface format"))
    };

    // Choose the swapchain surface present mode
    let present_mode = if present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
        vk::PresentModeKHR::MAILBOX
    } else if present_modes.contains(&vk::PresentModeKHR::FIFO) {
        vk::PresentModeKHR::FIFO
    } else {
        vk::PresentModeKHR::IMMEDIATE
    };

    // Choose the swapchain extent
    let extent = if capabilities.current_extent.width != std::u32::MAX {
        capabilities.current_extent
    } else {
        let min = capabilities.min_image_extent;
        let max = capabilities.max_image_extent;
        let width = WINDOW_WIDTH.min(max.width).max(min.width);
        let height = WINDOW_HEIGHT.min(max.height).max(min.height);
        vk::Extent2D { width, height }
    };

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
    let swapchain = Swapchain::new(&instance, &logical_device);
    let swapchain_khr = unsafe {
        swapchain
            .create_swapchain(&swapchain_create_info, None)
            .unwrap()
    };

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

    // Define the entry point for shaders
    let entry_point_name = CString::new("main").unwrap();

    // Create the vertex shader module
    let mut vertex_shader_file = std::fs::File::open("shaders/shader.vert.spv").unwrap();
    let vertex_shader_source = ash::util::read_spv(&mut vertex_shader_file).unwrap();
    let vertex_shader_create_info = vk::ShaderModuleCreateInfo::builder()
        .code(&vertex_shader_source)
        .build();
    let vertex_shader_module = unsafe {
        logical_device
            .create_shader_module(&vertex_shader_create_info, None)
            .unwrap()
    };
    let vertex_shader_state_info = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vertex_shader_module)
        .name(&entry_point_name)
        .build();

    // Create the fragment shader module
    let mut fragment_shader_file = std::fs::File::open("shaders/shader.frag.spv").unwrap();
    let fragment_shader_source = ash::util::read_spv(&mut fragment_shader_file).unwrap();
    let fragment_shader_create_info = vk::ShaderModuleCreateInfo::builder()
        .code(&fragment_shader_source)
        .build();
    let fragment_shader_module = unsafe {
        logical_device
            .create_shader_module(&fragment_shader_create_info, None)
            .unwrap()
    };
    let fragment_shader_state_info = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::FRAGMENT)
        .module(fragment_shader_module)
        .name(&entry_point_name)
        .build();

    // Build vertex input creation info
    let vertex_input_create_info = vk::PipelineVertexInputStateCreateInfo::builder()
        .vertex_binding_descriptions(&[Vertex::get_binding_description()])
        .vertex_attribute_descriptions(&Vertex::get_attribute_descriptions())
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
        width: extent.width as _,
        height: extent.height as _,
        min_depth: 0.0,
        max_depth: 1.0,
    };

    // Create a stencil
    let scissor = vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent,
    };

    // Build the viewport info using the viewport and stencil
    let viewport_create_info = vk::PipelineViewportStateCreateInfo::builder()
        .viewports(&[viewport])
        .scissors(&[scissor])
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

    // Build the color blending info using the color blend attachment
    let color_blending_info = vk::PipelineColorBlendStateCreateInfo::builder()
        .logic_op_enable(false)
        .logic_op(vk::LogicOp::COPY)
        .attachments(&[color_blend_attachment])
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

    // Create a render pass
    let attachment_description = vk::AttachmentDescription::builder()
        .format(surface_format.format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .build();

    let attachment_reference = vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .build();

    let subpass_description = vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&[attachment_reference])
        .build();

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

    let render_pass_info = vk::RenderPassCreateInfo::builder()
        .attachments(&[attachment_description])
        .subpasses(&[subpass_description])
        .dependencies(&[subpass_dependency])
        .build();

    let render_pass = unsafe {
        logical_device
            .create_render_pass(&render_pass_info, None)
            .unwrap()
    };

    // Create the pipeline info
    let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
        .stages(&[vertex_shader_state_info, fragment_shader_state_info])
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

    // Create the pipeline
    let pipeline = unsafe {
        logical_device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
            .unwrap()[0]
    };

    // Delete shader modules
    unsafe {
        logical_device.destroy_shader_module(vertex_shader_module, None);
        logical_device.destroy_shader_module(fragment_shader_module, None);
    };

    // Create one framebuffer for each image in the swapchain
    let framebuffers = image_views
        .as_slice()
        .iter()
        .map(|view| [*view])
        .map(|attachments| {
            let framebuffer_info = vk::FramebufferCreateInfo::builder()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(extent.width)
                .height(extent.height)
                .layers(1)
                .build();
            unsafe {
                logical_device
                    .create_framebuffer(&framebuffer_info, None)
                    .unwrap()
            }
        })
        .collect::<Vec<_>>();

    // Build the command pool info
    let command_pool_info = vk::CommandPoolCreateInfo::builder()
        .queue_family_index(graphics_family_index.unwrap())
        .flags(vk::CommandPoolCreateFlags::empty())
        .build();

    // Create the command pool
    let command_pool = unsafe {
        logical_device
            .create_command_pool(&command_pool_info, None)
            .unwrap()
    };

    // Build the transient command pool info
    let transient_command_pool_info = vk::CommandPoolCreateInfo::builder()
        .queue_family_index(graphics_family_index.unwrap())
        .flags(vk::CommandPoolCreateFlags::TRANSIENT)
        .build();

    // Create the command pool
    let transient_command_pool = unsafe {
        logical_device
            .create_command_pool(&transient_command_pool_info, None)
            .unwrap()
    };

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

        // Copy the bytes of the staging buffer to the vertex buffer
        unsafe {
            logical_device.cmd_copy_buffer(command_buffer, staging_buffer, vertex_buffer, &[region])
        };

        // End command buffer recording
        unsafe { logical_device.end_command_buffer(command_buffer).unwrap() };

        // Build the submission info
        let submit_info = vk::SubmitInfo::builder()
            .command_buffers(&[command_buffer])
            .build();

        // Submit the command buffer
        unsafe {
            logical_device
                .queue_submit(graphics_queue, &[submit_info], vk::Fence::null())
                .unwrap()
        };

        // Wait for the command buffer to be executed
        unsafe { logical_device.queue_wait_idle(graphics_queue).unwrap() };

        // Free the command buffer
        unsafe { logical_device.free_command_buffers(command_pool, &[command_buffer]) };
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
                logical_device.cmd_bind_vertex_buffers(command_buffer, 0, &[vertex_buffer], &[0])
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

    // Run the main loop
    log::debug!("Running application.");
    let mut should_stop = false;
    loop {
        event_loop.poll_events(|event| match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            }
            | Event::WindowEvent {
                event:
                    WindowEvent::KeyboardInput {
                        input:
                            winit::KeyboardInput {
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    },
                ..
            } => should_stop = true,
            Event::WindowEvent {
                event:
                    WindowEvent::Resized(LogicalSize {
                        width: _width,
                        height: _height,
                    }),
                ..
            } => {
                // TODO: Handle resizing by recreating the swapchain
            }
            _ => {}
        });

        if should_stop {
            break;
        }

        let image_available_semaphore = image_available_semaphores[current_frame];

        let render_finished_semaphore = render_finished_semaphores[current_frame];

        let in_flight_fence = in_flight_fences[current_frame];

        unsafe {
            logical_device
                .wait_for_fences(&[in_flight_fence], true, std::u64::MAX)
                .unwrap();
            logical_device.reset_fences(&[in_flight_fence]).unwrap();
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

        // Submit the command buffer
        {
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let submit_info = vk::SubmitInfo::builder()
                .wait_semaphores(&[image_available_semaphore])
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&[command_buffers[image_index as usize]])
                .signal_semaphores(&[render_finished_semaphore])
                .build();

            unsafe {
                logical_device
                    .queue_submit(graphics_queue, &[submit_info], in_flight_fence)
                    .unwrap()
            };
        }

        // Present the rendered image
        {
            let present_info = vk::PresentInfoKHR::builder()
                .wait_semaphores(&[render_finished_semaphore])
                .swapchains(&[swapchain_khr])
                .image_indices(&[image_index])
                .build();

            unsafe {
                swapchain
                    .queue_present(present_queue, &present_info)
                    .unwrap()
            };
        }

        current_frame += (1 + current_frame) % MAX_FRAMES_IN_FLIGHT as usize;
    }

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
        surface.destroy_surface(surface_khr, None);
        if let Some((debug_utils, messenger)) = debug_messenger {
            debug_utils.destroy_debug_utils_messenger(messenger, None);
        }
        instance.destroy_instance(None);
    }
}
