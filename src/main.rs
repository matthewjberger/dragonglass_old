use ash::{
    extensions::{
        ext::DebugUtils,
        khr::{Surface, Swapchain},
    },
    version::{DeviceV1_0, EntryV1_0, InstanceV1_0},
    vk::{
        self, Bool32, DebugUtilsMessageSeverityFlagsEXT, DebugUtilsMessageTypeFlagsEXT,
        DebugUtilsMessengerCallbackDataEXT,
    },
    vk_make_version,
};
use std::{
    ffi::{CStr, CString},
    os::raw::c_void,
};
use winit::{Event, EventsLoop, VirtualKeyCode, WindowEvent};

// Enable validation layers only in debug mode

#[cfg(debug_assertions)]
pub const ENABLE_VALIDATION_LAYERS: bool = true;

#[cfg(not(debug_assertions))]
pub const ENABLE_VALIDATION_LAYERS: bool = false;

pub const REQUIRED_LAYERS: [&str; 1] = ["VK_LAYER_LUNARG_standard_validation"];

// Setup the callback for the debug utils extension
unsafe extern "system" fn vulkan_debug_callback(
    flags: DebugUtilsMessageSeverityFlagsEXT,
    type_flags: DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> Bool32 {
    let type_flag = if type_flags == DebugUtilsMessageTypeFlagsEXT::GENERAL {
        "General"
    } else if type_flags == DebugUtilsMessageTypeFlagsEXT::PERFORMANCE {
        "Performance"
    } else if type_flags == DebugUtilsMessageTypeFlagsEXT::VALIDATION {
        "Validation"
    } else {
        unreachable!()
    };

    let message = format!(
        "[{}] {:?}",
        type_flag,
        CStr::from_ptr((*p_callback_data).p_message)
    );

    if flags == DebugUtilsMessageSeverityFlagsEXT::ERROR {
        log::error!("{}", message);
    } else if flags == DebugUtilsMessageSeverityFlagsEXT::INFO {
        log::info!("{}", message);
    } else if flags == DebugUtilsMessageSeverityFlagsEXT::WARNING {
        log::warn!("{}", message);
    } else if flags == DebugUtilsMessageSeverityFlagsEXT::VERBOSE {
        log::trace!("{}", message);
    }
    vk::FALSE
}

// Determine required surface extensions based on platform

#[cfg(target_os = "windows")]
pub fn required_extension_names() -> Vec<*const i8> {
    use ash::extensions::khr::Win32Surface;
    vec![Surface::name().as_ptr(), Win32Surface::name().as_ptr()]
}

#[cfg(target_os = "linux")]
pub fn required_extension_names() -> Vec<*const i8> {
    use ash::extensions::khr::XlibSurface;
    vec![Surface::name().as_ptr(), XlibSurface::name().as_ptr()]
}

#[cfg(target_os = "windows")]
unsafe fn create_surface<E: EntryV1_0, I: InstanceV1_0>(
    entry: &E,
    instance: &I,
    window: &winit::Window,
) -> Result<vk::SurfaceKHR, vk::Result> {
    use ash::extensions::khr::Win32Surface;
    use std::ptr;
    use winapi::{shared::windef::HWND, um::libloaderapi::GetModuleHandleW};
    use winit::os::windows::WindowExt;

    let hwnd = window.get_hwnd() as HWND;
    let hinstance = GetModuleHandleW(ptr::null()) as *const c_void;
    let win32_create_info = vk::Win32SurfaceCreateInfoKHR {
        s_type: vk::StructureType::WIN32_SURFACE_CREATE_INFO_KHR,
        p_next: ptr::null(),
        flags: Default::default(),
        hinstance,
        hwnd: hwnd as *const c_void,
    };
    let win32_surface_loader = Win32Surface::new(entry, instance);
    win32_surface_loader.create_win32_surface(&win32_create_info, None)
}

#[cfg(target_os = "linux")]
unsafe fn create_surface<E: EntryV1_0, I: InstanceV1_0>(
    entry: &E,
    instance: &I,
    window: &winit::Window,
) -> Result<vk::SurfaceKHR, vk::Result> {
    use ash::extensions::khr::XlibSurface;
    use winit::os::unix::WindowExt;
    let x11_display = window.get_xlib_display().unwrap();
    let x11_window = window.get_xlib_window().unwrap();
    let x11_create_info = vk::XlibSurfaceCreateInfoKHR::builder()
        .window(x11_window)
        .dpy(x11_display as *mut vk::Display);

    let xlib_surface_loader = XlibSurface::new(entry, instance);
    xlib_surface_loader.create_xlib_surface(&x11_create_info, None)
}

const WINDOW_TITLE: &str = "Vulkan Tutorial";
const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;
const APPLICATION_VERSION: u32 = vk_make_version!(1, 0, 0);
const API_VERSION: u32 = vk_make_version!(1, 0, 0);
const ENGINE_VERSION: u32 = vk_make_version!(1, 0, 0);
const ENGINE_NAME: &str = "Sepia Engine";

fn main() {
    env_logger::init();

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
    let mut instance_extension_names = required_extension_names();

    // If validation layers are enabled
    // add the Debug Utils extension name to the list of required extension names
    if ENABLE_VALIDATION_LAYERS {
        instance_extension_names.push(DebugUtils::name().as_ptr());
    }
    // Create the instance creation info
    let mut instance_create_info = vk::InstanceCreateInfo::builder()
        .application_info(&app_info)
        .enabled_extension_names(&instance_extension_names);

    // Determine the required layer names
    let layer_names = REQUIRED_LAYERS
        .iter()
        .map(|name| CString::new(*name).expect("Failed to build CString"))
        .collect::<Vec<_>>();

    // Determine required layer name pointers
    let layer_name_ptrs = layer_names
        .iter()
        .map(|name| name.as_ptr())
        .collect::<Vec<_>>();

    if ENABLE_VALIDATION_LAYERS {
        // Check if the required validation layers are supported
        for required in REQUIRED_LAYERS.iter() {
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
        create_surface(&entry, &instance, &window).expect("Failed to create window surface!")
    };

    // Setup the debug messenger
    let mut debug_utils: Option<DebugUtils> = None;
    let messenger = if ENABLE_VALIDATION_LAYERS {
        debug_utils = Some(DebugUtils::new(&entry, &instance));
        let create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .flags(vk::DebugUtilsMessengerCreateFlagsEXT::all())
            .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
            .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
            .pfn_user_callback(Some(vulkan_debug_callback))
            .build();
        unsafe {
            Some(
                debug_utils
                    .as_mut()
                    .unwrap()
                    .create_debug_utils_messenger(&create_info, None)
                    .expect("Failed to create debug utils messenger"),
            )
        }
    } else {
        None
    };

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

    if ENABLE_VALIDATION_LAYERS {
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
        // Will need these when shader has vertices that aren't hard-coded
        //.vertex_binding_descriptions()
        //.vertex_attribute_descriptions()
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
            let buffer = *buffer;

            // Begin the command buffer
            {
                let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE)
                    .build();
                unsafe {
                    logical_device
                        .begin_command_buffer(buffer, &command_buffer_begin_info)
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
                        buffer,
                        &render_pass_begin_info,
                        vk::SubpassContents::INLINE,
                    )
                };
            }

            // Bind pipeline
            unsafe {
                logical_device.cmd_bind_pipeline(buffer, vk::PipelineBindPoint::GRAPHICS, pipeline)
            };

            // Draw
            unsafe {
                logical_device.cmd_draw(buffer, 3, 1, 0, 0);
            }

            // End render pass
            unsafe { logical_device.cmd_end_render_pass(buffer) };

            // End command buffer
            unsafe { logical_device.end_command_buffer(buffer).unwrap() };
        });

    // Create the image available semaphore
    let image_available_semaphore = {
        let semaphore_info = vk::SemaphoreCreateInfo::builder().build();
        unsafe {
            logical_device
                .create_semaphore(&semaphore_info, None)
                .unwrap()
        }
    };

    // Create the render finished semaphore
    let render_finished_semaphore = {
        let semaphore_info = vk::SemaphoreCreateInfo::builder().build();
        unsafe {
            logical_device
                .create_semaphore(&semaphore_info, None)
                .unwrap()
        }
    };

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
            _ => {}
        });

        if should_stop {
            break;
        }

        log::trace!("Drawing frame.");

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
                    .queue_submit(graphics_queue, &[submit_info], vk::Fence::null())
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
    }

    unsafe { logical_device.device_wait_idle().unwrap() };

    // Free objects
    log::debug!("Dropping application");
    unsafe {
        logical_device.destroy_semaphore(render_finished_semaphore, None);
        logical_device.destroy_semaphore(image_available_semaphore, None);
        logical_device.destroy_command_pool(command_pool, None);
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
        if let Some(messenger) = messenger {
            let debug_utils = debug_utils.take().unwrap();
            debug_utils.destroy_debug_utils_messenger(messenger, None);
        }
        instance.destroy_instance(None);
    }
}
