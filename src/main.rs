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
use winit::{ControlFlow, Event, EventsLoop, VirtualKeyCode, WindowEvent};

// Enable validation layers only in debug mode

#[cfg(debug_assertions)]
pub const ENABLE_VALIDATION_LAYERS: bool = true;

#[cfg(not(debug_assertions))]
pub const ENABLE_VALIDATION_LAYERS: bool = false;

pub const REQUIRED_LAYERS: [&str; 1] = ["VK_LAYER_LUNARG_standard_validation"];

// Setup the callback for the debug utils extension
unsafe extern "system" fn vulkan_debug_callback(
    _: DebugUtilsMessageSeverityFlagsEXT,
    _: DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> Bool32 {
    log::debug!(
        "Validation layer: {:?}",
        CStr::from_ptr((*p_callback_data).p_message)
    );
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
            .unwrap_or(formats.first().expect("Failed to get first surface format"))
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

    // Delete shader modules
    unsafe {
        logical_device.destroy_shader_module(vertex_shader_module, None);
        logical_device.destroy_shader_module(fragment_shader_module, None);
    };

    // Run the main loop
    // event_loop.run_forever(|event| match event {
    //     Event::WindowEvent { event, .. } => match event {
    //         WindowEvent::KeyboardInput { input, .. } => {
    //             if let Some(VirtualKeyCode::Escape) = input.virtual_keycode {
    //                 ControlFlow::Break
    //             } else {
    //                 ControlFlow::Continue
    //             }
    //         }
    //         WindowEvent::CloseRequested => ControlFlow::Break,
    //         _ => ControlFlow::Continue,
    //     },
    //     _ => ControlFlow::Continue,
    // });

    // Free objects
    log::debug!("Dropping application");
    unsafe {
        image_views
            .iter()
            .for_each(|v| logical_device.destroy_image_view(*v, None));
        logical_device.destroy_device(None);
        surface.destroy_surface(surface_khr, None);
        if let Some(messenger) = messenger {
            let debug_utils = debug_utils.take().unwrap();
            debug_utils.destroy_debug_utils_messenger(messenger, None);
        }
        instance.destroy_instance(None);
    }
}
