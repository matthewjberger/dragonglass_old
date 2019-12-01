use crate::context::VulkanContext;
use crate::shader;
use crate::vertex::Vertex;
use ash::{
    extensions::khr::{Surface, Swapchain},
    version::DeviceV1_0,
    vk,
};
use nalgebra_glm as glm;
use std::{ffi::CString, sync::Arc};

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

#[derive(Clone, Copy, Debug)]
pub struct SwapchainProperties {
    pub format: vk::SurfaceFormatKHR,
    pub present_mode: vk::PresentModeKHR,
    pub extent: vk::Extent2D,
}

pub struct SwapchainSupportDetails {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupportDetails {
    pub fn new(
        physical_device: vk::PhysicalDevice,
        surface: &Surface,
        surface_khr: vk::SurfaceKHR,
    ) -> Self {
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

        Self {
            capabilities,
            formats,
            present_modes,
        }
    }

    pub fn suitable_properties(&self, preferred_dimensions: [u32; 2]) -> SwapchainProperties {
        let format = Self::choose_surface_format(&self.formats);
        let present_mode = Self::choose_surface_present_mode(&self.present_modes);
        let extent = Self::choose_swapchain_extent(self.capabilities, preferred_dimensions);
        SwapchainProperties {
            format,
            present_mode,
            extent,
        }
    }

    fn choose_surface_format(available_formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
        // Specify a default format and color space
        let (default_format, default_color_space) = (
            vk::Format::B8G8R8A8_UNORM,
            vk::ColorSpaceKHR::SRGB_NONLINEAR,
        );

        // Choose the default format if available or choose the first available format
        if available_formats.len() == 1 && available_formats[0].format == vk::Format::UNDEFINED {
            // If only one format is available
            // but it is undefined, assign a default
            vk::SurfaceFormatKHR {
                format: default_format,
                color_space: default_color_space,
            }
        } else {
            *available_formats
                .iter()
                .find(|format| {
                    format.format == default_format && format.color_space == default_color_space
                })
                .unwrap_or_else(|| {
                    available_formats
                        .first()
                        .expect("Failed to get first surface format")
                })
        }
    }

    fn choose_surface_present_mode(
        available_present_modes: &[vk::PresentModeKHR],
    ) -> vk::PresentModeKHR {
        if available_present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else if available_present_modes.contains(&vk::PresentModeKHR::FIFO) {
            vk::PresentModeKHR::FIFO
        } else {
            vk::PresentModeKHR::IMMEDIATE
        }
    }

    fn choose_swapchain_extent(
        capabilities: vk::SurfaceCapabilitiesKHR,
        preferred_dimensions: [u32; 2],
    ) -> vk::Extent2D {
        if capabilities.current_extent.width != std::u32::MAX {
            capabilities.current_extent
        } else {
            let min = capabilities.min_image_extent;
            let max = capabilities.max_image_extent;
            let width = preferred_dimensions[0].min(max.width).max(min.width);
            let height = preferred_dimensions[1].min(max.height).max(min.height);
            vk::Extent2D { width, height }
        }
    }
}

pub struct VulkanSwapchain {
    pub swapchain: Swapchain,
    pub swapchain_khr: vk::SwapchainKHR,
    pub swapchain_properties: SwapchainProperties,
    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
    pub render_pass: vk::RenderPass,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub framebuffers: Vec<vk::Framebuffer>,
    context: Arc<VulkanContext>,
}

impl VulkanSwapchain {
    pub fn new(context: Arc<VulkanContext>) -> Self {
        let (swapchain, swapchain_khr, swapchain_properties, images) = create_swapchain(
            context.instance(),
            context.physical_device(),
            context.logical_device(),
            context.surface(),
            context.surface_khr(),
            context.graphics_queue_family_index(),
            context.present_queue_family_index(),
        );
        let image_views =
            create_image_views(context.logical_device(), &swapchain_properties, &images);
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

        let number_of_images = images.len();
        let descriptor_pool =
            create_descriptor_pool(context.logical_device(), number_of_images as _);

        VulkanSwapchain {
            swapchain,
            swapchain_khr,
            swapchain_properties,
            images,
            image_views,
            render_pass,
            descriptor_set_layout,
            descriptor_pool,
            pipeline,
            pipeline_layout,
            framebuffers,
            context,
        }
    }
}

impl Drop for VulkanSwapchain {
    fn drop(&mut self) {
        unsafe {
            self.framebuffers
                .iter()
                .for_each(|f| self.context.logical_device().destroy_framebuffer(*f, None));
            self.context
                .logical_device()
                .destroy_pipeline(self.pipeline, None);
            self.context
                .logical_device()
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.context
                .logical_device()
                .destroy_render_pass(self.render_pass, None);
            self.image_views
                .iter()
                .for_each(|v| self.context.logical_device().destroy_image_view(*v, None));
            self.swapchain.destroy_swapchain(self.swapchain_khr, None);
            self.context
                .logical_device()
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.context
                .logical_device()
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.context.logical_device().destroy_device(None);
        }
    }
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
        .cull_mode(vk::CullModeFlags::NONE)
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
