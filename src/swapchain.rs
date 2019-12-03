use crate::buffer::{create_buffer, create_device_local_buffer};
use crate::context::VulkanContext;
use crate::shader;
use crate::vertex::Vertex;
use ash::{
    extensions::khr::{Surface, Swapchain},
    version::DeviceV1_0,
    vk,
};
use nalgebra_glm as glm;
use std::{ffi::CString, mem, sync::Arc, time::Instant};

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
    context: Arc<VulkanContext>,
    pub command_buffers: Vec<vk::CommandBuffer>,
    pub command_pool: vk::CommandPool,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub framebuffers: Vec<vk::Framebuffer>,
    pub graphics_queue: vk::Queue,
    pub image_views: Vec<vk::ImageView>,
    pub images: Vec<vk::Image>,
    pub index_buffer: vk::Buffer,
    pub index_buffer_memory: vk::DeviceMemory,
    pub number_of_indices: u32,
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub present_queue: vk::Queue,
    pub render_pass: vk::RenderPass,
    pub swapchain: Swapchain,
    pub swapchain_khr: vk::SwapchainKHR,
    pub swapchain_properties: SwapchainProperties,
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

        let (swapchain, swapchain_khr, swapchain_properties, images) = create_swapchain(&context);
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
            images.len(),
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
            image_views,
            images,
            index_buffer,
            index_buffer_memory,
            number_of_indices: indices.len() as _,
            pipeline,
            pipeline_layout,
            present_queue,
            render_pass,
            swapchain,
            swapchain_khr,
            swapchain_properties,
            transient_command_pool,
            uniform_buffer_memory_list,
            uniform_buffers,
            vertex_buffer,
            vertex_buffer_memory,
        };

        vulkan_swapchain.command_buffers = vulkan_swapchain.create_command_buffers();
        vulkan_swapchain
    }

    pub fn recreate_swapchain(&mut self) {
        unsafe { self.context.logical_device().device_wait_idle().unwrap() };

        self.cleanup_swapchain();

        let (swapchain, swapchain_khr, swapchain_properties, images) =
            create_swapchain(&self.context);
        self.swapchain = swapchain;
        self.swapchain_khr = swapchain_khr;

        self.image_views = create_image_views(
            self.context.logical_device(),
            &swapchain_properties,
            &images,
        );

        self.render_pass = create_render_pass(self.context.logical_device(), &swapchain_properties);

        self.descriptor_set_layout = create_descriptor_set_layout(self.context.logical_device());

        let (pipeline, pipeline_layout) = create_pipeline(
            self.context.logical_device(),
            &swapchain_properties,
            self.render_pass,
            self.descriptor_set_layout,
        );
        self.pipeline = pipeline;
        self.pipeline_layout = pipeline_layout;

        self.framebuffers = create_framebuffers(
            self.context.logical_device(),
            self.image_views.as_slice(),
            &swapchain_properties,
            self.render_pass,
        );

        let number_of_images = images.len();
        self.descriptor_pool =
            create_descriptor_pool(self.context.logical_device(), number_of_images as _);

        self.create_command_buffers();
    }

    fn cleanup_swapchain(&self) {
        let logical_device = self.context.logical_device();
        unsafe {
            self.framebuffers
                .iter()
                .for_each(|f| logical_device.destroy_framebuffer(*f, None));

            logical_device.free_command_buffers(self.command_pool, &self.command_buffers);

            logical_device.destroy_pipeline(self.pipeline, None);
            logical_device.destroy_pipeline_layout(self.pipeline_layout, None);

            logical_device.destroy_render_pass(self.render_pass, None);

            self.image_views
                .iter()
                .for_each(|v| logical_device.destroy_image_view(*v, None));

            self.swapchain.destroy_swapchain(self.swapchain_khr, None);
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
                let framebuffer = self.framebuffers[index];
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
            .render_pass(self.render_pass)
            .framebuffer(framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.swapchain_properties.extent,
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
                self.pipeline,
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
            self.pipeline,
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
            self.pipeline_layout,
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
        swapchain_properties: SwapchainProperties,
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

fn create_swapchain(
    context: &VulkanContext,
) -> (
    Swapchain,
    vk::SwapchainKHR,
    SwapchainProperties,
    Vec<vk::Image>,
) {
    let swapchain_support_details = SwapchainSupportDetails::new(
        context.physical_device(),
        context.surface(),
        context.surface_khr(),
    );
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
            .surface(context.surface_khr())
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT);

        let mut queue_family_indices = vec![
            context.graphics_queue_family_index(),
            context.present_queue_family_index(),
        ];
        queue_family_indices.dedup();

        builder = if context.graphics_queue_family_index() != context.present_queue_family_index() {
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
    let swapchain = Swapchain::new(context.instance(), context.logical_device());
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
