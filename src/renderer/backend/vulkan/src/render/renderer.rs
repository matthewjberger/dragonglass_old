use crate::{
    core::VulkanContext,
    model::{gltf::GltfAsset, ModelBuffers},
    pipelines::{
        pbr::{PbrPipeline, PbrPipelineData, PbrRenderer},
        skybox::{SkyboxPipeline, SkyboxPipelineData, SkyboxRenderer, VERTICES},
    },
    render::{Framebuffer, GraphicsPipeline, RenderPass, VulkanSwapchain},
    resource::{
        texture::{Cubemap, CubemapFaces, Texture, TextureDescription},
        CommandPool, DescriptorPool, DescriptorSetLayout, ImageView, PipelineLayout, Sampler,
        Shader,
    },
    sync::SynchronizationSet,
};
use ash::{version::DeviceV1_0, vk};
use nalgebra_glm as glm;
use std::{ffi::CString, sync::Arc};

pub struct Renderer {
    pub context: Arc<VulkanContext>,
    pub vulkan_swapchain: VulkanSwapchain,
    pub synchronization_set: SynchronizationSet,
    pub current_frame: usize,
    pub command_pool: CommandPool,
    pub transient_command_pool: CommandPool,
    pub assets: Vec<GltfAsset>,
    pub pbr_pipeline: Option<PbrPipeline>,
    pub pbr_pipeline_data: Option<PbrPipelineData>,
    pub skybox_pipeline: Option<SkyboxPipeline>,
    pub skybox_pipeline_data: Option<SkyboxPipelineData>,
    pub cubemap: Option<Cubemap>,
}

impl Renderer {
    pub fn new(window: &winit::Window) -> Self {
        let context =
            Arc::new(VulkanContext::new(&window).expect("Failed to create VulkanContext"));

        let synchronization_set =
            SynchronizationSet::new(context.clone()).expect("Failed to create sync objects");

        let command_pool = CommandPool::new(context.clone(), vk::CommandPoolCreateFlags::empty());

        let transient_command_pool =
            CommandPool::new(context.clone(), vk::CommandPoolCreateFlags::TRANSIENT);

        let logical_size = window
            .get_inner_size()
            .expect("Failed to get the window's inner size!");
        let dimensions = [logical_size.width as u32, logical_size.height as u32];

        let vulkan_swapchain = VulkanSwapchain::new(context.clone(), dimensions, &command_pool);

        let mut renderer = Renderer {
            context,
            synchronization_set,
            current_frame: 0,
            vulkan_swapchain,
            command_pool,
            transient_command_pool,
            assets: Vec::new(),
            pbr_pipeline: None,
            pbr_pipeline_data: None,
            skybox_pipeline: None,
            skybox_pipeline_data: None,
            cubemap: None,
        };

        renderer.pbr_pipeline = Some(PbrPipeline::new(&mut renderer));
        renderer.skybox_pipeline = Some(SkyboxPipeline::new(&mut renderer));
        renderer
    }

    #[allow(dead_code)]
    pub fn recreate_swapchain(&mut self, _: Option<[u32; 2]>) {
        log::debug!("Recreating swapchain");
        self.context.logical_device().wait_idle();
        // TODO: Implement swapchain recreation
    }

    pub fn load_assets(&mut self, asset_names: &[String]) {
        Self::generate_brdf_lut(self.context.clone(), &self.transient_command_pool);

        let mut assets = Vec::new();
        for asset_name in asset_names.iter() {
            assets.push(GltfAsset::new(&self, asset_name));
        }

        let number_of_meshes = assets.iter().fold(0, |total_meshes, asset| {
            total_meshes + asset.number_of_meshes
        });

        let textures = assets
            .iter()
            .flat_map(|asset| &asset.textures)
            .collect::<Vec<_>>();

        let faces = CubemapFaces {
            left: "examples/assets/skyboxes/bluemountains/left.jpg".to_string(),
            right: "examples/assets/skyboxes/bluemountains/right.jpg".to_string(),
            top: "examples/assets/skyboxes/bluemountains/top.jpg".to_string(),
            bottom: "examples/assets/skyboxes/bluemountains/bottom.jpg".to_string(),
            front: "examples/assets/skyboxes/bluemountains/front.jpg".to_string(),
            back: "examples/assets/skyboxes/bluemountains/back.jpg".to_string(),
        };

        let cubemap = Cubemap::new(self.context.clone(), &self.transient_command_pool, &faces);

        self.pbr_pipeline_data = Some(PbrPipelineData::new(
            &self,
            number_of_meshes,
            &textures,
            &cubemap,
        ));
        let skybox_pipeline_data = SkyboxPipelineData::new(&self, &cubemap);

        Self::generate_irradiance_cubemap(
            self.context.clone(),
            &self.transient_command_pool,
            &cubemap,
            &skybox_pipeline_data.cube,
        );

        self.skybox_pipeline_data = Some(skybox_pipeline_data);
        self.assets = assets;
        self.cubemap = Some(cubemap);
    }

    pub fn allocate_command_buffers(&mut self) {
        // Allocate one command buffer per swapchain image
        let number_of_framebuffers = self.vulkan_swapchain.framebuffers.len();
        self.command_pool
            .allocate_command_buffers(number_of_framebuffers as _);
    }

    pub fn record_command_buffers(&self) {
        // Create a single render pass per swapchain image that will draw each mesh
        self.command_pool
            .command_buffers()
            .iter()
            .enumerate()
            .for_each(|(index, buffer)| {
                let command_buffer = *buffer;
                let framebuffer = self.vulkan_swapchain.framebuffers[index].framebuffer();
                self.draw(framebuffer, command_buffer);
            });
    }

    pub fn draw(&self, framebuffer: vk::Framebuffer, command_buffer: vk::CommandBuffer) {
        let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE)
            .build();
        unsafe {
            self.context
                .logical_device()
                .logical_device()
                .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                .expect("Failed to begin command buffer for the render pass!")
        };

        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.39, 0.58, 0.93, 1.0],
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
        ];

        let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.vulkan_swapchain.render_pass.render_pass())
            .framebuffer(framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.vulkan_swapchain.swapchain.properties().extent,
            })
            .clear_values(&clear_values)
            .build();

        unsafe {
            self.context
                .logical_device()
                .logical_device()
                .cmd_begin_render_pass(
                    command_buffer,
                    &render_pass_begin_info,
                    vk::SubpassContents::INLINE,
                );
        }

        self.render_skybox(command_buffer);
        self.render_assets(command_buffer);

        unsafe {
            self.context
                .logical_device()
                .logical_device()
                .cmd_end_render_pass(command_buffer);

            self.context
                .logical_device()
                .logical_device()
                .end_command_buffer(command_buffer)
                .expect("Failed to end the command buffer for a render pass!");
        }
    }

    pub fn render_assets(&self, command_buffer: vk::CommandBuffer) {
        let device = &self.context.logical_device().logical_device();

        let pbr_pipeline = self
            .pbr_pipeline
            .as_ref()
            .expect("Failed to get pbr pipeline!");

        pbr_pipeline.bind(device, command_buffer);

        let pbr_pipeline_data = self
            .pbr_pipeline_data
            .as_ref()
            .expect("Failed to get pbr pipeline data!");

        let pbr_renderer = PbrRenderer::new(command_buffer, &pbr_pipeline, &pbr_pipeline_data);

        self.update_viewport(command_buffer);

        self.assets
            .iter()
            .for_each(|asset| pbr_renderer.draw_asset(device, &asset));
    }

    pub fn render_skybox(&self, command_buffer: vk::CommandBuffer) {
        let device = &self.context.logical_device().logical_device();

        let skybox_pipeline = self
            .skybox_pipeline
            .as_ref()
            .expect("Failed to get skybox pipeline!");

        skybox_pipeline.bind(device, command_buffer);

        let skybox_pipeline_data = self
            .skybox_pipeline_data
            .as_ref()
            .expect("Failed to get skybox pipeline data!");

        let skybox_renderer =
            SkyboxRenderer::new(command_buffer, &skybox_pipeline, &skybox_pipeline_data);

        self.update_viewport(command_buffer);

        skybox_renderer.draw(device);
    }

    pub fn update_viewport(&self, command_buffer: vk::CommandBuffer) {
        let device = self.context.logical_device().logical_device();
        let extent = self.vulkan_swapchain.swapchain.properties().extent;

        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as _,
            height: extent.height as _,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let viewports = [viewport];

        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent,
        };
        let scissors = [scissor];

        unsafe {
            device.cmd_set_viewport(command_buffer, 0, &viewports);
            device.cmd_set_scissor(command_buffer, 0, &scissors);
        }
    }

    fn generate_brdf_lut(
        context: Arc<VulkanContext>,
        command_pool: &CommandPool,
    ) -> (Texture, ImageView, Sampler) {
        let dimension = 512;
        let format = vk::Format::R16G16_SFLOAT;
        let image_create_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width: dimension,
                height: dimension,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .format(format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(vk::SampleCountFlags::TYPE_1)
            .flags(vk::ImageCreateFlags::empty())
            .build();

        let allocation_create_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::GpuOnly,
            ..Default::default()
        };

        let texture = Texture::new(context.clone(), &allocation_create_info, &image_create_info);

        let create_info = vk::ImageViewCreateInfo::builder()
            .image(texture.image())
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
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
        let image_view = ImageView::new(context.clone(), create_info);

        let sampler_info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .anisotropy_enable(true)
            .max_anisotropy(1.0)
            .border_color(vk::BorderColor::INT_OPAQUE_WHITE)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(1.0)
            .build();
        let sampler = Sampler::new(context.clone(), sampler_info);

        // Create render pass

        let color_attachment_description = vk::AttachmentDescription::builder()
            .format(format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .build();

        let attachment_descriptions = [color_attachment_description];

        let color_attachment_reference = vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build();
        let color_attachment_references = [color_attachment_reference];

        let subpass_description = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachment_references)
            .build();
        let subpass_descriptions = [subpass_description];

        let subpass_dependency_one = vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::BOTTOM_OF_PIPE)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::MEMORY_READ)
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            )
            .dependency_flags(vk::DependencyFlags::BY_REGION)
            .build();
        let subpass_dependency_two = vk::SubpassDependency::builder()
            .src_subpass(0)
            .dst_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags::BOTTOM_OF_PIPE)
            .src_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            )
            .dst_access_mask(vk::AccessFlags::MEMORY_READ)
            .dependency_flags(vk::DependencyFlags::BY_REGION)
            .build();
        let subpass_dependencies = [subpass_dependency_one, subpass_dependency_two];

        let create_info = vk::RenderPassCreateInfo::builder()
            .attachments(&attachment_descriptions)
            .subpasses(&subpass_descriptions)
            .dependencies(&subpass_dependencies)
            .build();

        let render_pass = RenderPass::new(context.clone(), &create_info);

        let attachments = [image_view.view()];
        let create_info = vk::FramebufferCreateInfo::builder()
            .render_pass(render_pass.render_pass())
            .attachments(&attachments)
            .width(dimension)
            .height(dimension)
            .layers(1)
            .build();
        let framebuffer = Framebuffer::new(context.clone(), create_info);

        // Create an empty descriptor set layout
        let layout_create_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(&[])
            .build();
        let descriptor_set_layout = DescriptorSetLayout::new(context.clone(), layout_create_info);

        // Pipeline layout
        let descriptor_set_layouts = [descriptor_set_layout.layout()];
        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&descriptor_set_layouts)
            .build();
        let pipeline_layout = PipelineLayout::new(context.clone(), pipeline_layout_create_info);

        // Pipeline
        let input_assembly_create_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .build();

        let rasterizer_create_info = vk::PipelineRasterizationStateCreateInfo::builder()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0)
            .build();

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::all())
            .blend_enable(false)
            .build();
        let color_blend_attachments = [color_blend_attachment];

        let color_blend_state_info = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(&color_blend_attachments)
            .blend_constants([0.0, 0.0, 0.0, 0.0])
            .build();

        let back_stencil_op_state = vk::StencilOpState::builder()
            .compare_op(vk::CompareOp::LESS_OR_EQUAL)
            .build();
        let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(false)
            .depth_write_enable(false)
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
            .front(Default::default())
            .back(back_stencil_op_state)
            .build();

        let mut viewport_create_info = vk::PipelineViewportStateCreateInfo::default();
        viewport_create_info.viewport_count = 1;
        viewport_create_info.scissor_count = 1;

        let multisampling_create_info = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .build();

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_create_info = vk::PipelineDynamicStateCreateInfo::builder()
            .flags(vk::PipelineDynamicStateCreateFlags::empty())
            .dynamic_states(&dynamic_states)
            .build();

        let vertex_input_create_info = vk::PipelineVertexInputStateCreateInfo::builder().build();

        let shader_entry_point_name =
            CString::new("main").expect("Failed to create CString for shader entry point name!");

        let vertex_shader = Shader::from_file(
            context.clone(),
            "examples/assets/shaders/genbrdflut.vert.spv",
            vk::ShaderStageFlags::VERTEX,
            &shader_entry_point_name,
        )
        .expect("Failed to create vertex shader!");

        let fragment_shader = Shader::from_file(
            context.clone(),
            "examples/assets/shaders/genbrdflut.frag.spv",
            vk::ShaderStageFlags::FRAGMENT,
            &shader_entry_point_name,
        )
        .expect("Failed to create fragment shader!");

        let shader_state_info = [vertex_shader.state_info(), fragment_shader.state_info()];

        let pipeline_create_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&shader_state_info)
            .vertex_input_state(&vertex_input_create_info)
            .input_assembly_state(&input_assembly_create_info)
            .rasterization_state(&rasterizer_create_info)
            .multisample_state(&multisampling_create_info)
            .depth_stencil_state(&depth_stencil_info)
            .color_blend_state(&color_blend_state_info)
            .viewport_state(&viewport_create_info)
            .dynamic_state(&dynamic_state_create_info)
            .layout(pipeline_layout.layout())
            .render_pass(render_pass.render_pass())
            .subpass(0)
            .build();

        let pipeline = GraphicsPipeline::new(
            context.clone(),
            pipeline_create_info,
            pipeline_layout,
            descriptor_set_layout,
        );

        let clear_values = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        }];

        let extent = vk::Extent2D::builder()
            .width(dimension)
            .height(dimension)
            .build();

        let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(render_pass.render_pass())
            .framebuffer(framebuffer.framebuffer())
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            })
            .clear_values(&clear_values)
            .build();

        let device = context.logical_device().logical_device();

        command_pool.execute_command_once(context.graphics_queue(), |command_buffer| unsafe {
            device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_begin_info,
                vk::SubpassContents::INLINE,
            );

            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: dimension as _,
                height: dimension as _,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            let viewports = [viewport];

            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            };
            let scissors = [scissor];

            device.cmd_set_viewport(command_buffer, 0, &viewports);
            device.cmd_set_scissor(command_buffer, 0, &scissors);

            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.pipeline(),
            );
            device.cmd_draw(command_buffer, 3, 1, 0, 0);

            device.cmd_end_render_pass(command_buffer);
        });

        (texture, image_view, sampler)
    }

    fn generate_irradiance_cubemap(
        context: Arc<VulkanContext>,
        command_pool: &CommandPool,
        cubemap: &Cubemap,
        cube: &ModelBuffers,
    ) {
        let dimension = 64;
        let format = vk::Format::R32G32B32A32_SFLOAT;
        let mip_levels = TextureDescription::calculate_mip_levels(dimension, dimension);
        let image_create_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width: dimension,
                height: dimension,
                depth: 1,
            })
            .mip_levels(mip_levels)
            .array_layers(6)
            .format(format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(vk::SampleCountFlags::TYPE_1)
            .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE)
            .build();

        let allocation_create_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::GpuOnly,
            ..Default::default()
        };

        let texture = Texture::new(context.clone(), &allocation_create_info, &image_create_info);

        let create_info = vk::ImageViewCreateInfo::builder()
            .image(texture.image())
            .view_type(vk::ImageViewType::CUBE)
            .format(format)
            .components(vk::ComponentMapping {
                r: vk::ComponentSwizzle::IDENTITY,
                g: vk::ComponentSwizzle::IDENTITY,
                b: vk::ComponentSwizzle::IDENTITY,
                a: vk::ComponentSwizzle::IDENTITY,
            })
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 6,
            })
            .build();
        let image_view = ImageView::new(context.clone(), create_info);

        let sampler_info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .anisotropy_enable(true)
            .max_anisotropy(1.0)
            .border_color(vk::BorderColor::INT_OPAQUE_WHITE)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(mip_levels as _)
            .build();
        let sampler = Sampler::new(context.clone(), sampler_info);

        // Create render pass

        let color_attachment_description = vk::AttachmentDescription::builder()
            .format(format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build();
        let attachment_descriptions = [color_attachment_description];

        let color_attachment_reference = vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build();
        let color_attachment_references = [color_attachment_reference];

        let subpass_description = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachment_references)
            .build();
        let subpass_descriptions = [subpass_description];

        let subpass_dependency_one = vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::BOTTOM_OF_PIPE)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::MEMORY_READ)
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            )
            .dependency_flags(vk::DependencyFlags::BY_REGION)
            .build();
        let subpass_dependency_two = vk::SubpassDependency::builder()
            .src_subpass(0)
            .dst_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags::BOTTOM_OF_PIPE)
            .src_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            )
            .dst_access_mask(vk::AccessFlags::MEMORY_READ)
            .dependency_flags(vk::DependencyFlags::BY_REGION)
            .build();
        let subpass_dependencies = [subpass_dependency_one, subpass_dependency_two];

        let create_info = vk::RenderPassCreateInfo::builder()
            .attachments(&attachment_descriptions)
            .subpasses(&subpass_descriptions)
            .dependencies(&subpass_dependencies)
            .build();

        let render_pass = RenderPass::new(context.clone(), &create_info);

        // Create offscreen framebuffer
        let image_create_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width: dimension,
                height: dimension,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .format(format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(vk::SampleCountFlags::TYPE_1)
            .flags(vk::ImageCreateFlags::empty())
            .build();

        let allocation_create_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::GpuOnly,
            ..Default::default()
        };

        let offscreen_texture =
            Texture::new(context.clone(), &allocation_create_info, &image_create_info);

        let create_info = vk::ImageViewCreateInfo::builder()
            .image(offscreen_texture.image())
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
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
        let offscreen_image_view = ImageView::new(context.clone(), create_info);

        let attachments = [offscreen_image_view.view()];
        let create_info = vk::FramebufferCreateInfo::builder()
            .render_pass(render_pass.render_pass())
            .attachments(&attachments)
            .width(dimension)
            .height(dimension)
            .layers(1)
            .build();
        let framebuffer = Framebuffer::new(context.clone(), create_info);

        let barrier = vk::ImageMemoryBarrier::builder()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(offscreen_texture.image())
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .build();
        let barriers = [barrier];

        command_pool.transition_image_layout(
            &barriers,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS,
        );

        // Create descriptor set layout

        let binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build();
        let bindings = [binding];

        let layout_create_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(&bindings)
            .build();
        let descriptor_set_layout = DescriptorSetLayout::new(context.clone(), layout_create_info);

        // Create descriptor pool

        let pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
        };
        let pool_sizes = [pool_size];

        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes)
            .max_sets(2)
            .build();

        let descriptor_pool = DescriptorPool::new(context.clone(), pool_info);

        let descriptor_set =
            descriptor_pool.allocate_descriptor_sets(descriptor_set_layout.layout(), 1)[0];

        let image_info = vk::DescriptorImageInfo::builder()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(cubemap.view.view())
            .sampler(cubemap.sampler.sampler())
            .build();
        let image_infos = [image_info];

        let sampler_descriptor_write = vk::WriteDescriptorSet::builder()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image_infos)
            .build();

        let descriptor_writes = vec![sampler_descriptor_write];

        unsafe {
            context
                .logical_device()
                .logical_device()
                .update_descriptor_sets(&descriptor_writes, &[])
        }

        // Create pipeline

        // Pipeline layout
        let descriptor_set_layouts = [descriptor_set_layout.layout()];

        let push_constant_range = vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .size(std::mem::size_of::<PushBlockIrradiance>() as u32)
            .build();
        let push_constant_ranges = [push_constant_range];

        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&descriptor_set_layouts)
            .push_constant_ranges(&push_constant_ranges)
            .build();

        let pipeline_layout = PipelineLayout::new(context.clone(), pipeline_layout_create_info);

        // Pipeline
        let input_assembly_create_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .build();

        let rasterizer_create_info = vk::PipelineRasterizationStateCreateInfo::builder()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0)
            .build();

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::all())
            .blend_enable(false)
            .build();
        let color_blend_attachments = [color_blend_attachment];

        let color_blend_state_info = vk::PipelineColorBlendStateCreateInfo::builder()
            .attachments(&color_blend_attachments)
            .build();

        let back_stencil_op_state = vk::StencilOpState::builder()
            .compare_op(vk::CompareOp::ALWAYS)
            .build();
        let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(false)
            .depth_write_enable(false)
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
            .front(Default::default())
            .back(back_stencil_op_state)
            .build();

        let mut viewport_create_info = vk::PipelineViewportStateCreateInfo::default();
        viewport_create_info.viewport_count = 1;
        viewport_create_info.scissor_count = 1;

        let multisampling_create_info = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .build();

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_create_info = vk::PipelineDynamicStateCreateInfo::builder()
            .flags(vk::PipelineDynamicStateCreateFlags::empty())
            .dynamic_states(&dynamic_states)
            .build();

        let descriptions = SkyboxPipeline::create_vertex_input_descriptions();
        let attributes = SkyboxPipeline::create_vertex_attributes();
        let vertex_input_create_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&descriptions)
            .vertex_attribute_descriptions(&attributes)
            .build();

        let shader_entry_point_name =
            CString::new("main").expect("Failed to create CString for shader entry point name!");

        let vertex_shader = Shader::from_file(
            context.clone(),
            "examples/assets/shaders/filtercube.vert.spv",
            vk::ShaderStageFlags::VERTEX,
            &shader_entry_point_name,
        )
        .expect("Failed to create vertex shader!");

        let fragment_shader = Shader::from_file(
            context.clone(),
            "examples/assets/shaders/irradiancecube.frag.spv",
            vk::ShaderStageFlags::FRAGMENT,
            &shader_entry_point_name,
        )
        .expect("Failed to create fragment shader!");

        let shader_state_info = [vertex_shader.state_info(), fragment_shader.state_info()];

        let pipeline_create_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&shader_state_info)
            .vertex_input_state(&vertex_input_create_info)
            .input_assembly_state(&input_assembly_create_info)
            .rasterization_state(&rasterizer_create_info)
            .multisample_state(&multisampling_create_info)
            .depth_stencil_state(&depth_stencil_info)
            .color_blend_state(&color_blend_state_info)
            .viewport_state(&viewport_create_info)
            .dynamic_state(&dynamic_state_create_info)
            .layout(pipeline_layout.layout())
            .render_pass(render_pass.render_pass())
            .subpass(0)
            .build();

        let pipeline = GraphicsPipeline::new(
            context.clone(),
            pipeline_create_info,
            pipeline_layout,
            descriptor_set_layout,
        );

        let clear_values = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.2, 0.0],
            },
        }];

        let extent = vk::Extent2D::builder()
            .width(dimension)
            .height(dimension)
            .build();

        let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(render_pass.render_pass())
            .framebuffer(framebuffer.framebuffer())
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            })
            .clear_values(&clear_values)
            .build();

        let device = context.logical_device().logical_device();

        let matrices = vec![
            glm::rotate(
                &glm::rotate(
                    &glm::Mat4::identity(),
                    90_f32.to_radians(),
                    &glm::vec3(0.0, 1.0, 0.0),
                ),
                180_f32.to_radians(),
                &glm::vec3(1.0, 0.0, 0.0),
            ),
            glm::rotate(
                &glm::rotate(
                    &glm::Mat4::identity(),
                    (-90_f32).to_radians(),
                    &glm::vec3(0.0, 1.0, 0.0),
                ),
                180_f32.to_radians(),
                &glm::vec3(1.0, 0.0, 0.0),
            ),
            glm::rotate(
                &glm::Mat4::identity(),
                (-90_f32).to_radians(),
                &glm::vec3(1.0, 0.0, 0.0),
            ),
            glm::rotate(
                &glm::Mat4::identity(),
                90_f32.to_radians(),
                &glm::vec3(1.0, 0.0, 0.0),
            ),
            glm::rotate(
                &glm::Mat4::identity(),
                180_f32.to_radians(),
                &glm::vec3(1.0, 0.0, 0.0),
            ),
            glm::rotate(
                &glm::Mat4::identity(),
                180_f32.to_radians(),
                &glm::vec3(0.0, 0.0, 1.0),
            ),
        ];

        let barrier = vk::ImageMemoryBarrier::builder()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(texture.image())
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 6,
            })
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .build();
        let barriers = [barrier];

        command_pool.transition_image_layout(
            &barriers,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS,
        );

        let mut viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: dimension as _,
            height: dimension as _,
            min_depth: 0.0,
            max_depth: 1.0,
        };

        for mip_level in 0..mip_levels {
            for face in 0..6 {
                command_pool.execute_command_once(
                    context.graphics_queue(),
                    |command_buffer| unsafe {
                        let current_dimension = dimension as f32 * (mip_level as f32).sqrt();
                        viewport.width = current_dimension;
                        viewport.height = current_dimension;
                        let viewports = [viewport];

                        let scissor = vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent,
                        };
                        let scissors = [scissor];

                        device.cmd_set_viewport(command_buffer, 0, &viewports);
                        device.cmd_set_scissor(command_buffer, 0, &scissors);

                        // Render scene from cube face's pov
                        device.cmd_begin_render_pass(
                            command_buffer,
                            &render_pass_begin_info,
                            vk::SubpassContents::INLINE,
                        );

                        let push_block_irradiance = PushBlockIrradiance {
                            mvp: glm::perspective(std::f32::consts::PI / 2.0, 1.0, 0.1, 512.0)
                                * matrices[face],
                            ..Default::default()
                        };

                        device.cmd_push_constants(
                            command_buffer,
                            pipeline.layout(),
                            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                            0,
                            dragonglass_core::byte_slice_from(&push_block_irradiance),
                        );

                        device.cmd_bind_pipeline(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            pipeline.pipeline(),
                        );

                        let offsets = [0];
                        let vertex_buffers = [cube.vertex_buffer.buffer()];

                        device.cmd_bind_vertex_buffers(
                            command_buffer,
                            0,
                            &vertex_buffers,
                            &offsets,
                        );

                        device.cmd_bind_descriptor_sets(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            pipeline.layout(),
                            0,
                            &[descriptor_set],
                            &[],
                        );

                        device.cmd_draw(command_buffer, VERTICES.len() as _, 1, 0, 0);

                        device.cmd_end_render_pass(command_buffer);
                    },
                );

                let barrier = vk::ImageMemoryBarrier::builder()
                    .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(offscreen_texture.image())
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: mip_levels,
                        base_array_layer: 0,
                        layer_count: 6,
                    })
                    .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .build();
                let barriers = [barrier];

                command_pool.transition_image_layout(
                    &barriers,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                );

                let src_subresource = vk::ImageSubresourceLayers::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_array_layer(0)
                    .mip_level(0)
                    .layer_count(1)
                    .build();

                let dst_subresource = vk::ImageSubresourceLayers::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_array_layer(face as _)
                    .mip_level(mip_level)
                    .layer_count(1)
                    .build();

                let extent = vk::Extent3D::builder()
                    .width(viewport.width as _)
                    .height(viewport.height as _)
                    .depth(1)
                    .build();

                let region = vk::ImageCopy::builder()
                    .src_subresource(src_subresource)
                    .dst_subresource(dst_subresource)
                    .extent(extent)
                    .build();
                let regions = [region];

                command_pool.copy_image_to_image(
                    offscreen_texture.image(),
                    texture.image(),
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &regions,
                );

                let barrier = vk::ImageMemoryBarrier::builder()
                    .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(offscreen_texture.image())
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .build();
                let barriers = [barrier];

                command_pool.transition_image_layout(
                    &barriers,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                );
            }

            let barrier = vk::ImageMemoryBarrier::builder()
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(texture.image())
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: mip_levels,
                    base_array_layer: 0,
                    layer_count: 6,
                })
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_WRITE | vk::AccessFlags::TRANSFER_WRITE)
                .build();
            let barriers = [barrier];

            command_pool.transition_image_layout(
                &barriers,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
            );
        }
    }
}

struct PushBlockIrradiance {
    mvp: glm::Mat4,
    delta_phi: f32,
    delta_theta: f32,
}

impl Default for PushBlockIrradiance {
    fn default() -> Self {
        Self {
            mvp: glm::Mat4::identity(),
            delta_phi: 2_f32.to_radians(),
            delta_theta: (0.5_f32 * std::f32::consts::PI) / 64_f32,
        }
    }
}
