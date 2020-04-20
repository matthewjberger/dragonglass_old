use crate::{
    core::VulkanContext,
    pipelines::skybox::{SkyboxPipeline, VERTICES},
    render::{Framebuffer, GraphicsPipeline, RenderPass},
    resource::{
        texture::{Cubemap, ImageLayoutTransition, Texture, TextureDescription},
        CommandPool, DescriptorPool, DescriptorSetLayout, GeometryBuffer, ImageView,
        PipelineLayout, Shader,
    },
};
use ash::{version::DeviceV1_0, vk};
use nalgebra_glm as glm;
use std::{ffi::CString, sync::Arc};

#[allow(dead_code)]
struct PushBlockIrradiance {
    mvp: glm::Mat4,
    delta_phi: f32,
    delta_theta: f32,
}

pub struct IrradianceMap {
    pub cubemap: Cubemap,
}

impl IrradianceMap {
    pub fn new(
        context: Arc<VulkanContext>,
        command_pool: &CommandPool,
        cubemap: &Cubemap,
        cube: &GeometryBuffer,
    ) -> Self {
        let dimension = 64;
        let format = vk::Format::R32G32B32A32_SFLOAT;
        let output_cubemap = Cubemap::new(context.clone(), dimension, format);

        let render_pass = Self::create_render_pass(context.clone(), format);

        let offscreen = Offscreen::new(context.clone(), dimension, format);

        let attachments = [offscreen.view.view()];
        let create_info = vk::FramebufferCreateInfo::builder()
            .render_pass(render_pass.render_pass())
            .attachments(&attachments)
            .width(dimension)
            .height(dimension)
            .layers(1)
            .build();
        let framebuffer = Framebuffer::new(context.clone(), create_info);

        let transition = ImageLayoutTransition {
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            src_stage_mask: vk::PipelineStageFlags::ALL_COMMANDS,
            dst_stage_mask: vk::PipelineStageFlags::ALL_COMMANDS,
        };

        offscreen.texture.transition(
            &command_pool,
            &transition,
            output_cubemap.description.mip_levels,
        );

        let descriptor_set_layout = Self::create_descriptor_set_layout(context.clone());
        let descriptor_pool = Self::create_descriptor_pool(context.clone());
        let descriptor_set =
            descriptor_pool.allocate_descriptor_sets(descriptor_set_layout.layout(), 1)[0];

        Self::update_descriptor_set(context.clone(), descriptor_set, &cubemap);

        let pipeline_layout =
            Self::create_pipeline_layout(context.clone(), descriptor_set_layout.layout());

        let pipeline = Self::create_pipeline(
            context.clone(),
            pipeline_layout,
            &render_pass,
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

        let transition = ImageLayoutTransition {
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            src_stage_mask: vk::PipelineStageFlags::ALL_COMMANDS,
            dst_stage_mask: vk::PipelineStageFlags::ALL_COMMANDS,
        };
        output_cubemap.transition(&command_pool, &transition);

        let mut viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: dimension as _,
            height: dimension as _,
            min_depth: 0.0,
            max_depth: 1.0,
        };

        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent,
        };
        let scissors = [scissor];

        for mip_level in 0..output_cubemap.description.mip_levels {
            for (face, matrix) in matrices.iter().enumerate() {
                let current_dimension = dimension as f32 * 0.5_f32.powf(mip_level as f32);
                viewport.width = current_dimension;
                viewport.height = current_dimension;
                let viewports = [viewport];

                command_pool.execute_command_once(
                    context.graphics_queue(),
                    |command_buffer| unsafe {
                        device.cmd_set_viewport(command_buffer, 0, &viewports);
                        device.cmd_set_scissor(command_buffer, 0, &scissors);

                        // Render scene from cube face's pov
                        device.cmd_begin_render_pass(
                            command_buffer,
                            &render_pass_begin_info,
                            vk::SubpassContents::INLINE,
                        );

                        let push_block_irradiance = PushBlockIrradiance {
                            mvp: glm::perspective(90_f32.to_radians(), 1.0, 0.1, 512.0) * matrix,
                            delta_phi: 2_f32.to_radians(),
                            delta_theta: (0.5_f32 * std::f32::consts::PI) / 64_f32,
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

                let transition = ImageLayoutTransition {
                    old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    dst_access_mask: vk::AccessFlags::TRANSFER_READ,
                    src_stage_mask: vk::PipelineStageFlags::ALL_COMMANDS,
                    dst_stage_mask: vk::PipelineStageFlags::ALL_COMMANDS,
                };
                offscreen.texture.transition(&command_pool, &transition, 1);

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
                    .width(current_dimension as _)
                    .height(current_dimension as _)
                    .depth(1)
                    .build();

                let region = vk::ImageCopy::builder()
                    .src_subresource(src_subresource)
                    .dst_subresource(dst_subresource)
                    .extent(extent)
                    .build();
                let regions = [region];

                command_pool.copy_image_to_image(
                    offscreen.texture.image(),
                    output_cubemap.texture.image(),
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &regions,
                );

                let transition = ImageLayoutTransition {
                    old_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    src_access_mask: vk::AccessFlags::TRANSFER_READ,
                    dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    src_stage_mask: vk::PipelineStageFlags::ALL_COMMANDS,
                    dst_stage_mask: vk::PipelineStageFlags::ALL_COMMANDS,
                };

                offscreen.texture.transition(&command_pool, &transition, 1);
            }
        }

        let transition = ImageLayoutTransition {
            old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            dst_access_mask: vk::AccessFlags::HOST_WRITE | vk::AccessFlags::TRANSFER_WRITE,
            src_stage_mask: vk::PipelineStageFlags::ALL_COMMANDS,
            dst_stage_mask: vk::PipelineStageFlags::ALL_COMMANDS,
        };

        output_cubemap.transition(&command_pool, &transition);

        Self {
            cubemap: output_cubemap,
        }
    }

    fn create_render_pass(context: Arc<VulkanContext>, format: vk::Format) -> RenderPass {
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

        RenderPass::new(context, &create_info)
    }

    fn create_descriptor_set_layout(context: Arc<VulkanContext>) -> DescriptorSetLayout {
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

        DescriptorSetLayout::new(context, layout_create_info)
    }

    fn create_descriptor_pool(context: Arc<VulkanContext>) -> DescriptorPool {
        let pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
        };
        let pool_sizes = [pool_size];

        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes)
            .max_sets(2)
            .build();

        DescriptorPool::new(context, pool_info)
    }

    fn update_descriptor_set(
        context: Arc<VulkanContext>,
        descriptor_set: vk::DescriptorSet,
        cubemap: &Cubemap,
    ) {
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
    }

    fn create_pipeline_layout(
        context: Arc<VulkanContext>,
        descriptor_set_layout: vk::DescriptorSetLayout,
    ) -> PipelineLayout {
        let descriptor_set_layouts = [descriptor_set_layout];

        let push_constant_range = vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .size(std::mem::size_of::<PushBlockIrradiance>() as u32)
            .build();
        let push_constant_ranges = [push_constant_range];

        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&descriptor_set_layouts)
            .push_constant_ranges(&push_constant_ranges)
            .build();

        PipelineLayout::new(context, pipeline_layout_create_info)
    }

    fn create_pipeline(
        context: Arc<VulkanContext>,
        pipeline_layout: PipelineLayout,
        render_pass: &RenderPass,
        descriptor_set_layout: DescriptorSetLayout,
    ) -> GraphicsPipeline {
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

        let (vertex_shader, fragment_shader, _shader_entry_point_name) =
            Self::create_shaders(context.clone());
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

        GraphicsPipeline::new(
            context,
            pipeline_create_info,
            pipeline_layout,
            descriptor_set_layout,
        )
    }

    fn create_shaders(context: Arc<VulkanContext>) -> (Shader, Shader, CString) {
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
            context,
            "examples/assets/shaders/irradiancecube.frag.spv",
            vk::ShaderStageFlags::FRAGMENT,
            &shader_entry_point_name,
        )
        .expect("Failed to create fragment shader!");

        (vertex_shader, fragment_shader, shader_entry_point_name)
    }
}

pub struct Offscreen {
    pub texture: Texture,
    pub view: ImageView,
    pub description: TextureDescription,
}

impl Offscreen {
    pub fn new(context: Arc<VulkanContext>, dimension: u32, format: vk::Format) -> Self {
        let description = TextureDescription::empty(dimension, dimension, format);
        let texture = Self::create_texture(context.clone(), &description);
        let view = Self::create_view(context, &texture, &description);

        Self {
            texture,
            view,
            description,
        }
    }

    fn create_texture(context: Arc<VulkanContext>, description: &TextureDescription) -> Texture {
        let image_create_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width: description.width,
                height: description.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .format(description.format)
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

        Texture::new(context, &allocation_create_info, &image_create_info)
    }

    fn create_view(
        context: Arc<VulkanContext>,
        texture: &Texture,
        description: &TextureDescription,
    ) -> ImageView {
        let create_info = vk::ImageViewCreateInfo::builder()
            .image(texture.image())
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(description.format)
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
        ImageView::new(context, create_info)
    }
}
