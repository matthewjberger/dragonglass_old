use crate::{
    core::VulkanContext,
    model::{gltf::GltfAsset, pbr_asset::PbrAsset},
    render::{GraphicsPipeline, Renderer},
    resource::{DescriptorSetLayout, PipelineLayout, Shader},
};
use ash::{version::DeviceV1_0, vk};
use nalgebra_glm as glm;
use petgraph::{graph::NodeIndex, visit::Dfs};
use std::{ffi::CString, mem, slice, sync::Arc};
pub struct PushConstantBlockMaterial {
    base_color_factor: glm::Vec4,
    color_texture_set: i32,
}

pub struct GltfPipeline {
    pub pipeline: GraphicsPipeline,
    pub assets: Vec<GltfAsset>,
    pub pbr_asset: PbrAsset,
}

impl GltfPipeline {
    pub fn new(mut renderer: &mut Renderer, asset_names: &[String]) -> Self {
        let (vertex_shader, fragment_shader, _shader_entry_point_name) =
            Self::create_shaders(renderer.context.clone());
        let shader_state_info = [vertex_shader.state_info(), fragment_shader.state_info()];

        let descriptions = Self::create_vertex_input_descriptions();
        let attributes = Self::create_vertex_attributes();
        let vertex_input_create_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&descriptions)
            .vertex_attribute_descriptions(&attributes)
            .build();

        let input_assembly_create_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false)
            .build();

        // Build the rasterizer info
        let rasterizer_create_info = vk::PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false)
            .depth_bias_constant_factor(0.0)
            .depth_bias_clamp(0.0)
            .depth_bias_slope_factor(0.0)
            .build();

        let multisampling_create_info = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .min_sample_shading(1.0)
            // .sample_mask()
            .alpha_to_coverage_enable(false)
            .alpha_to_one_enable(false)
            .build();

        let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .min_depth_bounds(0.0)
            .max_depth_bounds(1.0)
            .stencil_test_enable(false)
            .front(Default::default())
            .back(Default::default())
            .build();

        let color_blend_attachments = Self::create_color_blend_attachments();
        let color_blending_info = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(&color_blend_attachments)
            .blend_constants([0.0, 0.0, 0.0, 0.0])
            .build();

        let descriptor_set_layout = PbrAsset::descriptor_set_layout(renderer.context.clone());
        let pipeline_layout =
            Self::create_pipeline_layout(renderer.context.clone(), &descriptor_set_layout);

        let mut viewport_create_info = vk::PipelineViewportStateCreateInfo::default();
        viewport_create_info.viewport_count = 1;
        viewport_create_info.scissor_count = 1;

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_create_info = vk::PipelineDynamicStateCreateInfo::builder()
            .flags(vk::PipelineDynamicStateCreateFlags::empty())
            .dynamic_states(&dynamic_states)
            .build();

        // Create the pipeline info
        let pipeline_create_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&shader_state_info)
            .vertex_input_state(&vertex_input_create_info)
            .input_assembly_state(&input_assembly_create_info)
            .rasterization_state(&rasterizer_create_info)
            .multisample_state(&multisampling_create_info)
            .depth_stencil_state(&depth_stencil_info)
            .color_blend_state(&color_blending_info)
            .viewport_state(&viewport_create_info)
            .dynamic_state(&dynamic_state_create_info)
            .layout(pipeline_layout.layout())
            .render_pass(renderer.vulkan_swapchain.render_pass.render_pass())
            .subpass(0)
            .build();

        let pipeline = GraphicsPipeline::new(
            renderer.context.clone(),
            pipeline_create_info,
            pipeline_layout,
            descriptor_set_layout,
        );

        let mut assets = Vec::new();
        for asset_name in asset_names.iter() {
            assets.push(GltfAsset::new(&renderer, asset_name));
        }

        let number_of_meshes = assets.iter().fold(0, |total_meshes, asset| {
            total_meshes + asset.number_of_meshes
        });

        let textures = assets
            .iter()
            .flat_map(|asset| &asset.textures)
            .collect::<Vec<_>>();

        let pbr_asset = PbrAsset::new(&renderer, number_of_meshes, &textures);

        let gltf_pipeline = Self {
            pipeline,
            assets,
            pbr_asset,
        };
        gltf_pipeline.create_gltf_render_passes(&mut renderer);
        gltf_pipeline
    }

    fn create_shaders(context: Arc<VulkanContext>) -> (Shader, Shader, CString) {
        let shader_entry_point_name =
            CString::new("main").expect("Failed to create CString for shader entry point name!");

        let vertex_shader = Shader::from_file(
            context.clone(),
            "examples/assets/shaders/shader.vert.spv",
            vk::ShaderStageFlags::VERTEX,
            &shader_entry_point_name,
        )
        .expect("Failed to create vertex shader!");

        let fragment_shader = Shader::from_file(
            context,
            "examples/assets/shaders/shader.frag.spv",
            vk::ShaderStageFlags::FRAGMENT,
            &shader_entry_point_name,
        )
        .expect("Failed to create fragment shader!");

        (vertex_shader, fragment_shader, shader_entry_point_name)
    }

    fn create_vertex_input_descriptions() -> [vk::VertexInputBindingDescription; 1] {
        let vertex_input_binding_description = vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride((8 * mem::size_of::<f32>()) as _)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build();
        [vertex_input_binding_description]
    }

    fn create_vertex_attributes() -> [vk::VertexInputAttributeDescription; 3] {
        let position_description = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0)
            .build();

        let normal_description = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset((3 * mem::size_of::<f32>()) as _)
            .build();

        let tex_coord_description = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(2)
            .format(vk::Format::R32G32_SFLOAT)
            .offset((6 * mem::size_of::<f32>()) as _)
            .build();

        [
            position_description,
            normal_description,
            tex_coord_description,
        ]
    }

    pub fn create_color_blend_attachments() -> [vk::PipelineColorBlendAttachmentState; 1] {
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
        [color_blend_attachment]
    }

    pub fn create_pipeline_layout(
        context: Arc<VulkanContext>,
        descriptor_set_layout: &DescriptorSetLayout,
    ) -> PipelineLayout {
        let descriptor_set_layouts = [descriptor_set_layout.layout()];

        let push_constant_range = vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::ALL_GRAPHICS)
            .size(mem::size_of::<PushConstantBlockMaterial>() as u32)
            .build();
        let push_constant_ranges = [push_constant_range];

        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&descriptor_set_layouts) // needed for uniforms in shaders
            .push_constant_ranges(&push_constant_ranges)
            .build();

        PipelineLayout::new(context, pipeline_layout_create_info)
    }

    pub fn create_gltf_render_passes(&self, renderer: &mut Renderer) {
        // Allocate one command buffer per swapchain image
        let number_of_framebuffers = renderer.vulkan_swapchain.framebuffers.len();
        renderer
            .command_pool
            .allocate_command_buffers(number_of_framebuffers as _);

        // Create a single render pass per swapchain image that will draw each mesh
        renderer
            .command_pool
            .command_buffers()
            .iter()
            .enumerate()
            .for_each(|(index, buffer)| {
                let command_buffer = buffer;
                let framebuffer = renderer.vulkan_swapchain.framebuffers[index].framebuffer();
                self.create_render_pass(
                    &renderer,
                    framebuffer,
                    *command_buffer,
                    |command_buffer| unsafe {
                        self.draw_asset(renderer, command_buffer);
                    },
                );
            });
    }

    unsafe fn draw_asset(&self, renderer: &Renderer, command_buffer: vk::CommandBuffer) {
        let offsets = [0];
        self.assets.iter().for_each(|asset| {
            let vertex_buffers = [asset.vertex_buffer.buffer()];
            renderer
                .context
                .logical_device()
                .logical_device()
                .cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets);

            renderer
                .context
                .logical_device()
                .logical_device()
                .cmd_bind_index_buffer(
                    command_buffer,
                    asset.index_buffer.buffer(),
                    0,
                    vk::IndexType::UINT32,
                );

            for scene in asset.scenes.iter() {
                for graph in scene.node_graphs.iter() {
                    let mut dfs = Dfs::new(&graph, NodeIndex::new(0));
                    while let Some(node_index) = dfs.next(&graph) {
                        if let Some(mesh) = graph[node_index].mesh.as_ref() {
                            renderer
                                .context
                                .logical_device()
                                .logical_device()
                                .cmd_bind_descriptor_sets(
                                    command_buffer,
                                    vk::PipelineBindPoint::GRAPHICS,
                                    self.pipeline.layout(),
                                    0,
                                    &[self.pbr_asset.descriptor_set],
                                    &[(mesh.mesh_id as u64 * self.pbr_asset.dynamic_alignment)
                                        as _],
                                );

                            for primitive in mesh.primitives.iter() {
                                let mut material = PushConstantBlockMaterial {
                                    base_color_factor: glm::vec4(0.0, 0.0, 0.0, 1.0),
                                    color_texture_set: -1,
                                };

                                if let Some(material_index) = primitive.material_index {
                                    let primitive_material = asset
                                        .gltf
                                        .materials()
                                        .nth(material_index)
                                        .expect("Failed to retrieve material!");
                                    let pbr = primitive_material.pbr_metallic_roughness();

                                    if let Some(base_color_texture) = pbr.base_color_texture() {
                                        material.color_texture_set =
                                            base_color_texture.texture().index() as i32;
                                    } else {
                                        material.base_color_factor =
                                            glm::Vec4::from(pbr.base_color_factor());
                                    }
                                } else {
                                    material.base_color_factor = glm::vec4(0.0, 0.0, 0.0, 1.0);
                                }

                                renderer
                                    .context
                                    .logical_device()
                                    .logical_device()
                                    .cmd_push_constants(
                                        command_buffer,
                                        self.pipeline.layout(),
                                        vk::ShaderStageFlags::ALL_GRAPHICS,
                                        0,
                                        Self::byte_slice_from(&material),
                                    );

                                renderer
                                    .context
                                    .logical_device()
                                    .logical_device()
                                    .cmd_draw_indexed(
                                        command_buffer,
                                        primitive.number_of_indices,
                                        1,
                                        primitive.first_index,
                                        0,
                                        0,
                                    );
                            }
                        }
                    }
                }
            }
        });
    }

    // TODO: Move this to a seperate class or even the mod.rs file
    unsafe fn byte_slice_from<T: Sized>(data: &T) -> &[u8] {
        let data_ptr = (data as *const T) as *const u8;
        slice::from_raw_parts(data_ptr, std::mem::size_of::<T>())
    }

    pub fn create_render_pass<F>(
        &self,
        renderer: &Renderer,
        framebuffer: vk::Framebuffer,
        command_buffer: vk::CommandBuffer,
        mut render_action: F,
    ) where
        F: FnMut(vk::CommandBuffer),
    {
        // TODO: Move render pass creation into here

        // Begin the command buffer
        let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE)
            .build();
        unsafe {
            renderer
                .context
                .logical_device()
                .logical_device()
                .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                .expect("Failed to begin command buffer for the render pass!")
        };

        // TODO: Pass in clear values
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
            .render_pass(renderer.vulkan_swapchain.render_pass.render_pass())
            .framebuffer(framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: renderer.vulkan_swapchain.swapchain.properties().extent,
            })
            .clear_values(&clear_values)
            .build();

        unsafe {
            renderer
                .context
                .logical_device()
                .logical_device()
                .cmd_begin_render_pass(
                    command_buffer,
                    &render_pass_begin_info,
                    vk::SubpassContents::INLINE,
                );

            renderer
                .context
                .logical_device()
                .logical_device()
                .cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline.pipeline(),
                );
        }

        Self::update_viewport(command_buffer, &renderer);

        render_action(command_buffer);

        unsafe {
            // End render pass
            renderer
                .context
                .logical_device()
                .logical_device()
                .cmd_end_render_pass(command_buffer);

            // End command buffer
            renderer
                .context
                .logical_device()
                .logical_device()
                .end_command_buffer(command_buffer)
                .expect("Failed to end the command buffer for a render pass!");
        }
    }

    fn update_viewport(command_buffer: vk::CommandBuffer, renderer: &Renderer) {
        let extent = renderer.vulkan_swapchain.swapchain.properties().extent;

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
            renderer
                .context
                .logical_device()
                .logical_device()
                .cmd_set_viewport(command_buffer, 0, &viewports);

            renderer
                .context
                .logical_device()
                .logical_device()
                .cmd_set_scissor(command_buffer, 0, &scissors);
        }
    }
}
