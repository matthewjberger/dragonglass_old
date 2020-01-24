use crate::{
    core::ImageView,
    render::{GraphicsPipeline, Renderer, UniformBufferObject},
    resource::{
        Buffer, DescriptorPool, DescriptorSetLayout, Dimension, PipelineLayout, Sampler, Shader,
        Texture, TextureDescription,
    },
};
use ash::{version::DeviceV1_0, vk};
use gltf::image::Format;
use image::{ImageBuffer, Pixel, RgbImage};
use nalgebra_glm as glm;
use std::{ffi::CString, mem, slice};

pub struct PushConstantBlockMaterial {
    base_color_factor: glm::Vec4,
    color_texture_set: i32,
}

pub struct VulkanGltfAsset {
    pub gltf: gltf::Document,
    pub textures: Vec<VulkanTexture>,
    pub meshes: Vec<Vec<Mesh>>,
    pub descriptor_pool: DescriptorPool,
}

pub struct VulkanTexture {
    pub texture: Texture,
    pub view: ImageView,
    pub sampler: Sampler,
}

pub struct Mesh {
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub primitives: Vec<Primitive>,
    pub uniform_buffers: Vec<Buffer>,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub node_index: usize,
}

pub struct Primitive {
    pub number_of_indices: u32,
    pub first_index: u32,
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub enum PipelineType {
    GltfAsset,
}

pub struct GltfPipeline {
    pub pipeline: GraphicsPipeline,
    pub assets: Vec<VulkanGltfAsset>,
}

impl GltfPipeline {
    pub fn new(mut renderer: &mut Renderer, asset_names: &[String]) -> Self {
        let shader_entry_point_name =
            &CString::new("main").expect("Failed to create CString for shader entry point name!");

        let vertex_shader = Shader::from_file(
            renderer.context.clone(),
            "examples/assets/shaders/shader.vert.spv",
            vk::ShaderStageFlags::VERTEX,
            shader_entry_point_name,
        )
        .unwrap();

        let fragment_shader = Shader::from_file(
            renderer.context.clone(),
            "examples/assets/shaders/shader.frag.spv",
            vk::ShaderStageFlags::FRAGMENT,
            shader_entry_point_name,
        )
        .unwrap();

        let shader_state_info = [vertex_shader.state_info(), fragment_shader.state_info()];

        let vertex_input_binding_description = vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride((8 * mem::size_of::<f32>()) as _)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build();
        let descriptions = [vertex_input_binding_description];

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

        let attributes = [
            position_description,
            normal_description,
            tex_coord_description,
        ];

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
            width: renderer
                .vulkan_swapchain
                .swapchain
                .properties()
                .extent
                .width as _,
            height: renderer
                .vulkan_swapchain
                .swapchain
                .properties()
                .extent
                .height as _,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let viewports = [viewport];

        // Create a stencil
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: renderer.vulkan_swapchain.swapchain.properties().extent,
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
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
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
        let ubo_binding = UniformBufferObject::get_descriptor_set_layout_bindings();
        let sampler_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build();
        let bindings = [ubo_binding, sampler_binding];

        let layout_create_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(&bindings)
            .build();
        let descriptor_set_layout =
            DescriptorSetLayout::new(renderer.context.clone(), layout_create_info);
        let descriptor_set_layouts = [descriptor_set_layout.layout()];
        let push_constant_range = vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .size(mem::size_of::<PushConstantBlockMaterial>() as u32)
            .build();
        let push_constant_ranges = [push_constant_range];
        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&descriptor_set_layouts) // needed for uniforms in shaders
            .push_constant_ranges(&push_constant_ranges)
            .build();
        let pipeline_layout =
            PipelineLayout::new(renderer.context.clone(), pipeline_layout_create_info);

        // Create the pipeline info
        let pipeline_create_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&shader_state_info)
            .vertex_input_state(&vertex_input_create_info)
            .input_assembly_state(&input_assembly_create_info)
            .viewport_state(&viewport_create_info)
            .rasterization_state(&rasterizer_create_info)
            .multisample_state(&multisampling_create_info)
            .depth_stencil_state(&depth_stencil_info)
            .color_blend_state(&color_blending_info)
            //.dynamic_state // no dynamic states
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

        let mut gltf_pipeline = Self {
            pipeline,
            assets: Vec::new(),
        };

        for asset_name in asset_names.iter() {
            gltf_pipeline.load_gltf_asset(&renderer, &asset_name);
        }

        gltf_pipeline.create_gltf_render_passes(&mut renderer);

        gltf_pipeline
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
                self.create_render_pass(&renderer, framebuffer, *command_buffer, |command_buffer|
                    // TODO: Batch models by which shader should be used to render them
                    unsafe {
                        self.draw_asset(renderer, command_buffer, index);
                    });
            });
    }

    unsafe fn draw_asset(
        &self,
        renderer: &Renderer,
        command_buffer: vk::CommandBuffer,
        command_buffer_index: usize,
    ) {
        self.assets.iter().for_each(|asset| {
            for (scene_index, scene) in asset.gltf.scenes().enumerate() {
                let meshes = &asset.meshes[scene_index];
                for node in scene.nodes() {
                    self.draw_mesh(
                        &renderer,
                        &node,
                        scene_index as u32,
                        &meshes,
                        command_buffer,
                        command_buffer_index,
                    );
                }
            }
        });
    }

    fn draw_mesh(
        &self,
        renderer: &Renderer,
        node: &gltf::Node,
        scene_index: u32,
        meshes: &[Mesh],
        command_buffer: vk::CommandBuffer,
        command_buffer_index: usize,
    ) {
        let offsets = [0];
        if let Some(mesh) = node.mesh() {
            let mesh_info = meshes
                .iter()
                .find(|mesh| mesh.node_index == node.index())
                .expect("Could not find corresponding mesh!");

            let vertex_buffers = [mesh_info.vertex_buffer.buffer()];
            unsafe {
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
                        mesh_info.index_buffer.buffer(),
                        0,
                        vk::IndexType::UINT32,
                    );

                let descriptor_set = mesh_info.descriptor_sets[command_buffer_index];

                renderer
                    .context
                    .logical_device()
                    .logical_device()
                    .cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipeline.layout(),
                        0,
                        &[descriptor_set],
                        &[],
                    );
            }

            for (primitive, primitive_info) in mesh.primitives().zip(mesh_info.primitives.iter()) {
                let mut material = PushConstantBlockMaterial {
                    base_color_factor: glm::vec4(0.0, 0.0, 0.0, 1.0),
                    color_texture_set: -1,
                };

                let primitive_material = primitive.material();
                let pbr = primitive_material.pbr_metallic_roughness();

                if pbr.base_color_texture().is_some() {
                    material.color_texture_set = 0;
                } else {
                    material.base_color_factor = glm::Vec4::from(pbr.base_color_factor());
                }

                unsafe {
                    renderer
                        .context
                        .logical_device()
                        .logical_device()
                        .cmd_push_constants(
                            command_buffer,
                            self.pipeline.layout(),
                            vk::ShaderStageFlags::FRAGMENT,
                            0,
                            Self::byte_slice_from(&material),
                        );

                    renderer
                        .context
                        .logical_device()
                        .logical_device()
                        .cmd_draw_indexed(
                            command_buffer,
                            primitive_info.number_of_indices,
                            1,
                            primitive_info.first_index,
                            0,
                            0,
                        );
                }
            }
        }

        for child_node in node.children() {
            self.draw_mesh(
                &renderer,
                &child_node,
                scene_index as u32,
                &meshes,
                command_buffer,
                command_buffer_index,
            );
        }
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

            // Bind pipeline
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

    pub fn load_gltf_asset(&mut self, renderer: &Renderer, asset_name: &str) {
        let (gltf, buffers, asset_textures) =
            gltf::import(&asset_name).expect("Couldn't import file!");

        let textures = self.load_textures(&renderer, &asset_textures);

        let uniform_buffer_size = mem::size_of::<UniformBufferObject>() as vk::DeviceSize;
        let number_of_meshes = gltf.meshes().len() as u32;
        let number_of_materials = gltf.materials().len() as u32;
        let number_of_swapchain_images = renderer.vulkan_swapchain.swapchain.images().len() as u32;
        let number_of_samplers = number_of_materials * number_of_swapchain_images;

        // TODO: Move descriptor pool creation to method
        let ubo_pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: (4 + number_of_meshes) * number_of_swapchain_images,
        };

        let sampler_pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: number_of_samplers * number_of_swapchain_images,
        };

        let pool_sizes = [ubo_pool_size, sampler_pool_size];

        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes)
            .max_sets((2 + number_of_materials + number_of_meshes) * number_of_swapchain_images)
            .build();

        let descriptor_pool = DescriptorPool::new(renderer.context.clone(), pool_info);

        // TODO: Refactor this to something cleaner
        let mut asset_meshes = Vec::new();
        for scene in gltf.scenes() {
            let mut meshes = Vec::new();
            for node in scene.nodes() {
                self.visit_node(
                    &renderer,
                    &node,
                    &textures,
                    uniform_buffer_size,
                    number_of_swapchain_images,
                    &descriptor_pool,
                    &buffers,
                    &mut meshes,
                );
            }
            asset_meshes.push(meshes);
        }

        let loaded_asset = VulkanGltfAsset {
            gltf,
            textures,
            meshes: asset_meshes,
            descriptor_pool,
        };

        self.assets.push(loaded_asset);
    }

    pub fn visit_node(
        &mut self,
        renderer: &Renderer,
        node: &gltf::Node,
        textures: &[VulkanTexture],
        uniform_buffer_size: vk::DeviceSize,
        number_of_swapchain_images: u32,
        descriptor_pool: &DescriptorPool,
        buffers: &[gltf::buffer::Data],
        mut meshes: &mut Vec<Mesh>,
    ) {
        if let Some(mesh) = node.mesh() {
            let mut vertices = Vec::new();
            let mut indices = Vec::new();

            let uniform_buffers = (0..number_of_swapchain_images)
                .map(|_| {
                    Buffer::new(
                        renderer.context.clone(),
                        uniform_buffer_size,
                        vk::BufferUsageFlags::UNIFORM_BUFFER,
                        vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_COHERENT,
                    )
                })
                .collect::<Vec<_>>();

            let descriptor_sets = descriptor_pool.allocate_descriptor_sets(
                self.pipeline.descriptor_set_layout(),
                number_of_swapchain_images as _,
            );

            let mut all_mesh_primitives = Vec::new();
            for primitive in mesh.primitives() {
                for (descriptor_set, uniform_buffer) in
                    descriptor_sets.iter().zip(uniform_buffers.iter())
                {
                    Self::update_primitive_descriptor_set(
                        &renderer,
                        *descriptor_set,
                        uniform_buffer,
                        primitive.material(),
                        &textures,
                    );
                }

                // Start reading primitive data
                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

                let positions = reader
                    .read_positions()
                    .expect("Failed to read any vertex positions from the model. Vertex positions are required.")
                    .map(glm::Vec3::from)
                    .collect::<Vec<_>>();

                let normals = reader
                    .read_normals()
                    .map_or(vec![glm::vec3(0.0, 0.0, 0.0); positions.len()], |normals| {
                        normals.map(glm::Vec3::from).collect::<Vec<_>>()
                    });

                let convert_coords =
                    |coords: gltf::mesh::util::ReadTexCoords<'_>| -> Vec<glm::Vec2> {
                        coords.into_f32().map(glm::Vec2::from).collect::<Vec<_>>()
                    };
                let tex_coords_0 = reader
                    .read_tex_coords(0)
                    .map_or(vec![glm::vec2(0.0, 0.0); positions.len()], convert_coords);

                // TODO: Add checks to see if normals and tex_coords are even available
                for ((position, normal), tex_coord_0) in positions
                    .iter()
                    .zip(normals.iter())
                    .zip(tex_coords_0.iter())
                {
                    vertices.extend_from_slice(position.as_slice());
                    vertices.extend_from_slice(normal.as_slice());
                    vertices.extend_from_slice(tex_coord_0.as_slice());
                }

                let first_index = indices.len() as u32;

                let primitive_indices = reader
                    .read_indices()
                    .map(|read_indices| read_indices.into_u32().collect::<Vec<_>>())
                    .expect("Failed to read indices!");
                indices.extend_from_slice(&primitive_indices);

                let number_of_indices = primitive_indices.len() as u32;

                all_mesh_primitives.push(Primitive {
                    first_index,
                    number_of_indices,
                });
            }

            let vertex_buffer = renderer.transient_command_pool.create_device_local_buffer(
                renderer.graphics_queue,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                &vertices,
            );

            let index_buffer = renderer.transient_command_pool.create_device_local_buffer(
                renderer.graphics_queue,
                vk::BufferUsageFlags::INDEX_BUFFER,
                &indices,
            );

            meshes.push(Mesh {
                primitives: all_mesh_primitives,
                uniform_buffers,
                descriptor_sets,
                node_index: node.index(),
                vertex_buffer,
                index_buffer,
            });
        }

        for child_node in node.children() {
            self.visit_node(
                &renderer,
                &child_node,
                &textures,
                uniform_buffer_size,
                number_of_swapchain_images,
                &descriptor_pool,
                &buffers,
                &mut meshes,
            );
        }
    }

    fn load_textures(
        &mut self,
        renderer: &Renderer,
        asset_textures: &[gltf::image::Data],
    ) -> Vec<VulkanTexture> {
        let mut textures = Vec::new();
        for texture_properties in asset_textures.iter() {
            let mut texture_format = Self::convert_to_vulkan_format(texture_properties.format);

            let pixels: Vec<u8> = match texture_format {
                vk::Format::R8G8B8_UNORM => {
                    texture_format = vk::Format::R8G8B8A8_UNORM;

                    let image_buffer: RgbImage = ImageBuffer::from_raw(
                        texture_properties.width,
                        texture_properties.height,
                        texture_properties.pixels.to_vec(),
                    )
                    .expect("Failed to create an image buffer");

                    image_buffer
                        .pixels()
                        .flat_map(|pixel| pixel.to_rgba().channels().to_vec())
                        .collect::<Vec<_>>()
                }
                vk::Format::B8G8R8_UNORM => {
                    texture_format = vk::Format::R8G8B8A8_UNORM;

                    let image_buffer: RgbImage = ImageBuffer::from_raw(
                        texture_properties.width,
                        texture_properties.height,
                        texture_properties.pixels.to_vec(),
                    )
                    .expect("Failed to create an image buffer");

                    image_buffer
                        .pixels()
                        .flat_map(|pixel| pixel.to_rgba().channels().to_vec())
                        .collect::<Vec<_>>()
                }
                _ => texture_properties.pixels.to_vec(),
            };

            let create_info_builder = vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .extent(vk::Extent3D {
                    width: texture_properties.width,
                    height: texture_properties.height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .format(texture_format)
                .tiling(vk::ImageTiling::OPTIMAL)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .samples(vk::SampleCountFlags::TYPE_1)
                .flags(vk::ImageCreateFlags::empty());

            let description = TextureDescription {
                format: texture_format,
                dimensions: Dimension {
                    width: texture_properties.width,
                    height: texture_properties.height,
                },
                pixels,
            };

            let texture = Texture::from_data(
                renderer.context.clone(),
                &renderer.command_pool,
                renderer.graphics_queue,
                description,
                create_info_builder.build(),
            );

            let create_info = vk::ImageViewCreateInfo::builder()
                .image(texture.image())
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(texture_format)
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
            let view = ImageView::new(renderer.context.clone(), create_info);

            let sampler_info = vk::SamplerCreateInfo::builder()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::REPEAT)
                .address_mode_v(vk::SamplerAddressMode::REPEAT)
                .address_mode_w(vk::SamplerAddressMode::REPEAT)
                // TODO: Request the anisotropy feature when getting the physical device
                // .anisotropy_enable(true)
                // .max_anisotropy(16.0)
                .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
                .unnormalized_coordinates(false)
                .compare_enable(false)
                .compare_op(vk::CompareOp::ALWAYS)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .mip_lod_bias(0.0)
                .min_lod(0.0)
                .max_lod(0.0)
                .build();
            let sampler = Sampler::new(renderer.context.clone(), sampler_info);

            let vulkan_gltf_texture = VulkanTexture {
                texture,
                view,
                sampler,
            };

            textures.push(vulkan_gltf_texture);
        }
        textures
    }

    fn update_primitive_descriptor_set(
        renderer: &Renderer,
        descriptor_set: vk::DescriptorSet,
        uniform_buffer: &Buffer,
        material: gltf::Material,
        textures: &[VulkanTexture],
    ) {
        let uniform_buffer_size = mem::size_of::<UniformBufferObject>() as vk::DeviceSize;
        let buffer_info = vk::DescriptorBufferInfo::builder()
            .buffer(uniform_buffer.buffer())
            .offset(0)
            .range(uniform_buffer_size)
            .build();
        let buffer_infos = [buffer_info];

        let ubo_descriptor_write = vk::WriteDescriptorSet::builder()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&buffer_infos)
            .build();

        let pbr = material.pbr_metallic_roughness();

        if let Some(base_color_texture) = pbr.base_color_texture() {
            let base_color_index = base_color_texture.texture().index();
            let texture_view = textures[base_color_index].view.view();
            let texture_sampler = textures[base_color_index].sampler.sampler();

            let image_info = vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(texture_view)
                .sampler(texture_sampler)
                .build();
            let image_infos = [image_info];

            let sampler_descriptor_write = vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&image_infos)
                .build();

            let descriptor_writes = [ubo_descriptor_write, sampler_descriptor_write];

            unsafe {
                renderer
                    .context
                    .logical_device()
                    .logical_device()
                    .update_descriptor_sets(&descriptor_writes, &[])
            }
        } else {
            let descriptor_writes = [ubo_descriptor_write];
            unsafe {
                renderer
                    .context
                    .logical_device()
                    .logical_device()
                    .update_descriptor_sets(&descriptor_writes, &[])
            }
        }
    }

    pub fn convert_to_vulkan_format(format: Format) -> vk::Format {
        match format {
            Format::R8 => vk::Format::R8_UNORM,
            Format::R8G8 => vk::Format::R8G8_UNORM,
            Format::R8G8B8A8 => vk::Format::R8G8B8A8_UNORM,
            Format::B8G8R8A8 => vk::Format::B8G8R8A8_UNORM,
            // 24-bit formats will have an alpha channel added
            // to make them 32-bit
            Format::R8G8B8 => vk::Format::R8G8B8_UNORM,
            Format::B8G8R8 => vk::Format::B8G8R8_UNORM,
        }
    }

    // TODO: Move this to a seperate class or even the mod.rs file
    unsafe fn byte_slice_from<T: Sized>(data: &T) -> &[u8] {
        let data_ptr = (data as *const T) as *const u8;
        slice::from_raw_parts(data_ptr, std::mem::size_of::<T>())
    }
}
