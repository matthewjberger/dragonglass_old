use crate::{
    core::{ImageView, Swapchain, SwapchainProperties, VulkanContext},
    render::{Framebuffer, GraphicsPipeline, RenderPass, UniformBufferObject},
    resource::{
        Buffer, CommandPool, DescriptorPool, DescriptorSetLayout, Dimension, PipelineLayout,
        Sampler, Shader, Texture, TextureDescription,
    },
    sync::SynchronizationSet,
};
use ash::{version::DeviceV1_0, vk};
use gltf::image::Format;
use image::{ImageBuffer, Pixel, RgbImage};
use nalgebra_glm as glm;
use std::{ffi::CString, mem, sync::Arc};

pub struct VulkanGltfAsset {
    pub gltf: gltf::Document,
    pub textures: Vec<VulkanTexture>,
    pub meshes: Vec<Mesh>,
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
    pub index: usize,
}

pub struct Primitive {
    pub number_of_indices: u32,
    pub first_index: u32,
}

pub struct Renderer {
    pub context: Arc<VulkanContext>,
    pub command_pool: CommandPool,
    pub descriptor_pools: Vec<DescriptorPool>,
    pub framebuffers: Vec<Framebuffer>,
    pub graphics_queue: vk::Queue,
    pub pipeline: GraphicsPipeline,
    pub present_queue: vk::Queue,
    pub render_pass: RenderPass,
    pub swapchain: Swapchain,
    pub transient_command_pool: CommandPool,
    pub depth_texture: Texture,
    pub depth_texture_view: ImageView,
    pub synchronization_set: SynchronizationSet,
    pub current_frame: usize,
    pub assets: Vec<VulkanGltfAsset>,
}

impl Renderer {
    pub fn new(window: &winit::Window) -> Self {
        let context =
            Arc::new(VulkanContext::new(&window).expect("Failed to create VulkanContext"));

        let synchronization_set =
            SynchronizationSet::new(context.clone()).expect("Failed to create sync objects");

        unsafe {
            context
                .logical_device()
                .logical_device()
                .device_wait_idle()
                .expect("Failed to wait for the logical device to be idle!")
        };

        let graphics_queue = unsafe {
            context
                .logical_device()
                .logical_device()
                .get_device_queue(context.graphics_queue_family_index(), 0)
        };

        let present_queue = unsafe {
            context
                .logical_device()
                .logical_device()
                .get_device_queue(context.present_queue_family_index(), 0)
        };

        let depth_format = context.determine_depth_format(
            vk::ImageTiling::OPTIMAL,
            vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
        );

        let logical_size = window
            .get_inner_size()
            .expect("Failed to get the window's inner size!");
        let dimensions = [logical_size.width as u32, logical_size.height as u32];
        let swapchain = Swapchain::new(context.clone(), dimensions);
        let render_pass = RenderPass::new(context.clone(), swapchain.properties(), depth_format);

        let pipeline = Self::create_graphics_pipeline(
            context.clone(),
            swapchain.properties(),
            render_pass.render_pass(),
        );

        let create_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width: swapchain.properties().extent.width,
                height: swapchain.properties().extent.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .format(depth_format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(vk::SampleCountFlags::TYPE_1)
            .flags(vk::ImageCreateFlags::empty())
            .build();
        let depth_texture = Texture::new(context.clone(), create_info);

        let command_pool = CommandPool::new(context.clone(), vk::CommandPoolCreateFlags::empty());
        let transient_command_pool =
            CommandPool::new(context.clone(), vk::CommandPoolCreateFlags::TRANSIENT);

        command_pool.transition_image_layout(
            graphics_queue,
            depth_texture.image(),
            depth_format,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        );

        let create_info = vk::ImageViewCreateInfo::builder()
            .image(depth_texture.image())
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(depth_format)
            .components(vk::ComponentMapping {
                r: vk::ComponentSwizzle::IDENTITY,
                g: vk::ComponentSwizzle::IDENTITY,
                b: vk::ComponentSwizzle::IDENTITY,
                a: vk::ComponentSwizzle::IDENTITY,
            })
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .build();
        let depth_texture_view = ImageView::new(context.clone(), create_info);

        // Create one framebuffer for each image in the swapchain
        let framebuffers = swapchain
            .image_views()
            .iter()
            .map(|view| [view.view(), depth_texture_view.view()])
            .map(|attachments| {
                let create_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(render_pass.render_pass())
                    .attachments(&attachments)
                    .width(swapchain.properties().extent.width)
                    .height(swapchain.properties().extent.height)
                    .layers(1)
                    .build();
                Framebuffer::new(context.clone(), create_info)
            })
            .collect::<Vec<_>>();

        Renderer {
            command_pool,
            context,
            descriptor_pools: Vec::new(), // TODO: maybe make this a map and have a main descriptor pool
            framebuffers,
            graphics_queue,
            pipeline,
            present_queue,
            render_pass,
            swapchain,
            synchronization_set,
            depth_texture,
            depth_texture_view,
            transient_command_pool,
            current_frame: 0,
            assets: Vec::new(),
        }
    }

    pub fn create_render_pass<F>(
        &self,
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
            .render_pass(self.render_pass.render_pass())
            .framebuffer(framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.swapchain.properties().extent,
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

            // Bind pipeline
            self.context
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
            self.context
                .logical_device()
                .logical_device()
                .cmd_end_render_pass(command_buffer);

            // End command buffer
            self.context
                .logical_device()
                .logical_device()
                .end_command_buffer(command_buffer)
                .expect("Failed to end the command buffer for a render pass!");
        }
    }

    #[allow(dead_code)]
    pub fn recreate_swapchain(&mut self, _: Option<[u32; 2]>) {
        log::debug!("Recreating swapchain");
        // TODO: Implement swapchain recreation
    }

    pub fn wait_idle(&self) {
        unsafe {
            self.context
                .logical_device()
                .logical_device()
                .device_wait_idle()
                .expect("Failed to wait for the logical device to be idle!")
        };
    }

    fn create_graphics_pipeline(
        context: Arc<VulkanContext>,
        swapchain_properties: &SwapchainProperties,
        render_pass: vk::RenderPass,
    ) -> GraphicsPipeline {
        let shader_entry_point_name =
            &CString::new("main").expect("Failed to create CString for shader entry point name!");

        let vertex_shader = Shader::from_file(
            context.clone(),
            "examples/assets/shaders/shader.vert.spv",
            vk::ShaderStageFlags::VERTEX,
            shader_entry_point_name,
        )
        .unwrap();

        let fragment_shader = Shader::from_file(
            context.clone(),
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
        let descriptor_set_layout = DescriptorSetLayout::new(context.clone(), layout_create_info);
        let descriptor_set_layouts = [descriptor_set_layout.layout()];
        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&descriptor_set_layouts) // needed for uniforms in shaders
            // .push_constant_ranges()
            .build();
        let pipeline_layout = PipelineLayout::new(context.clone(), pipeline_layout_create_info);

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
            .render_pass(render_pass)
            .subpass(0)
            .build();

        GraphicsPipeline::new(
            context,
            pipeline_create_info,
            pipeline_layout,
            descriptor_set_layout,
        )
    }

    pub fn load_gltf_asset(&mut self, asset_name: &str) {
        let (gltf, buffers, asset_textures) =
            gltf::import(&asset_name).expect("Couldn't import file!");

        let textures = self.load_textures(&asset_textures);

        let uniform_buffer_size = mem::size_of::<UniformBufferObject>() as vk::DeviceSize;
        let number_of_meshes = gltf.meshes().len() as u32;
        let number_of_materials = textures.len() as u32;
        let number_of_swapchain_images = self.swapchain.images().len() as u32;
        let number_of_samplers = number_of_materials * number_of_swapchain_images;

        let ubo_pool_size = (4 + number_of_meshes) * number_of_swapchain_images;
        let sampler_pool_size = number_of_samplers * number_of_swapchain_images;
        let max_number_of_pools =
            (2 + number_of_materials + number_of_meshes) * number_of_swapchain_images;

        let descriptor_pool = DescriptorPool::new(
            self.context.clone(),
            ubo_pool_size,
            sampler_pool_size,
            max_number_of_pools,
        );

        let mut asset_meshes = Vec::new();
        for mesh in gltf.meshes() {
            let mut vertices = Vec::new();
            let mut indices = Vec::new();

            let uniform_buffers = (0..number_of_swapchain_images)
                .map(|_| {
                    Buffer::new(
                        self.context.clone(),
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
                    Self::update_model_descriptor_set(
                        &self,
                        *descriptor_set,
                        uniform_buffer,
                        primitive.material(),
                        &textures,
                    );
                }

                // Start reading primitive data
                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

                let positions = reader.read_positions().map_or(Vec::new(), |positions| {
                    positions.map(glm::Vec3::from).collect::<Vec<_>>()
                });

                let normals = reader.read_normals().map_or(Vec::new(), |normals| {
                    normals.map(glm::Vec3::from).collect::<Vec<_>>()
                });

                let convert_coords =
                    |coords: gltf::mesh::util::ReadTexCoords<'_>| -> Vec<glm::Vec2> {
                        coords.into_f32().map(glm::Vec2::from).collect::<Vec<_>>()
                    };
                let tex_coords_0 = reader.read_tex_coords(0).map_or(Vec::new(), convert_coords);

                // TODO: Add checks to see if normals and tex_coords are even available
                for (index, position) in positions.iter().enumerate() {
                    vertices.extend_from_slice(position.as_slice());
                    vertices.extend_from_slice(normals.get(index).copied().unwrap().as_slice());
                    vertices
                        .extend_from_slice(tex_coords_0.get(index).copied().unwrap().as_slice());
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

            let vertex_buffer = self.transient_command_pool.create_device_local_buffer(
                self.graphics_queue,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                &vertices,
            );

            let index_buffer = self.transient_command_pool.create_device_local_buffer(
                self.graphics_queue,
                vk::BufferUsageFlags::INDEX_BUFFER,
                &indices,
            );

            asset_meshes.push(Mesh {
                primitives: all_mesh_primitives,
                uniform_buffers,
                descriptor_sets,
                index: mesh.index(),
                vertex_buffer,
                index_buffer,
            });
        }

        let loaded_asset = VulkanGltfAsset {
            gltf,
            textures,
            meshes: asset_meshes,
            descriptor_pool,
        };

        self.assets.push(loaded_asset);
    }

    fn load_textures(&mut self, asset_textures: &[gltf::image::Data]) -> Vec<VulkanTexture> {
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
                self.context.clone(),
                &self.command_pool,
                self.graphics_queue,
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
            let view = ImageView::new(self.context.clone(), create_info);

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
            let sampler = Sampler::new(self.context.clone(), sampler_info);

            let vulkan_gltf_texture = VulkanTexture {
                texture,
                view,
                sampler,
            };

            textures.push(vulkan_gltf_texture);
        }
        textures
    }

    fn update_model_descriptor_set(
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
        let base_color_index = pbr
            .base_color_texture()
            .expect("Failed to get base color texture!")
            .texture()
            .index();
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
    }

    pub fn create_render_passes(&mut self) {
        // Allocate one command buffer per swapchain image
        let number_of_framebuffers = self.framebuffers.len();
        self.command_pool
            .allocate_command_buffers(number_of_framebuffers as _);

        // Create a single render pass per swapchain image that will draw each mesh
        self.command_pool
            .command_buffers()
            .iter()
            .enumerate()
            .for_each(|(index, buffer)| {
                let command_buffer = buffer;
                let framebuffer = self.framebuffers[index].framebuffer();
                self.create_render_pass(framebuffer, *command_buffer, |command_buffer|
                    // TODO: Batch models by which shader should be used to render them
                    unsafe {
                        self.draw_asset(command_buffer, index);
                    });
            });
    }

    unsafe fn draw_asset(&self, command_buffer: vk::CommandBuffer, command_buffer_index: usize) {
        self.assets.iter().for_each(|asset| {
            let offsets = [0];
            for asset_mesh in asset.gltf.meshes() {
                let index = asset_mesh.index();
                let mesh = asset
                    .meshes
                    .iter()
                    .find(|mesh| mesh.index == index)
                    .expect("Could not find corresponding mesh!");

                let vertex_buffers = [mesh.vertex_buffer.buffer()];
                self.context
                    .logical_device()
                    .logical_device()
                    .cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets);

                self.context
                    .logical_device()
                    .logical_device()
                    .cmd_bind_index_buffer(
                        command_buffer,
                        mesh.index_buffer.buffer(),
                        0,
                        vk::IndexType::UINT32,
                    );

                let descriptor_set = mesh.descriptor_sets[command_buffer_index];

                self.context
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

                for primitive in mesh.primitives.iter() {
                    self.context
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
        });
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
}
