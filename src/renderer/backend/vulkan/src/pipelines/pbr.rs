use crate::{
    core::VulkanContext,
    model::gltf::{GltfAsset, Primitive},
    render::{GraphicsPipeline, Renderer},
    resource::{
        texture::TextureBundle, Buffer, DescriptorPool, DescriptorSetLayout, DummyImage,
        PipelineLayout, Shader,
    },
};
use ash::{version::DeviceV1_0, vk};
use dragonglass_core::byte_slice_from;
use nalgebra_glm as glm;
use std::{ffi::CString, mem, sync::Arc};

pub struct PushConstantBlockMaterial {
    pub base_color_factor: glm::Vec4,
    pub emissive_factor: glm::Vec3,
    pub color_texture_set: i32,
    pub metallic_roughness_texture_set: i32, // B channel - metalness values. G channel - roughness values
    pub normal_texture_set: i32,
    pub occlusion_texture_set: i32, // R channel - occlusion values
    pub emissive_texture_set: i32,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub alpha_mask: i32,
    pub alpha_mask_cutoff: f32,
}

pub struct PbrPipeline {
    pub pipeline: GraphicsPipeline,
}

impl PbrPipeline {
    pub fn new(renderer: &mut Renderer) -> Self {
        let (vertex_shader, fragment_shader, _shader_entry_point_name) =
            Self::create_shaders(renderer.context.clone());
        let shader_state_info = [vertex_shader.state_info(), fragment_shader.state_info()];

        let descriptions = GltfAsset::create_vertex_input_descriptions();
        let attributes = GltfAsset::create_vertex_attributes();
        let vertex_input_create_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&descriptions)
            .vertex_attribute_descriptions(&attributes)
            .build();

        let input_assembly_create_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false)
            .build();

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
            .sample_shading_enable(true)
            .rasterization_samples(renderer.context.max_usable_samples())
            .min_sample_shading(0.2)
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

        let descriptor_set_layout =
            PbrPipelineData::descriptor_set_layout(renderer.context.clone());
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
            .render_pass(renderer.vulkan_swapchain().render_pass.render_pass())
            .subpass(0)
            .build();

        let pipeline = GraphicsPipeline::new(
            renderer.context.clone(),
            pipeline_create_info,
            pipeline_layout,
            descriptor_set_layout,
        );

        Self { pipeline }
    }

    fn create_shaders(context: Arc<VulkanContext>) -> (Shader, Shader, CString) {
        let shader_entry_point_name =
            CString::new("main").expect("Failed to create CString for shader entry point name!");

        let vertex_shader = Shader::from_file(
            context.clone(),
            "examples/assets/shaders/pbr.vert.spv",
            vk::ShaderStageFlags::VERTEX,
            &shader_entry_point_name,
        )
        .expect("Failed to create vertex shader!");

        let fragment_shader = Shader::from_file(
            context,
            "examples/assets/shaders/pbr.frag.spv",
            vk::ShaderStageFlags::FRAGMENT,
            &shader_entry_point_name,
        )
        .expect("Failed to create fragment shader!");

        (vertex_shader, fragment_shader, shader_entry_point_name)
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
            .set_layouts(&descriptor_set_layouts)
            .push_constant_ranges(&push_constant_ranges)
            .build();

        PipelineLayout::new(context, pipeline_layout_create_info)
    }

    pub fn bind(&self, device: &ash::Device, command_buffer: vk::CommandBuffer) {
        unsafe {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.pipeline(),
            );
        }
    }
}

#[derive(Clone, Copy)]
pub struct UniformBufferObject {
    pub view: glm::Mat4,
    pub projection: glm::Mat4,
    pub camera_position: glm::Vec4,
    pub joint_matrices: [glm::Mat4; UniformBufferObject::MAX_NUM_JOINTS],
}

impl UniformBufferObject {
    // This needs to match the defined value in the shaders
    pub const MAX_NUM_JOINTS: usize = 128;
}

#[derive(Debug, Clone, Copy)]
pub struct DynamicUniformBufferObject {
    pub model: glm::Mat4,
    // X value is the joint count.
    // Y value is the joint matrix offset.
    // A vec4 is necessary for proper alignment
    pub joint_info: glm::Vec4,
}

pub struct PbrPipelineData {
    pub descriptor_pool: DescriptorPool,
    pub uniform_buffer: Buffer,
    pub dynamic_uniform_buffer: Buffer,
    pub dynamic_alignment: u64,
    pub descriptor_set: vk::DescriptorSet,
    pub dummy: DummyImage,
}

impl PbrPipelineData {
    // This should match the number of textures defined in the shader
    pub const MAX_TEXTURES: usize = 100;

    pub fn new(renderer: &Renderer, number_of_meshes: usize, textures: &[&TextureBundle]) -> Self {
        let descriptor_set_layout = Self::descriptor_set_layout(renderer.context.clone());
        let descriptor_pool = Self::create_descriptor_pool(renderer.context.clone());
        let descriptor_set =
            descriptor_pool.allocate_descriptor_sets(descriptor_set_layout.layout(), 1)[0];

        let uniform_buffer = Buffer::new_mapped_basic(
            renderer.context.clone(),
            mem::size_of::<UniformBufferObject>() as _,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk_mem::MemoryUsage::CpuToGpu,
        );

        let dynamic_alignment = Self::calculate_dynamic_alignment(renderer.context.clone());

        let dynamic_uniform_buffer = Buffer::new_mapped_basic(
            renderer.context.clone(),
            (number_of_meshes as u64 * dynamic_alignment) as vk::DeviceSize,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk_mem::MemoryUsage::CpuToGpu,
        );

        let data = PbrPipelineData {
            descriptor_pool,
            uniform_buffer,
            dynamic_uniform_buffer,
            descriptor_set,
            dynamic_alignment,
            dummy: DummyImage::new(renderer.context.clone(), &renderer.transient_command_pool),
        };

        data.update_descriptor_set(
            &renderer,
            renderer.context.clone(),
            number_of_meshes,
            &textures,
        );

        data
    }

    fn calculate_dynamic_alignment(context: Arc<VulkanContext>) -> u64 {
        let minimum_ubo_alignment = context
            .physical_device_properties()
            .limits
            .min_uniform_buffer_offset_alignment;
        let dynamic_alignment = std::mem::size_of::<DynamicUniformBufferObject>() as u64;
        if minimum_ubo_alignment > 0 {
            (dynamic_alignment + minimum_ubo_alignment - 1) & !(minimum_ubo_alignment - 1)
        } else {
            dynamic_alignment
        }
    }

    pub fn descriptor_set_layout(context: Arc<VulkanContext>) -> DescriptorSetLayout {
        let ubo_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .build();
        let dynamic_ubo_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .build();
        let sampler_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(2)
            .descriptor_count(Self::MAX_TEXTURES as _)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build();
        let irradiance_cubemap_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(3)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build();
        let prefilter_cubemap_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(4)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build();
        let brdflut_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(5)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build();

        let bindings = [
            ubo_binding,
            dynamic_ubo_binding,
            sampler_binding,
            irradiance_cubemap_binding,
            prefilter_cubemap_binding,
            brdflut_binding,
        ];

        let layout_create_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(&bindings)
            .build();
        DescriptorSetLayout::new(context, layout_create_info)
    }

    fn create_descriptor_pool(context: Arc<VulkanContext>) -> DescriptorPool {
        let ubo_pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
        };

        let dynamic_ubo_pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
            descriptor_count: 1,
        };

        let sampler_pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: Self::MAX_TEXTURES as _,
        };

        let irradiance_cubemap_pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
        };

        let prefilter_cubemap_pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
        };

        let brdflut_pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
        };

        let pool_sizes = [
            ubo_pool_size,
            dynamic_ubo_pool_size,
            sampler_pool_size,
            irradiance_cubemap_pool_size,
            prefilter_cubemap_pool_size,
            brdflut_pool_size,
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes)
            .max_sets(1)
            .build();

        DescriptorPool::new(context, pool_info)
    }

    fn update_descriptor_set(
        &self,
        renderer: &Renderer,
        context: Arc<VulkanContext>,
        number_of_meshes: usize,
        textures: &[&TextureBundle],
    ) {
        let uniform_buffer_size = mem::size_of::<UniformBufferObject>() as vk::DeviceSize;
        let buffer_info = vk::DescriptorBufferInfo::builder()
            .buffer(self.uniform_buffer.buffer())
            .offset(0)
            .range(uniform_buffer_size)
            .build();
        let buffer_infos = [buffer_info];

        let dynamic_uniform_buffer_size =
            (number_of_meshes as u64 * self.dynamic_alignment) as vk::DeviceSize;
        let dynamic_buffer_info = vk::DescriptorBufferInfo::builder()
            .buffer(self.dynamic_uniform_buffer.buffer())
            .offset(0)
            .range(dynamic_uniform_buffer_size)
            .build();
        let dynamic_buffer_infos = [dynamic_buffer_info];

        let mut image_infos = textures
            .iter()
            .map(|texture| {
                vk::DescriptorImageInfo::builder()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(texture.view.view())
                    .sampler(texture.sampler.sampler())
                    .build()
            })
            .collect::<Vec<_>>();

        let number_of_images = image_infos.len();
        let required_images = Self::MAX_TEXTURES;
        if number_of_images < required_images {
            let remaining = required_images - number_of_images;
            for _ in 0..remaining {
                image_infos.push(
                    vk::DescriptorImageInfo::builder()
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .image_view(self.dummy.view().view())
                        .sampler(self.dummy.sampler().sampler())
                        .build(),
                );
            }
        }

        let irradiance_map = renderer
            .irradiance_map
            .as_ref()
            .expect("Failed to get irradiance map!");
        let irradiance_cubemap_image_info = vk::DescriptorImageInfo::builder()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(irradiance_map.cubemap.view.view())
            .sampler(irradiance_map.cubemap.sampler.sampler())
            .build();
        let irradiance_cubemap_image_infos = [irradiance_cubemap_image_info];

        let prefilter_map = renderer
            .prefilter_map
            .as_ref()
            .expect("Failed to get prefilter map!");
        let prefilter_cubemap_image_info = vk::DescriptorImageInfo::builder()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(prefilter_map.cubemap.view.view())
            .sampler(prefilter_map.cubemap.sampler.sampler())
            .build();
        let prefilter_cubemap_image_infos = [prefilter_cubemap_image_info];

        let brdflut = renderer.brdflut.as_ref().expect("Failed to get brdflut!");
        let brdflut_image_info = vk::DescriptorImageInfo::builder()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(brdflut.view.view())
            .sampler(brdflut.sampler.sampler())
            .build();
        let brdflut_image_infos = [brdflut_image_info];

        let ubo_descriptor_write = vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&buffer_infos)
            .build();

        let dynamic_ubo_descriptor_write = vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set)
            .dst_binding(1)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
            .buffer_info(&dynamic_buffer_infos)
            .build();

        let sampler_descriptor_write = vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set)
            .dst_binding(2)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image_infos)
            .build();

        let irradiance_cubemap_descriptor_write = vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set)
            .dst_binding(3)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&irradiance_cubemap_image_infos)
            .build();

        let prefilter_cubemap_descriptor_write = vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set)
            .dst_binding(4)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&prefilter_cubemap_image_infos)
            .build();

        let brdflut_descriptor_write = vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set)
            .dst_binding(5)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&brdflut_image_infos)
            .build();

        let descriptor_writes = vec![
            ubo_descriptor_write,
            dynamic_ubo_descriptor_write,
            sampler_descriptor_write,
            irradiance_cubemap_descriptor_write,
            prefilter_cubemap_descriptor_write,
            brdflut_descriptor_write,
        ];

        unsafe {
            context
                .logical_device()
                .logical_device()
                .update_descriptor_sets(&descriptor_writes, &[])
        }
    }
}

pub struct PbrRenderer {
    command_buffer: vk::CommandBuffer,
    pipeline_layout: vk::PipelineLayout,
    dynamic_alignment: u64,
    descriptor_set: vk::DescriptorSet,
}

impl PbrRenderer {
    pub fn new(
        command_buffer: vk::CommandBuffer,
        pipeline: &PbrPipeline,
        pipeline_data: &PbrPipelineData,
    ) -> Self {
        Self {
            command_buffer,
            pipeline_layout: pipeline.pipeline.layout(),
            dynamic_alignment: pipeline_data.dynamic_alignment,
            descriptor_set: pipeline_data.descriptor_set,
        }
    }

    pub fn draw_asset(
        &self,
        device: &ash::Device,
        asset: &GltfAsset,
        texture_offset: i32,
        mesh_offset: usize,
        index_offset: u32,
        vertex_offset: u32,
    ) {
        asset.walk(|node_index, graph| {
            if let Some(mesh) = graph[node_index].mesh.as_ref() {
                unsafe {
                    device.cmd_bind_descriptor_sets(
                        self.command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipeline_layout,
                        0,
                        &[self.descriptor_set],
                        &[((mesh_offset + mesh.mesh_id) as u64 * self.dynamic_alignment) as _],
                    );
                }

                for primitive in mesh.primitives.iter() {
                    let material = Self::create_material(&asset, &primitive, texture_offset);
                    unsafe {
                        device.cmd_push_constants(
                            self.command_buffer,
                            self.pipeline_layout,
                            vk::ShaderStageFlags::ALL_GRAPHICS,
                            0,
                            byte_slice_from(&material),
                        );

                        device.cmd_draw_indexed(
                            self.command_buffer,
                            primitive.number_of_indices,
                            1,
                            index_offset + primitive.first_index,
                            vertex_offset as _,
                            0,
                        );
                    }
                }
            }
        });
    }

    fn create_material(
        asset: &GltfAsset,
        primitive: &Primitive,
        texture_offset: i32,
    ) -> PushConstantBlockMaterial {
        let mut material = PushConstantBlockMaterial {
            base_color_factor: glm::vec4(0.0, 0.0, 0.0, 1.0),
            emissive_factor: glm::Vec3::identity(),
            color_texture_set: -1,
            metallic_roughness_texture_set: -1,
            normal_texture_set: -1,
            occlusion_texture_set: -1,
            emissive_texture_set: -1,
            metallic_factor: 0.0,
            roughness_factor: 0.0,
            alpha_mask: gltf::material::AlphaMode::Opaque as i32,
            alpha_mask_cutoff: 0.0,
        };

        if let Some(material_index) = primitive.material_index {
            let primitive_material = asset
                .gltf
                .materials()
                .nth(material_index)
                .expect("Failed to retrieve material!");
            let pbr = primitive_material.pbr_metallic_roughness();

            material.base_color_factor = glm::Vec4::from(pbr.base_color_factor());
            material.metallic_factor = pbr.metallic_factor();
            material.roughness_factor = pbr.roughness_factor();
            material.emissive_factor = glm::Vec3::from(primitive_material.emissive_factor());
            material.alpha_mask_cutoff = primitive_material.alpha_cutoff();
            material.alpha_mask = primitive_material.alpha_mode() as i32;

            if let Some(base_color_texture) = pbr.base_color_texture() {
                material.color_texture_set =
                    texture_offset + base_color_texture.texture().index() as i32;
            }

            if let Some(metallic_roughness_texture) = pbr.metallic_roughness_texture() {
                material.metallic_roughness_texture_set =
                    texture_offset + metallic_roughness_texture.texture().index() as i32;
            }

            if let Some(normal_texture) = primitive_material.normal_texture() {
                material.normal_texture_set =
                    texture_offset + normal_texture.texture().index() as i32;
            }

            if let Some(occlusion_texture) = primitive_material.occlusion_texture() {
                material.occlusion_texture_set =
                    texture_offset + occlusion_texture.texture().index() as i32;
            }

            if let Some(emissive_texture) = primitive_material.emissive_texture() {
                material.emissive_texture_set =
                    texture_offset + emissive_texture.texture().index() as i32;
            }
        }

        material
    }
}
