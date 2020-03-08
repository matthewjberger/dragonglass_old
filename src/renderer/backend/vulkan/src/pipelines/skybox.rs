use crate::{
    core::VulkanContext,
    model::gltf::GltfTextureData,
    render::{GraphicsPipeline, Renderer},
    resource::{
        Buffer, CommandPool, DescriptorPool, DescriptorSetLayout, ImageView, PipelineLayout,
        Sampler, Shader, Texture, TextureDescription,
    },
};
use ash::{version::DeviceV1_0, vk};
use nalgebra_glm as glm;
use std::{ffi::CString, mem, sync::Arc};

pub struct SkyboxPipeline {
    pub pipeline: GraphicsPipeline,
}

impl SkyboxPipeline {
    pub fn new(renderer: &mut Renderer) -> Self {
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
            SkyboxPipelineData::descriptor_set_layout(renderer.context.clone());
        let descriptor_set_layouts = [descriptor_set_layout.layout()];
        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&descriptor_set_layouts) // needed for uniforms in shaders
            .build();
        let pipeline_layout =
            PipelineLayout::new(renderer.context.clone(), pipeline_layout_create_info);

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
            .render_pass(renderer.vulkan_swapchain.render_pass.render_pass())
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
            "examples/assets/shaders/skybox.vert.spv",
            vk::ShaderStageFlags::VERTEX,
            &shader_entry_point_name,
        )
        .expect("Failed to create vertex shader!");

        let fragment_shader = Shader::from_file(
            context,
            "examples/assets/shaders/skybox.frag.spv",
            vk::ShaderStageFlags::FRAGMENT,
            &shader_entry_point_name,
        )
        .expect("Failed to create fragment shader!");

        (vertex_shader, fragment_shader, shader_entry_point_name)
    }

    fn create_vertex_input_descriptions() -> [vk::VertexInputBindingDescription; 1] {
        let vertex_input_binding_description = vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride((5 * mem::size_of::<f32>()) as _)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build();
        [vertex_input_binding_description]
    }

    fn create_vertex_attributes() -> [vk::VertexInputAttributeDescription; 2] {
        let position_description = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0)
            .build();

        let tex_coord_description = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32_SFLOAT)
            .offset((3 * mem::size_of::<f32>()) as _)
            .build();

        [position_description, tex_coord_description]
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
}

#[derive(Debug, Clone, Copy)]
pub struct UniformBufferObject {
    pub view: glm::Mat4,
    pub projection: glm::Mat4,
}

pub struct SkyboxPipelineData {
    pub descriptor_pool: DescriptorPool,
    pub descriptor_set: vk::DescriptorSet,
    pub uniform_buffer: Buffer,
}

impl SkyboxPipelineData {
    pub fn new(renderer: &Renderer, textures: &[&GltfTextureData]) -> Self {
        let descriptor_set_layout = Self::descriptor_set_layout(renderer.context.clone());
        let descriptor_pool = Self::create_descriptor_pool(renderer.context.clone());
        let descriptor_set =
            descriptor_pool.allocate_descriptor_sets(descriptor_set_layout.layout(), 1)[0];

        let uniform_buffer = Buffer::new_mapped_basic(
            renderer.context.clone(),
            mem::size_of::<UniformBufferObject>() as _,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk_mem::MemoryUsage::CpuOnly,
        );

        let data = SkyboxPipelineData {
            descriptor_pool,
            descriptor_set,
            uniform_buffer,
        };

        data.update_descriptor_set(renderer.context.clone(), &textures);
        data
    }

    pub fn descriptor_set_layout(context: Arc<VulkanContext>) -> DescriptorSetLayout {
        let ubo_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .build();
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
        DescriptorSetLayout::new(context, layout_create_info)
    }

    fn create_descriptor_pool(context: Arc<VulkanContext>) -> DescriptorPool {
        let ubo_pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
        };

        let sampler_pool_size = vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
        };

        let pool_sizes = [ubo_pool_size, sampler_pool_size];

        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes)
            .max_sets(1)
            .build();

        DescriptorPool::new(context, pool_info)
    }

    fn update_descriptor_set(&self, context: Arc<VulkanContext>, textures: &[&GltfTextureData]) {
        let uniform_buffer_size = mem::size_of::<UniformBufferObject>() as vk::DeviceSize;
        let buffer_info = vk::DescriptorBufferInfo::builder()
            .buffer(self.uniform_buffer.buffer())
            .offset(0)
            .range(uniform_buffer_size)
            .build();
        let buffer_infos = [buffer_info];

        let image_infos = textures
            .iter()
            .map(|texture| {
                vk::DescriptorImageInfo::builder()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(texture.view.view())
                    .sampler(texture.sampler.sampler())
                    .build()
            })
            .collect::<Vec<_>>();

        let ubo_descriptor_write = vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&buffer_infos)
            .build();

        let sampler_descriptor_write = vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set)
            .dst_binding(1)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image_infos)
            .build();

        let descriptor_writes = vec![ubo_descriptor_write, sampler_descriptor_write];

        unsafe {
            context
                .logical_device()
                .logical_device()
                .update_descriptor_sets(&descriptor_writes, &[])
        }
    }
}

pub struct SkyboxTextureData {
    pub texture: Texture,
    pub view: ImageView,
    pub sampler: Sampler,
}

impl SkyboxTextureData {
    pub fn new(renderer: &Renderer, image_data: &gltf::image::Data) -> Self {
        let description = TextureDescription::from_gltf(&image_data);

        let texture = Self::create_texture(renderer.context.clone(), &description);

        Self::upload_texture_data(
            &renderer,
            &renderer.command_pool,
            renderer.context.graphics_queue(),
            &texture,
            &description,
        );

        let view = Self::create_image_view(renderer.context.clone(), &texture, description.format);

        let sampler = Self::create_sampler(renderer.context.clone());

        SkyboxTextureData {
            texture,
            view,
            sampler,
        }
    }

    pub fn upload_texture_data(
        renderer: &Renderer,
        command_pool: &CommandPool,
        graphics_queue: vk::Queue,
        texture: &Texture,
        description: &TextureDescription,
    ) {
        let region = vk::BufferImageCopy::builder()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 6,
            })
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                width: description.width,
                height: description.height,
                depth: 1,
            })
            .build();
        let regions = [region];
        let buffer = Buffer::new_mapped_basic(
            renderer.context.clone(),
            texture.allocation_info().get_size() as _,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk_mem::MemoryUsage::CpuToGpu,
        );
        buffer.upload_to_buffer(&description.pixels, 0, std::mem::align_of::<u8>() as _);

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
                layer_count: 1,
            })
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .build();
        let barriers = [barrier];

        command_pool.transition_image_layout(
            graphics_queue,
            &barriers,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
        );

        command_pool.copy_buffer_to_image(
            graphics_queue,
            buffer.buffer(),
            texture.image(),
            &regions,
        );

        let barrier = vk::ImageMemoryBarrier::builder()
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(texture.image())
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .build();
        let barriers = [barrier];

        command_pool.transition_image_layout(
            graphics_queue,
            &barriers,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        );
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
            .array_layers(6)
            .format(description.format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
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

    fn create_image_view(
        context: Arc<VulkanContext>,
        texture: &Texture,
        format: vk::Format,
    ) -> ImageView {
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
                level_count: 1,
                base_array_layer: 0,
                layer_count: 6,
            })
            .build();
        ImageView::new(context, create_info)
    }

    fn create_sampler(context: Arc<VulkanContext>) -> Sampler {
        let sampler_info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .anisotropy_enable(true)
            .max_anisotropy(16.0)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(0.0)
            .build();
        Sampler::new(context, sampler_info)
    }
}
