use crate::{
    core::{SwapchainProperties, VulkanContext},
    resource::{PipelineLayout, Shader},
};
use ash::{version::DeviceV1_0, vk};
use std::{ffi::CString, mem, sync::Arc};

pub struct GraphicsPipeline {
    pipeline: vk::Pipeline,
    pipeline_layout: PipelineLayout,
    context: Arc<VulkanContext>,
}

impl GraphicsPipeline {
    // TODO: Refactor this to use less parameters
    // TODO: Refactor fixed function steps to separate functions
    // TODO: Take create info as a parameter and construct the pipeline somewhere else
    pub fn new(
        context: Arc<VulkanContext>,
        swapchain_properties: &SwapchainProperties,
        render_pass: vk::RenderPass,
        descriptor_set_layout: vk::DescriptorSetLayout,
    ) -> Self {
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
        let descriptor_set_layouts = [descriptor_set_layout];
        let pipeline_layout = PipelineLayout::new(context.clone(), &descriptor_set_layouts);

        // Create the pipeline info
        let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
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

        let pipeline_info_arr = [pipeline_info];

        // Create the pipeline
        let pipeline = unsafe {
            context
                .logical_device()
                .logical_device()
                .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_info_arr, None)
                .expect("Failed to create graphics pipelines!")[0]
        };

        GraphicsPipeline {
            pipeline,
            pipeline_layout,
            context,
        }
    }

    pub fn pipeline(&self) -> vk::Pipeline {
        self.pipeline
    }

    pub fn layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout.layout()
    }
}

impl Drop for GraphicsPipeline {
    fn drop(&mut self) {
        unsafe {
            self.context
                .logical_device()
                .logical_device()
                .destroy_pipeline(self.pipeline, None);
        }
    }
}
