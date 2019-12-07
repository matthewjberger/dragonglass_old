use crate::{core::SwapchainProperties, shader, vertex::Vertex, VulkanContext};
use ash::{version::DeviceV1_0, vk};
use std::{ffi::CString, sync::Arc};

pub struct GraphicsPipeline {
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    context: Arc<VulkanContext>,
}

impl GraphicsPipeline {
    // TODO: Refactor this to use less parameters
    // TODO: Breakout shader creation to seperate module
    pub fn new(
        context: Arc<VulkanContext>,
        swapchain_properties: &SwapchainProperties,
        render_pass: vk::RenderPass,
        descriptor_set_layout: vk::DescriptorSetLayout,
    ) -> Self {
        // Define the entry point for shaders
        let entry_point_name = &CString::new("main").unwrap();

        // Create the vertex shader module
        let vertex_shader_module =
            shader::create_shader_from_file("shaders/shader.vert.spv", context.logical_device());

        let vertex_shader_state_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_shader_module)
            .name(entry_point_name)
            .build();

        // Create the fragment shader module
        let fragment_shader_module =
            shader::create_shader_from_file("shaders/shader.frag.spv", context.logical_device());

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
            context
                .logical_device()
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
            context
                .logical_device()
                .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_info_arr, None)
                .unwrap()[0]
        };

        // Delete shader modules
        unsafe {
            context
                .logical_device()
                .destroy_shader_module(vertex_shader_module, None);
            context
                .logical_device()
                .destroy_shader_module(fragment_shader_module, None);
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

    pub fn pipeline_layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }
}

impl Drop for GraphicsPipeline {
    fn drop(&mut self) {
        unsafe {
            self.context
                .logical_device()
                .destroy_pipeline(self.pipeline, None);
            self.context
                .logical_device()
                .destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}
