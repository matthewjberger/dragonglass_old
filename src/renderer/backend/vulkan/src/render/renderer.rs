use crate::{
    core::VulkanContext,
    model::{gltf::GltfAsset, ModelBuffers},
    pipelines::{
        pbr::{PbrPipeline, PbrPipelineData, PbrRenderer},
        skybox::{SkyboxPipeline, SkyboxPipelineData, SkyboxRenderer, VERTICES},
    },
    render::{
        environment::{Brdflut, HdrCubemap, IrradianceMap, PrefilterMap},
        shader_compilation::compile_shaders,
        VulkanSwapchain,
    },
    resource::CommandPool,
    sync::SynchronizationSet,
};
use ash::{version::DeviceV1_0, vk};
use nalgebra_glm as glm;
use std::sync::Arc;

pub struct Renderer {
    pub context: Arc<VulkanContext>,
    vulkan_swapchain: Option<VulkanSwapchain>,
    pub synchronization_set: SynchronizationSet,
    pub current_frame: usize,
    pub command_pool: CommandPool,
    pub transient_command_pool: CommandPool,
    pub assets: Vec<GltfAsset>,
    pub pbr_pipeline: Option<PbrPipeline>,
    pub pbr_pipeline_data: Option<PbrPipelineData>,
    pub skybox_pipeline: Option<SkyboxPipeline>,
    pub skybox_pipeline_data: Option<SkyboxPipelineData>,
    pub cubemap: Option<HdrCubemap>,
    pub irradiance_map: Option<IrradianceMap>,
    pub prefilter_map: Option<PrefilterMap>,
    pub brdflut: Option<Brdflut>,
    pub can_reload: bool,
}

impl Renderer {
    pub fn new(window: &winit::Window) -> Self {
        let context =
            Arc::new(VulkanContext::new(&window).expect("Failed to create VulkanContext"));

        let synchronization_set =
            SynchronizationSet::new(context.clone()).expect("Failed to create sync objects");

        let command_pool = CommandPool::new(
            context.clone(),
            vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
        );

        let transient_command_pool =
            CommandPool::new(context.clone(), vk::CommandPoolCreateFlags::TRANSIENT);

        let logical_size = window
            .get_inner_size()
            .expect("Failed to get the window's inner size!");
        let dimensions = [logical_size.width as u32, logical_size.height as u32];

        let vulkan_swapchain = Some(VulkanSwapchain::new(
            context.clone(),
            dimensions,
            &command_pool,
        ));

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
            irradiance_map: None,
            prefilter_map: None,
            brdflut: None,
            can_reload: false,
        };

        renderer.pbr_pipeline = Some(PbrPipeline::new(&mut renderer));
        renderer.skybox_pipeline = Some(SkyboxPipeline::new(&mut renderer));
        renderer
    }

    pub fn vulkan_swapchain(&self) -> &VulkanSwapchain {
        self.vulkan_swapchain
            .as_ref()
            .expect("Failed to get vulkan swapchain!")
    }

    pub fn reload_pbr_pipeline(&mut self) {
        let shader_directory = "examples/assets/shaders";
        let shader_glob = shader_directory.to_owned() + "/**/shader*.glsl";
        if compile_shaders(&shader_glob).is_err() {
            println!("Failed to recompile shaders!");
        }

        self.pbr_pipeline = None;
        self.pbr_pipeline_data = None;

        let pbr_pipeline = PbrPipeline::new(self);

        let textures = self
            .assets
            .iter()
            .flat_map(|asset| &asset.textures)
            .collect::<Vec<_>>();

        let number_of_meshes = self.assets.iter().fold(0, |total_meshes, asset| {
            total_meshes + asset.number_of_meshes
        });

        let pbr_pipeline_data = PbrPipelineData::new(&self, number_of_meshes, &textures);
        self.pbr_pipeline = Some(pbr_pipeline);
        self.pbr_pipeline_data = Some(pbr_pipeline_data);

        self.record_command_buffers();
        println!("Reloaded pbr shaders.");
    }

    pub fn recreate_swapchain(&mut self, dimensions: glm::Vec2) {
        self.context.logical_device().wait_idle();

        self.vulkan_swapchain = None;
        let new_swapchain = VulkanSwapchain::new(
            self.context.clone(),
            [dimensions.x as _, dimensions.y as _],
            &self.command_pool,
        );
        self.vulkan_swapchain = Some(new_swapchain);

        let pbr_pipeline = PbrPipeline::new(self);
        let skybox_pipeline = SkyboxPipeline::new(self);

        self.pbr_pipeline = None;
        self.skybox_pipeline = None;

        self.pbr_pipeline = Some(pbr_pipeline);
        self.skybox_pipeline = Some(skybox_pipeline);

        self.record_command_buffers();
    }

    pub fn load_environment(&mut self) {
        let cube = ModelBuffers::new(&self.transient_command_pool, VERTICES, None);

        let cubemap_path = "examples/assets/skyboxes/walk_of_fame/walk_of_fame.hdr";
        let hdr_cubemap = HdrCubemap::new(
            self.context.clone(),
            &self.command_pool,
            &cube,
            &cubemap_path,
        );

        let brdflut = Brdflut::new(self.context.clone(), &self.transient_command_pool);
        self.brdflut = Some(brdflut);

        let irradiance_map = IrradianceMap::new(
            self.context.clone(),
            &self.transient_command_pool,
            &hdr_cubemap.cubemap,
            &cube,
        );
        self.irradiance_map = Some(irradiance_map);

        // Manual prefiltering
        let prefilter_map = PrefilterMap::new(
            self.context.clone(),
            &self.transient_command_pool,
            &hdr_cubemap.cubemap,
            &cube,
        );
        self.prefilter_map = Some(prefilter_map);

        self.cubemap = Some(hdr_cubemap);
    }

    pub fn load_assets(&mut self, asset_names: &[String]) {
        self.load_environment();

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

        self.pbr_pipeline_data = Some(PbrPipelineData::new(&self, number_of_meshes, &textures));

        let skybox_pipeline_data = SkyboxPipelineData::new(
            &self,
            &self
                .cubemap
                .as_ref()
                .expect("Failed to get cubemap!")
                .cubemap,
        );
        self.skybox_pipeline_data = Some(skybox_pipeline_data);
        self.assets = assets;
    }

    pub fn allocate_command_buffers(&mut self) {
        // Allocate one command buffer per swapchain image
        let number_of_framebuffers = self.vulkan_swapchain().framebuffers.len();
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
                let framebuffer = self.vulkan_swapchain().framebuffers[index].framebuffer();
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
            .render_pass(self.vulkan_swapchain().render_pass.render_pass())
            .framebuffer(framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.vulkan_swapchain().swapchain.properties().extent,
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
        let extent = self.vulkan_swapchain().swapchain.properties().extent;

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
}

impl Drop for Renderer {
    fn drop(&mut self) {
        self.context.logical_device().wait_idle();
    }
}
