use crate::{
    core::VulkanContext,
    model::gltf::GltfAsset,
    pipelines::gltf::{GltfPipeline, GltfPipelineData, PushConstantBlockMaterial},
    render::VulkanSwapchain,
    resource::CommandPool,
    sync::SynchronizationSet,
};
use ash::{version::DeviceV1_0, vk};
use nalgebra_glm as glm;
use petgraph::{graph::NodeIndex, visit::Dfs};
use std::{slice, sync::Arc};

pub struct Renderer {
    pub context: Arc<VulkanContext>,
    pub vulkan_swapchain: VulkanSwapchain,
    pub synchronization_set: SynchronizationSet,
    pub current_frame: usize,
    pub command_pool: CommandPool,
    pub transient_command_pool: CommandPool,
    pub pipeline_gltf: Option<GltfPipeline>,
    pub assets: Vec<GltfAsset>,
    pub gltf_pipeline_data: Option<GltfPipelineData>,
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
            pipeline_gltf: None,
            synchronization_set,
            current_frame: 0,
            vulkan_swapchain,
            command_pool,
            transient_command_pool,
            assets: Vec::new(),
            gltf_pipeline_data: None,
        };

        renderer.pipeline_gltf = Some(GltfPipeline::new(&mut renderer));
        renderer
    }

    #[allow(dead_code)]
    pub fn recreate_swapchain(&mut self, _: Option<[u32; 2]>) {
        log::debug!("Recreating swapchain");
        self.context.logical_device().wait_idle();
        // TODO: Implement swapchain recreation
    }

    pub fn load_assets(&mut self, asset_names: &[String]) {
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

        self.gltf_pipeline_data = Some(GltfPipelineData::new(&self, number_of_meshes, &textures));
        self.assets = assets;
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
                self.create_render_pass(framebuffer, command_buffer);
            });
    }

    pub fn create_render_pass(
        &self,
        framebuffer: vk::Framebuffer,
        command_buffer: vk::CommandBuffer,
    ) {
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

            self.context
                .logical_device()
                .logical_device()
                .cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_gltf.as_ref().unwrap().pipeline.pipeline(),
                );
        }

        self.update_viewport(command_buffer);

        self.assets
            .iter()
            .for_each(|asset| unsafe { self.draw_asset(&asset, command_buffer) });

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

    fn update_viewport(&self, command_buffer: vk::CommandBuffer) {
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
            self.context
                .logical_device()
                .logical_device()
                .cmd_set_viewport(command_buffer, 0, &viewports);

            self.context
                .logical_device()
                .logical_device()
                .cmd_set_scissor(command_buffer, 0, &scissors);
        }
    }

    // TODO: Move this to a seperate class or even the mod.rs file
    unsafe fn byte_slice_from<T: Sized>(data: &T) -> &[u8] {
        let data_ptr = (data as *const T) as *const u8;
        slice::from_raw_parts(data_ptr, std::mem::size_of::<T>())
    }

    unsafe fn draw_asset(&self, asset: &GltfAsset, command_buffer: vk::CommandBuffer) {
        let gltf_pipeline_data = self
            .gltf_pipeline_data
            .as_ref()
            .expect("Failed to get pbr asset!");
        let pipeline_layout = self.pipeline_gltf.as_ref().unwrap().pipeline.layout();
        let offsets = [0];
        let vertex_buffers = [asset.vertex_buffer.buffer()];
        self.context
            .logical_device()
            .logical_device()
            .cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets);

        self.context
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
                        self.context
                            .logical_device()
                            .logical_device()
                            .cmd_bind_descriptor_sets(
                                command_buffer,
                                vk::PipelineBindPoint::GRAPHICS,
                                pipeline_layout,
                                0,
                                &[gltf_pipeline_data.descriptor_set],
                                &[(mesh.mesh_id as u64 * gltf_pipeline_data.dynamic_alignment)
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

                            self.context
                                .logical_device()
                                .logical_device()
                                .cmd_push_constants(
                                    command_buffer,
                                    pipeline_layout,
                                    vk::ShaderStageFlags::ALL_GRAPHICS,
                                    0,
                                    Self::byte_slice_from(&material),
                                );

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
                }
            }
        }
    }
}
