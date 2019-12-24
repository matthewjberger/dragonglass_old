use crate::{
    core::{ImageView, Swapchain, VulkanContext},
    model::GltfAsset,
    render::{
        component::{MeshComponent, TransformComponent},
        Framebuffer, GraphicsPipeline, RenderPass,
    },
    resource::{
        Buffer, CommandPool, DescriptorPool, DescriptorSetLayout, Dimension, Sampler, Texture,
        TextureDescription,
    },
    sync::{SynchronizationSet, SynchronizationSetConstants},
};
use ash::{
    version::{DeviceV1_0, InstanceV1_0},
    vk,
};
use gltf::image::Format;
use image::{ImageBuffer, Pixel, RgbImage};
use nalgebra_glm as glm;
use petgraph::{prelude::*, visit::Dfs};
use specs::prelude::*;
use std::{mem, sync::Arc};

// TODO: rename this
pub struct ModelData {
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub number_of_indices: u32,
    pub uniform_buffers: Vec<Buffer>,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub material_index: Option<usize>,
    pub asset_index: usize, // The asset this primitive data belongs to
}

#[derive(Debug, Clone, Copy)]
pub struct UniformBufferObject {
    pub model: glm::Mat4,
    pub view: glm::Mat4,
    pub projection: glm::Mat4,
}

impl UniformBufferObject {
    fn get_descriptor_set_layout_bindings() -> vk::DescriptorSetLayoutBinding {
        vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .build()
    }
}

pub struct TransformationSystem;

impl<'a> System<'a> for TransformationSystem {
    type SystemData = WriteStorage<'a, TransformComponent>;

    fn run(&mut self, data: Self::SystemData) {
        let mut transforms = data;
        for transform in (&mut transforms).join() {
            transform.rotate = glm::rotate(
                &transform.rotate,
                0.1_f32.to_radians(),
                &glm::vec3(0.0, 1.0, 0.0),
            );
        }
    }
}

pub struct PrepareRendererSystem;

impl<'a> System<'a> for PrepareRendererSystem {
    type SystemData = (WriteExpect<'a, Renderer>, ReadStorage<'a, MeshComponent>);

    fn run(&mut self, data: Self::SystemData) {
        let (mut renderer, meshes) = data;
        let renderer = &mut renderer;

        // TODO: Make a command in the renderer to reset resources

        let number_of_swapchain_images = renderer.swapchain.images().len();
        let number_of_meshes = meshes.join().count();

        // TODO: This should have n_meshes * n_primitives * n_swapchain_images
        // TODO: Push this after all model data has been created
        let descriptor_pool = DescriptorPool::new(
            renderer.context.clone(),
            (number_of_meshes * 10 * number_of_swapchain_images) as _,
        );
        renderer.descriptor_pools.push(descriptor_pool);

        // TODO: Push this after all model data has been created
        let texture_image_sampler = Sampler::new(renderer.context.clone());
        renderer.texture_samplers.push(texture_image_sampler);

        // TODO: Batch assets into a large vertex buffer
        let uniform_buffer_size = mem::size_of::<UniformBufferObject>() as vk::DeviceSize;
        for mesh in meshes.join() {
            // TODO: Cache assets
            let asset = GltfAsset::from_file(&mesh.mesh_name);

            let mut textures = Vec::new();
            let mut texture_views = Vec::new();
            for texture_properties in asset.textures.iter() {
                let mut texture_format = match texture_properties.format {
                    Format::R8 => vk::Format::R8_UNORM,
                    Format::R8G8 => vk::Format::R8G8_UNORM,
                    Format::R8G8B8A8 => vk::Format::R8G8B8A8_UNORM,
                    Format::B8G8R8A8 => vk::Format::B8G8R8A8_UNORM,
                    // 24-bit formats will have an alpha channel added
                    // to make them 32-bit
                    Format::R8G8B8 => vk::Format::R8G8B8_UNORM,
                    Format::B8G8R8 => vk::Format::B8G8R8_UNORM,
                };

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
                let texture_view = ImageView::new(renderer.context.clone(), create_info);

                textures.push(texture);
                texture_views.push(texture_view);
            }
            renderer.textures.push(textures);
            renderer.texture_views.push(texture_views);

            let mut scene_vertices: Vec<f32> = Vec::new();
            let mut scene_indices: Vec<u32> = Vec::new();
            for scene in asset.scenes.iter() {
                for graph in scene.node_graphs.iter() {
                    // Start at the root of the node graph
                    let mut dfs = Dfs::new(&graph, NodeIndex::new(0));

                    let mut asset_index = 0;

                    // Walk the scene graph
                    while let Some(node_index) = dfs.next(&graph) {
                        // If there is a mesh, handle its primitives
                        if let Some(mesh) = graph[node_index].mesh.as_ref() {
                            for primitive_info in mesh.primitives.iter() {
                                scene_vertices
                                    .extend(primitive_info.vertex_set.pack_vertices().iter());
                                scene_indices.extend(primitive_info.indices.iter());

                                let vertex_buffer =
                                    renderer.transient_command_pool.create_device_local_buffer(
                                        renderer.graphics_queue,
                                        vk::BufferUsageFlags::VERTEX_BUFFER,
                                        &scene_vertices,
                                    );

                                let index_buffer =
                                    renderer.transient_command_pool.create_device_local_buffer(
                                        renderer.graphics_queue,
                                        vk::BufferUsageFlags::INDEX_BUFFER,
                                        &scene_indices,
                                    );

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

                                let descriptor_sets = renderer.descriptor_pools[0]
                                    .allocate_descriptor_sets(
                                        renderer.descriptor_set_layout.layout(),
                                        number_of_swapchain_images as _,
                                    );

                                // TODO: Add calculated primitive transform and
                                // change draw call to use dynamic ubos
                                let model_data = ModelData {
                                    vertex_buffer,
                                    index_buffer,
                                    number_of_indices: scene_indices.len() as _,
                                    uniform_buffers,
                                    descriptor_sets,
                                    material_index: primitive_info.material_index,
                                    asset_index,
                                };

                                model_data
                                    .descriptor_sets
                                    .iter()
                                    .zip(model_data.uniform_buffers.iter())
                                    .for_each(|(set, buffer)| {
                                        let buffer_info = vk::DescriptorBufferInfo::builder()
                                            .buffer(buffer.buffer())
                                            .offset(0)
                                            .range(uniform_buffer_size)
                                            .build();
                                        let buffer_infos = [buffer_info];

                                        let ubo_descriptor_write =
                                            vk::WriteDescriptorSet::builder()
                                                .dst_set(*set)
                                                .dst_binding(0)
                                                .dst_array_element(0)
                                                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                                                .buffer_info(&buffer_infos)
                                                .build();

                                        // TODO: Make material optional
                                        let texture_view = renderer.texture_views
                                            [model_data.asset_index]
                                            [model_data.material_index.unwrap()]
                                        .view();
                                        let image_info = vk::DescriptorImageInfo::builder()
                                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                            .image_view(texture_view)
                                            .sampler(renderer.texture_samplers[0].sampler())
                                            .build();
                                        let image_infos = [image_info];

                                        let sampler_descriptor_write =
                                            vk::WriteDescriptorSet::builder()
                                                .dst_set(*set)
                                                .dst_binding(1)
                                                .dst_array_element(0)
                                                .descriptor_type(
                                                    vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                                                )
                                                .image_info(&image_infos)
                                                .build();

                                        let descriptor_writes =
                                            [ubo_descriptor_write, sampler_descriptor_write];

                                        unsafe {
                                            renderer
                                                .context
                                                .logical_device()
                                                .logical_device()
                                                .update_descriptor_sets(&descriptor_writes, &[])
                                        }
                                    });

                                renderer.models.push(model_data);
                            }
                            asset_index += 1;
                        }
                    }
                }
            }
        }

        let number_of_framebuffers = renderer.framebuffers.len();
        // Allocate one command buffer per swapchain image
        renderer
            .command_pool
            .allocate_command_buffers(number_of_framebuffers as _);

        // Create a single render pass that will draw each mesh
        renderer
            .command_pool
            .command_buffers()
            .iter()
            .enumerate()
            .for_each(|(index, buffer)| {
                let command_buffer = buffer;
                let framebuffer = renderer.framebuffers[index].framebuffer();

                renderer.create_render_pass(
                    framebuffer,
                    *command_buffer,
                    |command_buffer| unsafe {
                        // TODO: Batch models by which shader should be used to render them
                        renderer.models.iter().for_each(|model_data| {
                            // Bind vertex buffer
                            let offsets = [0];
                            let vertex_buffers = [model_data.vertex_buffer.buffer()];
                            renderer
                                .context
                                .logical_device()
                                .logical_device()
                                .cmd_bind_vertex_buffers(
                                    command_buffer,
                                    0,
                                    &vertex_buffers,
                                    &offsets,
                                );

                            // Bind index buffer
                            renderer
                                .context
                                .logical_device()
                                .logical_device()
                                .cmd_bind_index_buffer(
                                    command_buffer,
                                    model_data.index_buffer.buffer(),
                                    0,
                                    vk::IndexType::UINT32,
                                );

                            // Bind descriptor sets
                            renderer
                                .context
                                .logical_device()
                                .logical_device()
                                .cmd_bind_descriptor_sets(
                                    command_buffer,
                                    vk::PipelineBindPoint::GRAPHICS,
                                    renderer.pipeline.layout(),
                                    0,
                                    &[model_data.descriptor_sets[index]],
                                    &[],
                                );

                            // Draw
                            renderer
                                .context
                                .logical_device()
                                .logical_device()
                                .cmd_draw_indexed(
                                    command_buffer,
                                    model_data.number_of_indices,
                                    1,
                                    0,
                                    0,
                                    0,
                                );
                        });
                    },
                );
            });
    }
}

pub struct RenderSystem;

impl<'a> System<'a> for RenderSystem {
    type SystemData = (
        WriteExpect<'a, Renderer>,
        ReadStorage<'a, TransformComponent>,
    );

    fn run(&mut self, data: Self::SystemData) {
        let (mut renderer, transform) = data;
        let renderer = &mut renderer;

        let current_frame_synchronization = renderer
            .synchronization_set
            .current_frame_synchronization(renderer.current_frame);

        renderer
            .context
            .logical_device()
            .wait_for_fence(&current_frame_synchronization);

        // Acquire the next image from the swapchain
        let image_index_result = renderer.swapchain.acquire_next_image(
            current_frame_synchronization.image_available(),
            vk::Fence::null(),
        );

        let image_index = match image_index_result {
            Ok((image_index, _)) => image_index,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                // TODO: Recreate the swapchain
                return;
            }
            Err(error) => panic!("Error while acquiring next image. Cause: {}", error),
        };
        let image_indices = [image_index];

        renderer
            .context
            .logical_device()
            .reset_fence(&current_frame_synchronization);

        let projection = glm::perspective_zo(
            renderer.swapchain.properties().aspect_ratio(),
            90_f32.to_radians(),
            0.1_f32,
            1000_f32,
        );

        let view = glm::look_at(
            &glm::vec3(300.0, 300.0, 300.0),
            &glm::vec3(0.0, 0.0, 0.0),
            &glm::vec3(0.0, 1.0, 0.0),
        );

        for (index, transform) in (&transform).join().enumerate() {
            let ubo = UniformBufferObject {
                model: transform.translate * transform.rotate * transform.scale,
                view,
                projection,
            };

            let ubos = [ubo];
            let buffer = &renderer.models[index].uniform_buffers[image_index as usize];
            buffer.upload_to_buffer(&ubos, 0);
        }

        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        renderer.command_pool.submit_command_buffer(
            image_index as usize,
            renderer.graphics_queue,
            &wait_stages,
            &current_frame_synchronization,
        );

        let swapchain_presentation_result = renderer.swapchain.present_rendered_image(
            &current_frame_synchronization,
            &image_indices,
            renderer.present_queue,
        );

        match swapchain_presentation_result {
            Ok(is_suboptimal) if is_suboptimal => {
                // TODO: Recreate the swapchain
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                // TODO: Recreate the swapchain
            }
            Err(error) => panic!("Failed to present queue. Cause: {}", error),
            _ => {}
        }

        // TODO: Recreate the swapchain if resize was requested

        renderer.current_frame +=
            (1 + renderer.current_frame) % SynchronizationSet::MAX_FRAMES_IN_FLIGHT as usize;
    }
}

pub struct Renderer {
    context: Arc<VulkanContext>,
    pub command_pool: CommandPool,
    pub descriptor_pools: Vec<DescriptorPool>,
    pub descriptor_set_layout: DescriptorSetLayout,
    pub framebuffers: Vec<Framebuffer>,
    pub graphics_queue: vk::Queue,
    pub models: Vec<ModelData>,
    pub pipeline: GraphicsPipeline,
    pub present_queue: vk::Queue,
    pub render_pass: RenderPass,
    pub swapchain: Swapchain,
    pub transient_command_pool: CommandPool,
    pub depth_texture: Texture,
    pub depth_texture_view: ImageView,
    pub textures: Vec<Vec<Texture>>,        // TODO: Make this a cache
    pub texture_views: Vec<Vec<ImageView>>, // TODO: Make this a cache
    pub texture_samplers: Vec<Sampler>,     // TODO: Make this a cache
    pub synchronization_set: SynchronizationSet,
    pub current_frame: usize,
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
                .unwrap()
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

        let depth_format = Self::determine_depth_format(
            context.clone(),
            vk::ImageTiling::OPTIMAL,
            vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
        );

        let logical_size = window.get_inner_size().unwrap();
        let dimensions = [logical_size.width as u32, logical_size.height as u32];
        let swapchain = Swapchain::new(context.clone(), dimensions);
        let render_pass = RenderPass::new(context.clone(), swapchain.properties(), depth_format);

        let ubo_binding = UniformBufferObject::get_descriptor_set_layout_bindings();
        let sampler_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build();
        let bindings = [ubo_binding, sampler_binding];
        let descriptor_set_layout = DescriptorSetLayout::new(context.clone(), &bindings);

        let pipeline = GraphicsPipeline::new(
            context.clone(),
            swapchain.properties(),
            render_pass.render_pass(),
            descriptor_set_layout.layout(),
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
                Framebuffer::new(
                    context.clone(),
                    swapchain.properties(),
                    render_pass.render_pass(),
                    &attachments,
                )
            })
            .collect::<Vec<_>>();

        Renderer {
            command_pool,
            context,
            descriptor_pools: Vec::new(), // TODO: maybe make this a map and have a main descriptor pool
            descriptor_set_layout,
            framebuffers,
            graphics_queue,
            models: Vec::new(),
            pipeline,
            present_queue,
            render_pass,
            swapchain,
            synchronization_set,
            depth_texture,
            depth_texture_view,
            textures: Vec::new(),
            texture_views: Vec::new(),
            texture_samplers: Vec::new(),
            transient_command_pool,
            current_frame: 0,
        }
    }

    fn create_render_pass<F>(
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
                .unwrap()
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
                .unwrap();
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
                .unwrap()
        };
    }

    // TODO: Move this to a more specific component
    pub fn determine_depth_format(
        context: Arc<VulkanContext>,
        tiling: vk::ImageTiling,
        features: vk::FormatFeatureFlags,
    ) -> vk::Format {
        let candidates = vec![
            vk::Format::D32_SFLOAT,
            vk::Format::D32_SFLOAT_S8_UINT,
            vk::Format::D24_UNORM_S8_UINT,
        ];
        candidates
            .iter()
            .copied()
            .find(|candidate| {
                let properties = unsafe {
                    context.instance().get_physical_device_format_properties(
                        context.physical_device(),
                        *candidate,
                    )
                };

                let linear_tiling_feature_support = tiling == vk::ImageTiling::LINEAR
                    && properties.linear_tiling_features.contains(features);

                let optimal_tiling_feature_support = tiling == vk::ImageTiling::OPTIMAL
                    && properties.optimal_tiling_features.contains(features);

                linear_tiling_feature_support || optimal_tiling_feature_support
            })
            .expect("Failed to find a supported depth format")
    }
}
