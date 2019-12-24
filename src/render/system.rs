use crate::{
    core::ImageView,
    model::GltfAsset,
    render::{
        component::{MeshComponent, TransformComponent},
        ModelData, Renderer,
    },
    resource::{Buffer, DescriptorPool, Dimension, Sampler, Texture, TextureDescription},
    sync::{SynchronizationSet, SynchronizationSetConstants},
};
use ash::{version::DeviceV1_0, vk};
use gltf::image::Format;
use image::{ImageBuffer, Pixel, RgbImage};
use nalgebra_glm as glm;
use petgraph::{prelude::*, visit::Dfs};
use specs::prelude::*;
use std::mem;

#[derive(Debug, Clone, Copy)]
pub struct UniformBufferObject {
    pub model: glm::Mat4,
    pub view: glm::Mat4,
    pub projection: glm::Mat4,
}

impl UniformBufferObject {
    pub fn get_descriptor_set_layout_bindings() -> vk::DescriptorSetLayoutBinding {
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
