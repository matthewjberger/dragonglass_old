use crate::{
    core::ImageView,
    render::{
        component::GltfAssetComponent, system::UniformBufferObject, Mesh, MeshLocation, Primitive,
        Renderer, VulkanGltfAsset, VulkanGltfTexture,
    },
    resource::{Buffer, DescriptorPool, Dimension, Sampler, Texture, TextureDescription},
};
use ash::{version::DeviceV1_0, vk};
use dragonglass_model_gltf::GltfAsset;
use gltf::image::Format;
use image::{ImageBuffer, Pixel, RgbImage};
use petgraph::{prelude::*, visit::Dfs};
use specs::prelude::*;
use std::mem;

// TODO: Move this somewhere more general
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

pub struct PrepareRendererSystem;

impl<'a> System<'a> for PrepareRendererSystem {
    type SystemData = (
        WriteExpect<'a, Renderer>,
        ReadStorage<'a, GltfAssetComponent>,
    );

    fn run(&mut self, data: Self::SystemData) {
        let (mut renderer, assets) = data;
        let renderer = &mut renderer;

        // TODO: Make a command in the renderer to reset resources

        // TODO: Batch assets into a large vertex buffer
        for asset in assets.join() {
            let asset = GltfAsset::from_file(&asset.asset_name);
            let vulkan_gltf_asset = Self::load_asset(renderer, asset);
            renderer.assets.push(vulkan_gltf_asset);
        }

        Self::create_render_passes(renderer);
    }
}

impl PrepareRendererSystem {
    fn load_asset(renderer: &mut Renderer, asset: GltfAsset) -> VulkanGltfAsset {
        let uniform_buffer_size = mem::size_of::<UniformBufferObject>() as vk::DeviceSize;
        let number_of_swapchain_images = renderer.swapchain.images().len();

        let descriptor_pool = Self::setup_descriptor_pool(renderer, &asset);

        let mut asset_vertices: Vec<f32> = Vec::new();
        let mut asset_indices: Vec<u32> = Vec::new();
        let mut meshes = Vec::new();
        for (scene_index, scene) in asset.scenes.iter().enumerate() {
            for (graph_index, graph) in scene.node_graphs.iter().enumerate() {
                // Start at the root of the node graph
                let mut dfs = Dfs::new(&graph, NodeIndex::new(0));
                while let Some(node_index) = dfs.next(&graph) {
                    if let Some(mesh) = graph[node_index].mesh.as_ref() {
                        let mut primitives = Vec::new();
                        let mut first_index: usize = 0;
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
                            renderer.descriptor_set_layout.layout(),
                            number_of_swapchain_images as _,
                        );

                        for primitive_info in mesh.primitives.iter() {
                            let primitive_vertices = primitive_info.vertex_set.pack_vertices();
                            let primitive_indices = &primitive_info.indices;

                            asset_vertices.extend(primitive_vertices.iter());
                            asset_indices.extend(primitive_indices.iter());

                            // TODO: Add calculated primitive transform and
                            // change draw call to use dynamic ubos
                            let primitive = Primitive {
                                number_of_indices: asset_indices.len() as _,
                                material_index: primitive_info.material_index,
                                first_index: first_index as u32,
                            };

                            primitives.push(primitive);
                            first_index += primitive_indices.len();
                        }

                        let location = MeshLocation {
                            scene_index,
                            graph_index,
                            node_index,
                        };

                        let mesh = Mesh {
                            primitives,
                            uniform_buffers,
                            descriptor_sets,
                            location,
                        };

                        meshes.push(mesh);
                    }
                }
            }
        }

        let vertex_buffer = renderer.transient_command_pool.create_device_local_buffer(
            renderer.graphics_queue,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            &asset_vertices,
        );

        let index_buffer = renderer.transient_command_pool.create_device_local_buffer(
            renderer.graphics_queue,
            vk::BufferUsageFlags::INDEX_BUFFER,
            &asset_indices,
        );

        let textures = Self::load_textures(renderer, &asset);

        let vulkan_gltf_asset = VulkanGltfAsset {
            asset,
            vertex_buffer,
            index_buffer,
            textures,
            descriptor_pool,
            meshes,
        };

        Self::update_model_descriptor_sets(renderer, &vulkan_gltf_asset);

        vulkan_gltf_asset
    }

    fn setup_descriptor_pool(renderer: &mut Renderer, asset: &GltfAsset) -> DescriptorPool {
        let number_of_meshes = asset.number_of_meshes();
        let number_of_materials = asset.textures.len() as u32;
        let number_of_swapchain_images = renderer.swapchain.images().len() as u32;
        let number_of_samplers = number_of_materials * number_of_swapchain_images;

        let ubo_pool_size = (4 + number_of_meshes) * number_of_swapchain_images;
        let sampler_pool_size = number_of_samplers * number_of_swapchain_images;
        let max_number_of_pools =
            (2 + number_of_materials + number_of_meshes) * number_of_swapchain_images;

        DescriptorPool::new(
            renderer.context.clone(),
            ubo_pool_size,
            sampler_pool_size,
            max_number_of_pools,
        )
    }

    fn load_textures(renderer: &mut Renderer, asset: &GltfAsset) -> Vec<VulkanGltfTexture> {
        let mut textures = Vec::new();
        for texture_properties in asset.textures.iter() {
            let mut texture_format = convert_to_vulkan_format(texture_properties.format);

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
            let sampler = Sampler::new(renderer.context.clone());

            let vulkan_gltf_texture = VulkanGltfTexture {
                texture,
                view,
                sampler,
            };

            textures.push(vulkan_gltf_texture);
        }
        textures
    }

    fn update_model_descriptor_sets(renderer: &mut Renderer, asset: &VulkanGltfAsset) {
        asset.meshes.iter().for_each(|mesh| {
            mesh.descriptor_sets
                .iter()
                .zip(mesh.uniform_buffers.iter())
                .for_each(|(descriptor_set, buffer)| {
                    for primitive in mesh.primitives.iter() {
                        let uniform_buffer_size =
                            mem::size_of::<UniformBufferObject>() as vk::DeviceSize;
                        let buffer_info = vk::DescriptorBufferInfo::builder()
                            .buffer(buffer.buffer())
                            .offset(0)
                            .range(uniform_buffer_size)
                            .build();
                        let buffer_infos = [buffer_info];

                        let ubo_descriptor_write = vk::WriteDescriptorSet::builder()
                            .dst_set(*descriptor_set)
                            .dst_binding(0)
                            .dst_array_element(0)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                            .buffer_info(&buffer_infos)
                            .build();

                        // TODO: Make material optional
                        let material_index = primitive
                            .material_index
                            .expect("Failed to get material index!");
                        let texture_view = asset.textures[material_index].view.view();
                        let texture_sampler = asset.textures[material_index].sampler.sampler();

                        let image_info = vk::DescriptorImageInfo::builder()
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                            .image_view(texture_view)
                            .sampler(texture_sampler)
                            .build();
                        let image_infos = [image_info];

                        let sampler_descriptor_write = vk::WriteDescriptorSet::builder()
                            .dst_set(*descriptor_set)
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
                });
        })
    }

    fn create_render_passes(renderer: &mut Renderer) {
        // Allocate one command buffer per swapchain image
        let number_of_framebuffers = renderer.framebuffers.len();
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
                renderer.create_render_pass(framebuffer, *command_buffer, |command_buffer|
                    // TODO: Batch models by which shader should be used to render them
                    // TODO: Gather assets from VulkanGltfAsset components added to entities?? May not be the best way
                    unsafe {
                        Self::draw_asset(command_buffer, renderer, index);
                    });
            });
    }

    unsafe fn draw_asset(
        command_buffer: vk::CommandBuffer,
        renderer: &Renderer,
        command_buffer_index: usize,
    ) {
        renderer.assets.iter().for_each(|asset| {
            let offsets = [0];
            let vertex_buffers = [asset.vertex_buffer.buffer()];
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
                    asset.index_buffer.buffer(),
                    0,
                    vk::IndexType::UINT32,
                );

            for mesh in asset.meshes.iter() {
                renderer
                    .context
                    .logical_device()
                    .logical_device()
                    .cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        renderer.pipeline.layout(),
                        0,
                        &[mesh.descriptor_sets[command_buffer_index]],
                        &[],
                    );

                for primitive in mesh.primitives.iter() {
                    renderer
                        .context
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
}
