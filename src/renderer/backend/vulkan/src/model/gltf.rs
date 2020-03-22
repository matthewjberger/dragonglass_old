use crate::{
    core::VulkanContext,
    model::ModelBuffers,
    render::Renderer,
    resource::{Buffer, CommandPool, ImageView, Sampler, Texture, TextureDescription},
};
use ash::vk;
use nalgebra_glm as glm;
use petgraph::{
    graph::{Graph, NodeIndex},
    prelude::*,
    visit::Dfs,
};
use std::sync::Arc;

pub type NodeGraph = Graph<Node, ()>;

pub struct Node {
    pub local_transform: glm::Mat4,
    pub mesh: Option<Mesh>,
    pub index: usize,
}

pub struct Scene {
    pub node_graphs: Vec<NodeGraph>,
}

pub struct Mesh {
    pub primitives: Vec<Primitive>,
    pub mesh_id: usize,
}

pub struct Primitive {
    pub number_of_indices: u32,
    pub first_index: u32,
    pub material_index: Option<usize>,
}

pub struct GltfAsset {
    pub gltf: gltf::Document,
    pub textures: Vec<GltfTextureData>,
    pub scenes: Vec<Scene>,
    pub number_of_meshes: usize,
    pub buffers: ModelBuffers,
}

impl GltfAsset {
    pub fn new(renderer: &Renderer, asset_name: &str) -> GltfAsset {
        let (gltf, buffers, asset_textures) =
            gltf::import(&asset_name).expect("Couldn't import file!");

        let textures = asset_textures
            .iter()
            .map(|properties| GltfTextureData::new(&renderer, properties))
            .collect::<Vec<_>>();

        let (mut scenes, vertices, indices) = Self::prepare_scenes(&gltf, &buffers, &renderer);
        Self::update_ubo_indices(&mut scenes);

        let number_of_meshes = gltf.nodes().filter(|node| node.mesh().is_some()).count();

        let buffers =
            ModelBuffers::new(&renderer.transient_command_pool, &vertices, Some(&indices));
        GltfAsset {
            gltf,
            textures,
            scenes,
            number_of_meshes,
            buffers,
        }
    }

    fn determine_transform(node: &gltf::Node) -> glm::Mat4 {
        let transform: Vec<f32> = node
            .transform()
            .matrix()
            .iter()
            .flat_map(|array| array.iter())
            .cloned()
            .collect();
        glm::make_mat4(&transform.as_slice())
    }

    fn prepare_scenes(
        gltf: &gltf::Document,
        buffers: &[gltf::buffer::Data],
        renderer: &Renderer,
    ) -> (Vec<Scene>, Vec<f32>, Vec<u32>) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut scenes: Vec<Scene> = Vec::new();
        for scene in gltf.scenes() {
            let mut node_graphs: Vec<NodeGraph> = Vec::new();
            for node in scene.nodes() {
                let mut node_graph = NodeGraph::new();
                Self::visit_children(
                    &node,
                    &buffers,
                    &mut node_graph,
                    NodeIndex::new(0_usize),
                    &renderer,
                    &mut vertices,
                    &mut indices,
                );
                node_graphs.push(node_graph);
            }
            scenes.push(Scene { node_graphs });
        }
        (scenes, vertices, indices)
    }

    fn visit_children(
        node: &gltf::Node,
        buffers: &[gltf::buffer::Data],
        node_graph: &mut NodeGraph,
        parent_index: NodeIndex,
        renderer: &Renderer,
        vertices: &mut Vec<f32>,
        indices: &mut Vec<u32>,
    ) {
        let mesh = Self::load_mesh(node, buffers, vertices, indices);
        let node_info = Node {
            local_transform: Self::determine_transform(node),
            mesh,
            index: node.index(),
        };

        let node_index = node_graph.add_node(node_info);
        if parent_index != node_index {
            node_graph.add_edge(parent_index, node_index, ());
        }

        for child in node.children() {
            Self::visit_children(
                &child, buffers, node_graph, node_index, renderer, vertices, indices,
            );
        }
    }

    fn load_mesh(
        node: &gltf::Node,
        buffers: &[gltf::buffer::Data],
        vertices: &mut Vec<f32>,
        indices: &mut Vec<u32>,
    ) -> Option<Mesh> {
        if let Some(mesh) = node.mesh() {
            let mut all_mesh_primitives = Vec::new();
            for primitive in mesh.primitives() {
                // Position (3), Normal (3), TexCoords_0 (2)
                let stride = 8 * std::mem::size_of::<f32>();
                let vertex_list_size = vertices.len() * std::mem::size_of::<u32>();
                let vertex_count = (vertex_list_size / stride) as u32;

                // Start reading primitive data
                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

                let positions = reader
                    .read_positions()
                    .expect("Failed to read any vertex positions from the model. Vertex positions are required.")
                    .map(glm::Vec3::from)
                    .collect::<Vec<_>>();

                let normals = reader
                    .read_normals()
                    .map_or(vec![glm::vec3(0.0, 0.0, 0.0); positions.len()], |normals| {
                        normals.map(glm::Vec3::from).collect::<Vec<_>>()
                    });

                let convert_coords =
                    |coords: gltf::mesh::util::ReadTexCoords<'_>| -> Vec<glm::Vec2> {
                        coords.into_f32().map(glm::Vec2::from).collect::<Vec<_>>()
                    };
                let tex_coords_0 = reader
                    .read_tex_coords(0)
                    .map_or(vec![glm::vec2(0.0, 0.0); positions.len()], convert_coords);

                // TODO: Add checks to see if normals and tex_coords are even available
                for ((position, normal), tex_coord_0) in positions
                    .iter()
                    .zip(normals.iter())
                    .zip(tex_coords_0.iter())
                {
                    vertices.extend_from_slice(position.as_slice());
                    vertices.extend_from_slice(normal.as_slice());
                    vertices.extend_from_slice(tex_coord_0.as_slice());
                }

                let first_index = indices.len() as u32;

                let primitive_indices = reader
                    .read_indices()
                    .map(|read_indices| {
                        read_indices
                            .into_u32()
                            .map(|x| x + vertex_count)
                            .collect::<Vec<_>>()
                    })
                    .expect("Failed to read indices!");
                indices.extend_from_slice(&primitive_indices);

                let number_of_indices = primitive_indices.len() as u32;

                all_mesh_primitives.push(Primitive {
                    first_index,
                    number_of_indices,
                    material_index: primitive.material().index(),
                });
            }

            Some(Mesh {
                primitives: all_mesh_primitives,
                mesh_id: 0,
            })
        } else {
            None
        }
    }

    fn update_ubo_indices(scenes: &mut Vec<Scene>) {
        let mut indices = Vec::new();
        for (scene_index, scene) in scenes.iter().enumerate() {
            for (graph_index, graph) in scene.node_graphs.iter().enumerate() {
                let mut dfs = Dfs::new(&graph, NodeIndex::new(0));
                while let Some(node_index) = dfs.next(&graph) {
                    if graph[node_index].mesh.is_some() {
                        indices.push((scene_index, graph_index, node_index));
                    }
                }
            }
        }

        for (mesh_id, (scene_index, graph_index, node_index)) in indices.into_iter().enumerate() {
            scenes[scene_index].node_graphs[graph_index][node_index]
                .mesh
                .as_mut()
                .expect("Failed to get mesh!")
                .mesh_id = mesh_id;
        }
    }

    pub fn path_between_nodes(
        starting_node_index: NodeIndex,
        node_index: NodeIndex,
        graph: &NodeGraph,
    ) -> Vec<NodeIndex> {
        let mut indices = Vec::new();
        let mut dfs = Dfs::new(&graph, starting_node_index);
        while let Some(current_node_index) = dfs.next(&graph) {
            let mut incoming_walker = graph
                .neighbors_directed(current_node_index, Incoming)
                .detach();
            let mut outgoing_walker = graph
                .neighbors_directed(current_node_index, Outgoing)
                .detach();

            if let Some(parent) = incoming_walker.next_node(&graph) {
                while let Some(last_index) = indices.last() {
                    if *last_index == parent {
                        break;
                    }
                    // Discard indices for transforms that are no longer needed
                    indices.pop();
                }
            }

            indices.push(current_node_index);

            if node_index == current_node_index {
                break;
            }

            // If the node has no children, don't store the index
            if outgoing_walker.next(&graph).is_none() {
                indices.pop();
            }
        }
        indices
    }

    pub fn calculate_global_transform(node_index: NodeIndex, graph: &NodeGraph) -> glm::Mat4 {
        let indices = Self::path_between_nodes(NodeIndex::new(0), node_index, graph);
        indices
            .iter()
            .fold(glm::Mat4::identity(), |transform, index| {
                // TODO: Add the animation transform
                transform * graph[*index].local_transform //* graph[*index].animation_transform.matrix()
            })
    }

    pub fn walk<F>(&self, action: F)
    where
        F: Fn(NodeIndex, &NodeGraph),
    {
        for scene in self.scenes.iter() {
            for graph in scene.node_graphs.iter() {
                let mut dfs = Dfs::new(&graph, NodeIndex::new(0));
                while let Some(node_index) = dfs.next(&graph) {
                    action(node_index, &graph);
                }
            }
        }
    }

    pub fn create_vertex_attributes() -> [vk::VertexInputAttributeDescription; 3] {
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
            .offset((3 * std::mem::size_of::<f32>()) as _)
            .build();

        let tex_coord_description = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(2)
            .format(vk::Format::R32G32_SFLOAT)
            .offset((6 * std::mem::size_of::<f32>()) as _)
            .build();

        [
            position_description,
            normal_description,
            tex_coord_description,
        ]
    }

    pub fn create_vertex_input_descriptions() -> [vk::VertexInputBindingDescription; 1] {
        let vertex_input_binding_description = vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride((8 * std::mem::size_of::<f32>()) as _)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build();
        [vertex_input_binding_description]
    }
}

pub struct GltfTextureData {
    pub texture: Texture,
    pub view: ImageView,
    pub sampler: Sampler,
}

impl GltfTextureData {
    pub fn new(renderer: &Renderer, image_data: &gltf::image::Data) -> Self {
        let description = TextureDescription::from_gltf(&image_data);

        let texture = Self::create_texture(renderer.context.clone(), &description);

        Self::upload_texture_data(
            renderer.context.clone(),
            &renderer.command_pool,
            &texture,
            &description,
        );

        let view = Self::create_image_view(renderer.context.clone(), &texture, &description);

        let sampler = Self::create_sampler(renderer.context.clone(), description.mip_levels);

        Self {
            texture,
            view,
            sampler,
        }
    }

    pub fn upload_texture_data(
        context: Arc<VulkanContext>,
        command_pool: &CommandPool,
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
                layer_count: 1,
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
            context.clone(),
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
                level_count: description.mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .build();
        let barriers = [barrier];

        command_pool.transition_image_layout(
            &barriers,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
        );

        command_pool.copy_buffer_to_image(
            context.graphics_queue(),
            buffer.buffer(),
            texture.image(),
            &regions,
        );

        texture.generate_mipmaps(&command_pool, &description);
    }

    fn create_texture(context: Arc<VulkanContext>, description: &TextureDescription) -> Texture {
        let image_create_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width: description.width,
                height: description.height,
                depth: 1,
            })
            .mip_levels(description.mip_levels)
            .array_layers(1)
            .format(description.format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(
                vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::SAMPLED,
            )
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
        description: &TextureDescription,
    ) -> ImageView {
        let create_info = vk::ImageViewCreateInfo::builder()
            .image(texture.image())
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(description.format)
            .components(vk::ComponentMapping {
                r: vk::ComponentSwizzle::IDENTITY,
                g: vk::ComponentSwizzle::IDENTITY,
                b: vk::ComponentSwizzle::IDENTITY,
                a: vk::ComponentSwizzle::IDENTITY,
            })
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: description.mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            })
            .build();
        ImageView::new(context, create_info)
    }

    fn create_sampler(context: Arc<VulkanContext>, mip_levels: u32) -> Sampler {
        let sampler_info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(true)
            .max_anisotropy(16.0)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(mip_levels as _)
            .build();
        Sampler::new(context, sampler_info)
    }
}
