use crate::{
    render::{texture_bundle::GltfTextureBundle, Renderer},
    resource::{Buffer, DescriptorPool},
};
use ash::{
    version::{DeviceV1_0, InstanceV1_0},
    vk,
};
use nalgebra_glm as glm;
use petgraph::{
    graph::{Graph, NodeIndex},
    prelude::*,
    visit::Dfs,
};
use std::mem;

pub type NodeGraph = Graph<Node, ()>;

#[derive(Debug, Clone, Copy)]
pub struct UniformBufferObject {
    pub view: glm::Mat4,
    pub projection: glm::Mat4,
}

#[derive(Debug, Clone, Copy)]
pub struct DynamicUniformBufferObject {
    pub model: glm::Mat4,
}

pub struct Node {
    pub local_transform: glm::Mat4,
    pub mesh: Option<Mesh>,
    pub index: usize,
}

pub struct Scene {
    pub node_graphs: Vec<NodeGraph>,
}

pub struct Mesh {
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub primitives: Vec<Primitive>,
    pub ubo_index: usize,
}

pub struct Primitive {
    pub number_of_indices: u32,
    pub first_index: u32,
    pub material_index: Option<usize>,
}

pub struct VulkanGltfAsset {
    pub gltf: gltf::Document,
    pub textures: Vec<GltfTextureBundle>,
    pub scenes: Vec<Scene>,
    pub descriptor_pool: DescriptorPool,
    pub uniform_buffer: Buffer,
    pub dynamic_uniform_buffer: Buffer,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub dynamic_alignment: u64,
}

impl VulkanGltfAsset {
    pub fn new(
        renderer: &Renderer,
        asset_name: &str,
        descriptor_set_layout: vk::DescriptorSetLayout,
    ) -> VulkanGltfAsset {
        let (gltf, buffers, asset_textures) =
            gltf::import(&asset_name).expect("Couldn't import file!");

        let textures = asset_textures
            .iter()
            .map(|properties| GltfTextureBundle::new(&renderer, properties))
            .collect::<Vec<_>>();

        let descriptor_pool = Self::create_descriptor_pool(&renderer, &gltf);

        let scenes = Self::prepare_scenes(&gltf, &buffers, &renderer);

        // TODO: Move this logic to the VulkanContext
        let physical_device_properties = unsafe {
            renderer
                .context
                .instance()
                .get_physical_device_properties(renderer.context.physical_device())
        };

        let minimum_ubo_alignment = physical_device_properties
            .limits
            .min_uniform_buffer_offset_alignment;
        let mut dynamic_alignment = std::mem::size_of::<UniformBufferObject>() as u64;
        if minimum_ubo_alignment > 0 {
            dynamic_alignment =
                (dynamic_alignment + minimum_ubo_alignment - 1) & !(minimum_ubo_alignment - 1);
        }

        let uniform_buffer = Buffer::new(
            renderer.context.clone(),
            1,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        let dynamic_uniform_buffer = Buffer::new(
            renderer.context.clone(),
            (gltf.meshes().len() as u64 * dynamic_alignment) as vk::DeviceSize,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
        );

        let number_of_swapchain_images = renderer.vulkan_swapchain.swapchain.images().len();
        let descriptor_sets = descriptor_pool
            .allocate_descriptor_sets(descriptor_set_layout, number_of_swapchain_images as _);

        let mut asset = VulkanGltfAsset {
            gltf,
            textures,
            scenes,
            descriptor_pool,
            uniform_buffer,
            dynamic_uniform_buffer,
            descriptor_sets,
            dynamic_alignment,
        };

        asset.update_ubo_indices();
        asset.update_descriptor_sets(&renderer, number_of_swapchain_images as _);
        asset
    }

    fn create_descriptor_pool(renderer: &Renderer, gltf: &gltf::Document) -> DescriptorPool {
        let number_of_swapchain_images = renderer.vulkan_swapchain.swapchain.images().len() as u32;
        let number_of_materials = gltf.materials().len() as u32;
        let number_of_samplers = number_of_materials * number_of_swapchain_images;

        // TODO: Move descriptor pool creation to method
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
            descriptor_count: number_of_samplers * number_of_swapchain_images,
        };

        let pool_sizes = [ubo_pool_size, dynamic_ubo_pool_size, sampler_pool_size];

        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes)
            .max_sets(number_of_swapchain_images)
            .build();

        DescriptorPool::new(renderer.context.clone(), pool_info)
    }

    fn update_ubo_indices(&mut self) {
        let mut indices = Vec::new();
        for (scene_index, scene) in self.scenes.iter().enumerate() {
            for (graph_index, graph) in scene.node_graphs.iter().enumerate() {
                let mut dfs = Dfs::new(&graph, NodeIndex::new(0));
                while let Some(node_index) = dfs.next(&graph) {
                    if graph[node_index].mesh.is_some() {
                        indices.push((scene_index, graph_index, node_index));
                    }
                }
            }
        }

        for (ubo_index, (scene_index, graph_index, node_index)) in indices.into_iter().enumerate() {
            self.scenes[scene_index].node_graphs[graph_index][node_index]
                .mesh
                .as_mut()
                .unwrap()
                .ubo_index = ubo_index;
        }
    }

    fn update_descriptor_sets(&self, renderer: &Renderer, number_of_swapchain_images: usize) {
        let uniform_buffer_size = mem::size_of::<UniformBufferObject>() as vk::DeviceSize;
        let buffer_info = vk::DescriptorBufferInfo::builder()
            .buffer(self.uniform_buffer.buffer())
            .offset(0)
            .range(uniform_buffer_size)
            .build();
        let buffer_infos = [buffer_info];

        let image_infos = self
            .textures
            .iter()
            .map(|texture| {
                vk::DescriptorImageInfo::builder()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(texture.view.view())
                    .sampler(texture.sampler.sampler())
                    .build()
            })
            .collect::<Vec<_>>();

        for image_index in 0..number_of_swapchain_images {
            let descriptor_set = self.descriptor_sets[image_index];
            let ubo_descriptor_write = vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&buffer_infos)
                .build();

            let dynamic_ubo_descriptor_write = vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                .buffer_info(&buffer_infos)
                .build();

            let sampler_descriptor_write = vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(2)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&image_infos)
                .build();

            let mut descriptor_writes = vec![dynamic_ubo_descriptor_write, ubo_descriptor_write];
            if !image_infos.is_empty() {
                descriptor_writes.push(sampler_descriptor_write);
            }

            unsafe {
                renderer
                    .context
                    .logical_device()
                    .logical_device()
                    .update_descriptor_sets(&descriptor_writes, &[])
            }
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
    ) -> Vec<Scene> {
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
                );
                node_graphs.push(node_graph);
            }
            scenes.push(Scene { node_graphs });
        }
        scenes
    }

    fn visit_children(
        node: &gltf::Node,
        buffers: &[gltf::buffer::Data],
        node_graph: &mut NodeGraph,
        parent_index: NodeIndex,
        renderer: &Renderer,
    ) {
        let mesh = Self::load_mesh(node, buffers, renderer);
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
            Self::visit_children(&child, buffers, node_graph, node_index, renderer);
        }
    }

    fn load_mesh(
        node: &gltf::Node,
        buffers: &[gltf::buffer::Data],
        renderer: &Renderer,
    ) -> Option<Mesh> {
        if let Some(mesh) = node.mesh() {
            let mut vertices = Vec::new();
            let mut indices = Vec::new();

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

            let vertex_buffer = renderer.transient_command_pool.create_device_local_buffer(
                renderer.graphics_queue,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                &vertices,
            );

            let index_buffer = renderer.transient_command_pool.create_device_local_buffer(
                renderer.graphics_queue,
                vk::BufferUsageFlags::INDEX_BUFFER,
                &indices,
            );

            Some(Mesh {
                primitives: all_mesh_primitives,
                vertex_buffer,
                index_buffer,
                ubo_index: 0,
            })
        } else {
            None
        }
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
    let indices = path_between_nodes(NodeIndex::new(0), node_index, graph);
    indices
        .iter()
        .fold(glm::Mat4::identity(), |transform, index| {
            // TODO: Add the animation transform
            transform * graph[*index].local_transform //* graph[*index].animation_transform.matrix()
        })
}
