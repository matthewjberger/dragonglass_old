use crate::{
    render::Renderer,
    resource::{TextureBundle, TextureDescription},
};
use ash::vk;
use gltf::animation::{util::ReadOutputs, Interpolation};
use nalgebra::{Matrix4, Quaternion, UnitQuaternion};
use nalgebra_glm as glm;
use petgraph::{
    dot::{Config, Dot},
    graph::{Graph, NodeIndex},
    prelude::*,
    visit::Dfs,
};
use std::fmt;

#[derive(Debug)]
pub enum TransformationSet {
    Translations(Vec<glm::Vec3>),
    Rotations(Vec<glm::Vec4>),
    Scales(Vec<glm::Vec3>),
    MorphTargetWeights(Vec<f32>),
}

#[derive(Debug, Default)]
pub struct Transform {
    translation: Option<glm::Vec3>,
    rotation: Option<glm::Quat>,
    scale: Option<glm::Vec3>,
}

impl Transform {
    pub fn matrix(&self) -> glm::Mat4 {
        let mut matrix = glm::Mat4::identity();

        if let Some(translation) = self.translation {
            matrix *= Matrix4::new_translation(&translation);
        }

        if let Some(rotation) = self.rotation {
            matrix *= Matrix4::from(UnitQuaternion::from_quaternion(rotation));
        }

        if let Some(scale) = self.scale {
            matrix *= Matrix4::new_nonuniform_scaling(&scale);
        }

        matrix
    }
}

pub type NodeGraph = Graph<Node, ()>;

pub struct Node {
    pub animation_transform: Transform,
    pub local_transform: glm::Mat4,
    pub mesh: Option<Mesh>,
    pub skin: Option<Skin>,
    pub gltf_index: usize,
    pub name: String,
}

impl fmt::Debug for Node {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Node")
            .field("name", &self.name)
            .field("gltf_index", &self.gltf_index)
            .finish()
    }
}

pub struct Scene {
    pub node_graphs: Vec<NodeGraph>,
    pub name: String,
}

pub struct Mesh {
    pub primitives: Vec<Primitive>,
    pub mesh_id: usize,
}

pub struct Skin {
    pub joints: Vec<Joint>,
    pub name: String,
}

pub struct Joint {
    pub index: usize,
    pub inverse_bind_matrix: glm::Mat4,
}

pub struct Primitive {
    pub number_of_indices: u32,
    pub first_index: u32,
    pub material_index: Option<usize>,
}

// TODO: Properly decouple the animation state from the asset as a component to make it reusable.
pub struct Animation {
    pub time: f32,
    channels: Vec<Channel>,
    max_animation_time: f32,
    pub name: String,
}

pub struct Channel {
    node_index: usize,
    inputs: Vec<f32>,
    transformations: TransformationSet,
    _interpolation: Interpolation,
    previous_key: usize,
    previous_time: f32,
}

pub struct GltfAsset {
    pub gltf: gltf::Document,
    pub textures: Vec<TextureBundle>,
    pub scenes: Vec<Scene>,
    pub number_of_meshes: usize,
    pub animations: Vec<Animation>,
    pub vertices: Vec<f32>,
    pub indices: Vec<u32>,
}

impl GltfAsset {
    pub const DEFAULT_NAME: &'static str = "<Unnamed>";

    pub fn new(renderer: &Renderer, asset_name: &str) -> GltfAsset {
        let (gltf, buffers, asset_textures) =
            gltf::import(&asset_name).expect("Couldn't import file!");

        let textures = asset_textures
            .iter()
            .map(|image_data| {
                let description = TextureDescription::from_gltf(&image_data);
                TextureBundle::new(
                    renderer.context.clone(),
                    &renderer.command_pool,
                    &description,
                )
            })
            .collect::<Vec<_>>();

        let animations = Self::prepare_animations(&gltf, &buffers);

        let (mut scenes, vertices, indices) = Self::prepare_scenes(&gltf, &buffers, &renderer);
        Self::update_ubo_indices(&mut scenes);

        let number_of_meshes = gltf.nodes().filter(|node| node.mesh().is_some()).count();

        GltfAsset {
            gltf,
            textures,
            scenes,
            number_of_meshes,
            animations,
            vertices,
            indices,
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
            let name = scene.name().unwrap_or(&Self::DEFAULT_NAME).to_string();
            scenes.push(Scene { node_graphs, name });
        }
        (scenes, vertices, indices)
    }

    fn load_skin(node: &gltf::Node, buffers: &[gltf::buffer::Data]) -> Option<Skin> {
        if let Some(skin) = node.skin() {
            let reader = skin.reader(|buffer| Some(&buffers[buffer.index()]));
            let inverse_bind_matrices = reader
                .read_inverse_bind_matrices()
                .map_or(Vec::new(), |matrices| {
                    matrices.map(glm::Mat4::from).collect::<Vec<_>>()
                });

            let mut joints = Vec::new();
            for (index, joint_node) in skin.joints().enumerate() {
                let inverse_bind_matrix = if inverse_bind_matrices.is_empty() {
                    glm::Mat4::identity()
                } else {
                    inverse_bind_matrices[index]
                };
                joints.push(Joint {
                    inverse_bind_matrix,
                    index: joint_node.index(),
                });
            }

            let name = skin.name().unwrap_or(&Self::DEFAULT_NAME).to_string();

            Some(Skin { joints, name })
        } else {
            None
        }
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
        let skin = Self::load_skin(node, buffers);
        let name = node.name().unwrap_or(&Self::DEFAULT_NAME).to_string();
        let node_info = Node {
            animation_transform: Transform::default(),
            local_transform: Self::determine_transform(node),
            mesh,
            skin,
            gltf_index: node.index(),
            name,
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

    pub fn vertex_stride() -> usize {
        let position_length = 3;
        let normal_length = 3;
        let tex_coords_0_length = 2;
        let tex_coords_1_length = 2;
        let joints_0_length = 4;
        let weights_0_length = 4;

        position_length
            + normal_length
            + tex_coords_0_length
            + tex_coords_1_length
            + joints_0_length
            + weights_0_length
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
                let stride = Self::vertex_stride() * std::mem::size_of::<f32>();

                let vertex_list_size = vertices.len() * std::mem::size_of::<u32>();
                let vertex_count = (vertex_list_size / stride) as u32;

                // Start reading primitive data
                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

                let positions = reader
                    .read_positions()
                    .expect("Failed to read any vertex positions from the model. Vertex positions are required.")
                    .map(glm::Vec3::from)
                    .collect::<Vec<_>>();
                let data_length = positions.len();

                let normals = reader
                    .read_normals()
                    .map_or(vec![glm::vec3(0.0, 0.0, 0.0); data_length], |normals| {
                        normals.map(glm::Vec3::from).collect::<Vec<_>>()
                    });

                let convert_coords =
                    |coords: gltf::mesh::util::ReadTexCoords<'_>| -> Vec<glm::Vec2> {
                        coords.into_f32().map(glm::Vec2::from).collect::<Vec<_>>()
                    };

                let tex_coords_0 = reader
                    .read_tex_coords(0)
                    .map_or(vec![glm::vec2(0.0, 0.0); data_length], convert_coords);

                let tex_coords_1 = reader
                    .read_tex_coords(1)
                    .map_or(vec![glm::vec2(0.0, 0.0); data_length], convert_coords);

                let convert_joints = |coords: gltf::mesh::util::ReadJoints<'_>| -> Vec<glm::Vec4> {
                    coords
                        .into_u16()
                        .map(|joint| {
                            glm::vec4(joint[0] as _, joint[1] as _, joint[2] as _, joint[3] as _)
                        })
                        .collect::<Vec<_>>()
                };

                let joints_0 = reader.read_joints(0).map_or(
                    vec![glm::vec4(0.0, 0.0, 0.0, 0.0); data_length],
                    convert_joints,
                );

                let convert_weights =
                    |coords: gltf::mesh::util::ReadWeights<'_>| -> Vec<glm::Vec4> {
                        coords.into_f32().map(glm::Vec4::from).collect::<Vec<_>>()
                    };

                let weights_0 = reader.read_weights(0).map_or(
                    vec![glm::vec4(1.0, 0.0, 0.0, 0.0); data_length],
                    convert_weights,
                );

                for index in 0..positions.len() {
                    vertices.extend_from_slice(positions[index].as_slice());
                    vertices.extend_from_slice(normals[index].as_slice());
                    vertices.extend_from_slice(tex_coords_0[index].as_slice());
                    vertices.extend_from_slice(tex_coords_1[index].as_slice());
                    vertices.extend_from_slice(joints_0[index].as_slice());
                    vertices.extend_from_slice(weights_0[index].as_slice());
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

    // TODO: Write this method for vec3's and vec4's
    // fn interpolate(interpolation: Interpolation) {
    //     match interpolation {
    //         Interpolation::Linear => {}
    //         Interpolation::Step => {}
    //         Interpolation::CatmullRomSpline => {}
    //         Interpolation::CubicSpline => {}
    //     }
    // }

    fn prepare_animations(gltf: &gltf::Document, buffers: &[gltf::buffer::Data]) -> Vec<Animation> {
        let mut animations = Vec::new();
        for animation in gltf.animations() {
            let name = animation.name().unwrap_or(&Self::DEFAULT_NAME).to_string();
            let mut channels = Vec::new();
            for channel in animation.channels() {
                let sampler = channel.sampler();
                let _interpolation = sampler.interpolation();
                let node_index = channel.target().node().index();
                let reader = channel.reader(|buffer| Some(&buffers[buffer.index()]));
                let inputs = reader.read_inputs().unwrap().collect::<Vec<_>>();
                let outputs = reader.read_outputs().unwrap();
                let transformations: TransformationSet;
                match outputs {
                    ReadOutputs::Translations(translations) => {
                        let translations = translations.map(glm::Vec3::from).collect::<Vec<_>>();
                        transformations = TransformationSet::Translations(translations);
                    }
                    ReadOutputs::Rotations(rotations) => {
                        let rotations = rotations
                            .into_f32()
                            .map(glm::Vec4::from)
                            .collect::<Vec<_>>();
                        transformations = TransformationSet::Rotations(rotations);
                    }
                    ReadOutputs::Scales(scales) => {
                        let scales = scales.map(glm::Vec3::from).collect::<Vec<_>>();
                        transformations = TransformationSet::Scales(scales);
                    }
                    ReadOutputs::MorphTargetWeights(weights) => {
                        let morph_target_weights = weights.into_f32().collect::<Vec<_>>();
                        transformations =
                            TransformationSet::MorphTargetWeights(morph_target_weights);
                    }
                }
                channels.push(Channel {
                    node_index,
                    inputs,
                    transformations,
                    _interpolation,
                    previous_key: 0,
                    previous_time: 0.0,
                });
            }

            let max_animation_time = channels
                .iter()
                .flat_map(|channel| channel.inputs.iter().copied())
                .fold(0.0, f32::max);

            animations.push(Animation {
                channels,
                time: 0.0,
                max_animation_time,
                name,
            });
        }
        animations
    }

    pub fn animate(&mut self) {
        // TODO: Allow for specifying a specific animation by name
        for animation in self.animations.iter_mut() {
            if animation.time > animation.max_animation_time {
                animation.time = 0.0;
            }
            if animation.time < 0.0 {
                animation.time = animation.max_animation_time;
            }
            for channel in animation.channels.iter_mut() {
                for scene in self.scenes.iter_mut() {
                    for graph in scene.node_graphs.iter_mut() {
                        for node_index in graph.node_indices() {
                            if graph[node_index].gltf_index == channel.node_index {
                                let max = *channel.inputs.last().unwrap();
                                let mut time = animation.time % max;
                                let first_input = channel.inputs.first().unwrap();
                                if time.lt(first_input) {
                                    time = *first_input;
                                }

                                if channel.previous_time > time {
                                    channel.previous_key = 0;
                                }
                                channel.previous_time = time;

                                let mut next_key: usize = 0;
                                for index in channel.previous_key..channel.inputs.len() {
                                    let index = index as usize;
                                    if time <= channel.inputs[index] {
                                        next_key =
                                            nalgebra::clamp(index, 1, channel.inputs.len() - 1);
                                        break;
                                    }
                                }
                                channel.previous_key = nalgebra::clamp(next_key - 1, 0, next_key);

                                let key_delta =
                                    channel.inputs[next_key] - channel.inputs[channel.previous_key];
                                let normalized_time =
                                    (time - channel.inputs[channel.previous_key]) / key_delta;

                                // TODO: Interpolate with other methods
                                // Only Linear interpolation is used for now
                                match &channel.transformations {
                                    TransformationSet::Translations(translations) => {
                                        let start = translations[channel.previous_key];
                                        let end = translations[next_key];
                                        let translation = start.lerp(&end, normalized_time);
                                        let translation_vec =
                                            glm::make_vec3(translation.as_slice());
                                        graph[node_index].animation_transform.translation =
                                            Some(translation_vec);
                                    }
                                    TransformationSet::Rotations(rotations) => {
                                        let start = rotations[channel.previous_key];
                                        let end = rotations[next_key];
                                        let start_quat =
                                            Quaternion::new(start[3], start[0], start[1], start[2]);
                                        let end_quat =
                                            Quaternion::new(end[3], end[0], end[1], end[2]);
                                        let rotation_quat =
                                            start_quat.lerp(&end_quat, normalized_time);
                                        graph[node_index].animation_transform.rotation =
                                            Some(rotation_quat);
                                    }
                                    TransformationSet::Scales(scales) => {
                                        let start = scales[channel.previous_key];
                                        let end = scales[next_key];
                                        let scale = start.lerp(&end, normalized_time);
                                        let scale_vec = glm::make_vec3(scale.as_slice());
                                        graph[node_index].animation_transform.scale =
                                            Some(scale_vec);
                                    }
                                    TransformationSet::MorphTargetWeights(_weights) => {
                                        unimplemented!()
                                    }
                                }

                                break;
                            }
                        }
                    }
                }
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

    pub fn matching_node_index(gltf_index: usize, graph: &NodeGraph) -> Option<NodeIndex> {
        let mut dfs = Dfs::new(&graph, NodeIndex::new(0));
        while let Some(node_index) = dfs.next(&graph) {
            if graph[node_index].gltf_index == gltf_index {
                return Some(node_index);
            }
        }
        None
    }

    pub fn print_nodegraph(graph: &NodeGraph) {
        println!("{:?}", Dot::with_config(&graph, &[Config::EdgeNoLabel]));
    }

    pub fn calculate_global_transform(node_index: NodeIndex, graph: &NodeGraph) -> glm::Mat4 {
        let indices = Self::path_between_nodes(NodeIndex::new(0), node_index, graph);
        indices
            .iter()
            .fold(glm::Mat4::identity(), |transform, index| {
                transform
                    * graph[*index].local_transform
                    * graph[*index].animation_transform.matrix()
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

    pub fn create_vertex_attributes() -> [vk::VertexInputAttributeDescription; 6] {
        let float_size = std::mem::size_of::<f32>();
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
            .offset((3 * float_size) as _)
            .build();

        let tex_coord_0_description = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(2)
            .format(vk::Format::R32G32_SFLOAT)
            .offset((6 * float_size) as _)
            .build();

        let tex_coord_1_description = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(3)
            .format(vk::Format::R32G32_SFLOAT)
            .offset((8 * float_size) as _)
            .build();

        let joint_0_description = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(4)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .offset((10 * float_size) as _)
            .build();

        let weight_0_description = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(5)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .offset((14 * float_size) as _)
            .build();

        [
            position_description,
            normal_description,
            tex_coord_0_description,
            tex_coord_1_description,
            joint_0_description,
            weight_0_description,
        ]
    }

    pub fn create_vertex_input_descriptions() -> [vk::VertexInputBindingDescription; 1] {
        let vertex_input_binding_description = vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride((18 * std::mem::size_of::<f32>()) as _)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build();
        [vertex_input_binding_description]
    }
}
