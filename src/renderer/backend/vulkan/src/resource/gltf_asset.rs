use crate::{
    core::ImageView,
    resource::{Buffer, DescriptorPool, Sampler, Texture},
};
use ash::vk;
use gltf::animation::{util::ReadOutputs, Interpolation};
use nalgebra::{Matrix4, Quaternion, UnitQuaternion};
use nalgebra_glm as glm;
use petgraph::{
    graph::{Graph, NodeIndex},
    prelude::*,
    visit::Dfs,
};
use std::collections::HashMap;

// graph index -> mesh
pub type MeshMap = HashMap<usize, Mesh>;

pub struct VulkanGltfAsset {
    pub raw_asset: GltfAsset,
    pub textures: Vec<VulkanTexture>,
    pub meshes: HashMap<usize, MeshMap>,
    pub descriptor_pool: DescriptorPool,
}

pub struct VulkanTexture {
    pub texture: Texture,
    pub view: ImageView,
    pub sampler: Sampler,
}

pub struct Mesh {
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub primitives: Vec<Primitive>,
    pub uniform_buffers: Vec<Buffer>,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub node_index: NodeIndex,
}

pub struct Primitive {
    pub first_index: u32,
}

// GLTF

// TODO: Load bounding volumes using ncollide
pub type NodeGraph = Graph<Node, ()>;

#[derive(Debug)]
enum TransformationSet {
    Translations(Vec<glm::Vec3>),
    Rotations(Vec<glm::Vec4>),
    Scales(Vec<glm::Vec3>),
    MorphTargetWeights(Vec<f32>),
}

#[derive(Debug)]
pub struct Skin {
    pub joints: Vec<Joint>,
}

#[derive(Debug)]
pub struct Joint {
    pub index: usize,
    pub inverse_bind_matrix: glm::Mat4,
}

#[derive(Debug)]
pub struct Transform {
    translation: glm::Vec3,
    rotation: glm::Quat,
    scale: glm::Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            translation: glm::Vec3::identity(),
            rotation: glm::Quat::identity(),
            scale: glm::Vec3::identity(),
        }
    }
}

#[allow(dead_code)]
impl Transform {
    pub fn matrix(&self) -> glm::Mat4 {
        Matrix4::new_translation(&self.translation)
            * Matrix4::from(UnitQuaternion::from_quaternion(self.rotation))
            * Matrix4::new_nonuniform_scaling(&self.scale)
    }
}

#[derive(Debug)]
pub struct Node {
    pub local_transform: glm::Mat4,
    pub animation_transform: Transform,
    pub mesh: Option<GltfMesh>,
    pub skin: Option<Skin>,
    pub index: usize,
}

#[derive(Debug)]
pub struct GltfMesh {
    pub primitives: Vec<GltfPrimitive>,
}

#[derive(Debug)]
pub struct GltfPrimitive {
    pub number_of_indices: u32,
    pub material_index: usize,
    pub vertices: Vec<f32>,
    pub indices: Vec<u32>,
}

#[derive(Debug)]
pub struct Channel {
    node_index: usize,
    inputs: Vec<f32>,
    transformations: TransformationSet,
    interpolation: Interpolation,
    previous_key: usize,
    previous_time: f32,
}

#[derive(Debug)]
pub struct Animation {
    channels: Vec<Channel>,
}

#[derive(Debug)]
pub struct Scene {
    pub node_graphs: Vec<NodeGraph>,
}

#[derive(Debug)]
pub struct GltfAsset {
    pub gltf: gltf::Document,
    pub textures: Vec<gltf::image::Data>,
    pub scenes: Vec<Scene>,
    pub animations: Vec<Animation>,
}

#[allow(dead_code)]
impl GltfAsset {
    pub fn from_file(path: &str) -> Self {
        let (gltf, buffers, textures) = gltf::import(path).expect("Couldn't import file!");
        let scenes = prepare_scenes(&gltf, &buffers);
        let animations = prepare_animations(&gltf, &buffers);

        GltfAsset {
            gltf,
            scenes,
            textures,
            animations,
        }
    }

    // TODO: Group the closure's params into a struct
    pub fn walk<F>(&self, mut action: F)
    where
        F: FnMut(usize, usize, NodeIndex, &Node, &NodeGraph),
    {
        for (scene_index, scene) in self.scenes.iter().enumerate() {
            for (graph_index, graph) in scene.node_graphs.iter().enumerate() {
                let mut dfs = Dfs::new(&graph, NodeIndex::new(0));
                while let Some(node_index) = dfs.next(&graph) {
                    action(
                        scene_index,
                        graph_index,
                        node_index,
                        &graph[node_index],
                        &graph,
                    );
                }
            }
        }
    }

    pub fn lookup_material(&self, index: usize) -> gltf::Material {
        self.gltf
            .materials()
            .nth(index)
            .expect("Failed to lookup material on gltf asset!")
    }

    // TODO: Do this with an ecs system
    pub fn animate(&mut self, seconds: f32) {
        // TODO: Allow for specifying a specific animation by name
        for animation in self.animations.iter_mut() {
            for channel in animation.channels.iter_mut() {
                for scene in self.scenes.iter_mut() {
                    for graph in scene.node_graphs.iter_mut() {
                        for node_index in graph.node_indices() {
                            if graph[node_index].index == channel.node_index {
                                let mut time = seconds
                                    % channel
                                        .inputs
                                        .last()
                                        .expect("Failed to get channel's last input!");
                                let first_input = channel
                                    .inputs
                                    .first()
                                    .expect("Failed to get channel's first input!");
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
                                            translation_vec;
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
                                            rotation_quat;
                                    }
                                    TransformationSet::Scales(scales) => {
                                        let start = scales[channel.previous_key];
                                        let end = scales[next_key];
                                        let scale = start.lerp(&end, normalized_time);
                                        let scale_vec = glm::make_vec3(scale.as_slice());
                                        graph[node_index].animation_transform.scale = scale_vec;
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

    pub fn number_of_meshes(&self) -> u32 {
        let mut number_of_meshes = 0;
        for scene in self.scenes.iter() {
            for graph in scene.node_graphs.iter() {
                let mut dfs = Dfs::new(&graph, NodeIndex::new(0));
                while let Some(node_index) = dfs.next(&graph) {
                    if graph[node_index].mesh.as_ref().is_some() {
                        number_of_meshes += 1;
                    }
                }
            }
        }
        number_of_meshes
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
    // TODO: load names if present as well
    let mut animations = Vec::new();
    for animation in gltf.animations() {
        let mut channels = Vec::new();
        for channel in animation.channels() {
            let sampler = channel.sampler();
            let interpolation = sampler.interpolation();
            let node_index = channel.target().node().index();
            let reader = channel.reader(|buffer| Some(&buffers[buffer.index()]));
            let inputs = reader
                .read_inputs()
                .expect("Failed to read inputs!")
                .collect::<Vec<_>>();
            let outputs = reader.read_outputs().expect("Failed to read outputs!");
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
                    transformations = TransformationSet::MorphTargetWeights(morph_target_weights);
                }
            }
            channels.push(Channel {
                node_index,
                inputs,
                transformations,
                interpolation,
                previous_key: 0,
                previous_time: 0.0,
            });
        }
        animations.push(Animation { channels });
    }
    animations
}

// TODO: Make graph a collection of collections of graphs belonging to the scene (Vec<Vec<NodeGraph>>)
// TODO: Load names for scenes and nodes
fn prepare_scenes(gltf: &gltf::Document, buffers: &[gltf::buffer::Data]) -> Vec<Scene> {
    let mut scenes: Vec<Scene> = Vec::new();
    for scene in gltf.scenes() {
        let mut node_graphs: Vec<NodeGraph> = Vec::new();
        for node in scene.nodes() {
            let mut node_graph = NodeGraph::new();
            visit_children(&node, &buffers, &mut node_graph, NodeIndex::new(0_usize));
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
) {
    let node_info = Node {
        local_transform: determine_transform(node),
        animation_transform: Transform::default(),
        mesh: load_mesh(node, buffers),
        skin: load_skin(node, buffers),
        index: node.index(),
    };

    let node_index = node_graph.add_node(node_info);
    if parent_index != node_index {
        node_graph.add_edge(parent_index, node_index, ());
    }

    for child in node.children() {
        visit_children(&child, buffers, node_graph, node_index);
    }
}

fn load_mesh(node: &gltf::Node, buffers: &[gltf::buffer::Data]) -> Option<GltfMesh> {
    if let Some(mesh) = node.mesh() {
        let mut all_primitive_info = Vec::new();
        for primitive in mesh.primitives() {
            let (vertices, indices) = read_buffer_data(&primitive, &buffers);

            let primitive_info = GltfPrimitive {
                number_of_indices: indices.len() as u32,
                material_index: primitive
                    .material()
                    .index()
                    .expect("Failed to get the index of a gltf material!"),
                vertices,
                indices,
            };

            all_primitive_info.push(primitive_info);
        }
        Some(GltfMesh {
            primitives: all_primitive_info,
        })
    } else {
        None
    }
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

        Some(Skin { joints })
    } else {
        None
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

fn read_buffer_data(
    primitive: &gltf::Primitive,
    buffers: &[gltf::buffer::Data],
) -> (Vec<f32>, Vec<u32>) {
    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

    let positions = reader
        .read_positions()
        .expect(
            "Failed to read any vertex positions from the model. Vertex positions are required.",
        )
        .map(glm::Vec3::from)
        .collect::<Vec<_>>();

    let normals = reader
        .read_normals()
        .map_or(vec![glm::vec3(0.0, 0.0, 0.0); positions.len()], |normals| {
            normals.map(glm::Vec3::from).collect::<Vec<_>>()
        });

    let convert_coords = |coords: gltf::mesh::util::ReadTexCoords<'_>| -> Vec<glm::Vec2> {
        coords.into_f32().map(glm::Vec2::from).collect::<Vec<_>>()
    };
    let tex_coords_0 = reader
        .read_tex_coords(0)
        .map_or(vec![glm::vec2(0.0, 0.0); positions.len()], convert_coords);

    let mut vertices = Vec::new();
    for ((position, normal), tex_coord_0) in positions
        .iter()
        .zip(normals.iter())
        .zip(tex_coords_0.iter())
    {
        vertices.extend_from_slice(position.as_slice());
        vertices.extend_from_slice(normal.as_slice());
        vertices.extend_from_slice(tex_coord_0.as_slice());
    }

    let indices = reader
        .read_indices()
        .map(|read_indices| read_indices.into_u32().collect::<Vec<_>>())
        .expect("Failed to read indices!");

    (vertices, indices)
}

#[allow(dead_code)]
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

#[allow(dead_code)]
pub fn calculate_global_transform(node_index: NodeIndex, graph: &NodeGraph) -> glm::Mat4 {
    let indices = path_between_nodes(NodeIndex::new(0), node_index, graph);
    indices
        .iter()
        .fold(glm::Mat4::identity(), |transform, index| {
            transform * graph[*index].local_transform * graph[*index].animation_transform.matrix()
        })
}
