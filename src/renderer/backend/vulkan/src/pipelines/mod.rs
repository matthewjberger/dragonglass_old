pub use self::gltf::pipeline::GltfPipeline;

pub mod gltf;

#[derive(Debug, PartialEq, Eq, Hash)]
pub enum PipelineType {
    GltfAsset,
}
