pub use self::gltf::GltfPipeline;

pub mod gltf;

#[derive(Debug, PartialEq, Eq, Hash)]
pub enum PipelineType {
    GltfAsset,
}
