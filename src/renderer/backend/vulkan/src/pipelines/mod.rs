pub use self::gltf_pipeline::GltfPipeline;

pub mod gltf_pipeline;

#[derive(Debug, PartialEq, Eq, Hash)]
pub enum PipelineType {
    GltfAsset,
}
