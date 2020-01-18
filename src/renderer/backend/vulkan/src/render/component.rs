use nalgebra_glm as glm;

// TODO: Rename MeshComponent to something more generic. (RenderComponent?)
#[derive(Debug)]
pub struct GltfAssetComponent {
    pub asset_name: String, // TODO: Make this a tag rather than a full path
}

#[derive(Debug)]
pub struct TransformComponent {
    pub translate: glm::Mat4,
    pub rotate: glm::Mat4,
    pub scale: glm::Mat4,
}

impl Default for TransformComponent {
    fn default() -> Self {
        Self {
            translate: glm::Mat4::identity(),
            rotate: glm::Mat4::identity(),
            scale: glm::Mat4::identity(),
        }
    }
}
