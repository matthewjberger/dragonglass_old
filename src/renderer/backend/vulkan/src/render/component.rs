use nalgebra_glm as glm;

#[derive(Debug, Default)]
pub struct GltfAssetComponent {
    pub asset_name: String, // TODO: Make this a tag rather than a full path
    pub loaded_asset_index: Option<usize>,
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

#[derive(Debug, Default)]
pub struct AnimationComponent {
    pub current_time: f32,
    pub previous_time: f32,
    pub previous_key: usize,
}
