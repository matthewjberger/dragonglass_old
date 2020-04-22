use nalgebra_glm as glm;

#[derive(Debug)]
pub struct AssetName(pub String); // TODO: Make this a key instead of a full path

#[derive(Debug)]
pub struct AssetIndex(pub usize);

#[derive(Debug)]
pub struct Transform {
    pub translate: glm::Mat4,
    pub rotate: glm::Mat4,
    pub scale: glm::Mat4,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            translate: glm::Mat4::identity(),
            rotate: glm::Mat4::identity(),
            scale: glm::Mat4::identity(),
        }
    }
}
