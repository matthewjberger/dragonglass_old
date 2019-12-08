use ash::vk;
use nalgebra_glm as glm;
use std::mem;

#[derive(Debug, Copy, Clone)]
pub struct Vertex {
    position: glm::Vec3,
    color: glm::Vec3,
    tex_coords: glm::Vec2,
}

impl Vertex {
    pub fn new(position: glm::Vec3, color: glm::Vec3, tex_coords: glm::Vec2) -> Self {
        Vertex {
            position,
            color,
            tex_coords,
        }
    }

    pub fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(mem::size_of::<Vertex>() as _)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()
    }

    pub fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3] {
        let position_description = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0)
            .build();

        let color_description = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(12)
            .build();

        let tex_coords_description = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(2)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(24)
            .build();

        [
            position_description,
            color_description,
            tex_coords_description,
        ]
    }
}
