use crate::render::{component::GltfAssetComponent, Renderer};
use specs::prelude::*;

pub struct PrepareRendererSystem;

impl<'a> System<'a> for PrepareRendererSystem {
    type SystemData = (
        WriteExpect<'a, Renderer>,
        ReadStorage<'a, GltfAssetComponent>,
    );

    fn run(&mut self, data: Self::SystemData) {
        let (mut renderer, assets) = data;
        let renderer = &mut renderer;
        for asset in assets.join() {
            // TODO: Make a command in the renderer to reset resources
            // TODO: Batch assets into a large vertex buffer
            renderer.load_gltf_asset(&asset.asset_name);
        }

        renderer.create_render_passes();
    }
}
