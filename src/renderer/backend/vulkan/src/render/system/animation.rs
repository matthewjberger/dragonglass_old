use crate::render::{
    component::{AnimationComponent, GltfAssetComponent},
    Renderer,
};
use gltf::animation::{Animation, Property};
use legion::prelude::*;

pub fn animation_system() -> Box<dyn Runnable> {
    SystemBuilder::new("render")
        .write_resource::<Renderer>()
        .with_query(<(Write<AnimationComponent>, Read<GltfAssetComponent>)>::query())
        .build_thread_local(move |_, mut world, renderer, query| {
            for (mut animation_info, asset_info) in query.iter(&mut world) {
                if let Some(asset_index) = asset_info.loaded_asset_index {
                    let asset = &renderer.assets[asset_index];

                    // TODO: Allow selecting animation. Here only the first animation is use
                    let animations = asset.gltf.animations().collect::<Vec<Animation>>();
                    let first_animation_index = 0;
                    let first_animation = &animations[first_animation_index];
                    for channel in first_animation.channels() {
                        let target = channel.target();
                        let node = target.node();

                        let node_index = target.node().index();
                        let channel_map = &asset.animations[first_animation_index];
                        let channel_info = &channel_map[&node_index];

                        let first_input = channel_info.inputs.first().unwrap();
                        let mut time: f32 =
                            animation_info.current_time % channel_info.inputs.last().unwrap();
                        if time.lt(first_input) {
                            time = *first_input;
                        }

                        if animation_info.previous_time > time {
                            animation_info.previous_key = 0;
                        }
                        animation_info.previous_time = time;

                        let mut next_key: usize = 0;
                        for index in animation_info.previous_key..channel_info.inputs.len() {
                            let index = index as usize;
                            if time <= channel_info.inputs[index] {
                                next_key = nalgebra::clamp(index, 1, channel_info.inputs.len() - 1);
                                break;
                            }
                        }
                        animation_info.previous_key = nalgebra::clamp(next_key - 1, 0, next_key);

                        let key_delta = channel_info.inputs[next_key]
                            - channel_info.inputs[animation_info.previous_key];
                        let normalized_time =
                            (time - channel_info.inputs[animation_info.previous_key]) / key_delta;

                        match target.property() {
                            Property::Rotation => {
                                // let node_info = &mut asset.nodes[&node_index];
                                // node_info.animation_transform = Some(

                                // );
                            }
                            Property::Scale => {}
                            Property::Translation => {}
                            Property::MorphTargetWeights => {}
                        }
                    }
                }
            }
        })
}
