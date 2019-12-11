mod app;
mod context;
mod core;
mod gltf;
mod render;
mod renderer;
mod resource;
mod sync;

use app::App;
use context::VulkanContext;

fn main() {
    env_logger::init();
    let mut app = App::new(800, 600, "Obsidian - Vulkan Rendering");
    app.run();
}
