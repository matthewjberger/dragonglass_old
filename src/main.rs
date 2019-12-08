mod app;
mod context;
mod core;
mod render;
mod renderer;
mod resource;
mod sync;
mod vertex;

use app::App;
use context::VulkanContext;

fn main() {
    env_logger::init();
    let mut app = App::new(800, 600, "Obsidian - Vulkan Rendering");
    app.run();
}
