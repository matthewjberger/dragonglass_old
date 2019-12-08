mod app;
mod core;
mod render;
mod renderer;
mod resource;
mod sync;
mod vertex;

use app::App;

fn main() {
    env_logger::init();
    let mut app = App::new(800, 600, "Obsidian - Vulkan Rendering");
    app.run();
}
