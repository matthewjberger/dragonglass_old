mod app;
mod context;
mod core;
mod render;
mod render_state;
mod resource;
mod sync;
mod vertex;

use app::App;
use context::VulkanContext;

fn main() {
    env_logger::init();
    let mut app = App::new(800, 600, "Vulkan Tutorial");
    app.run();
}
