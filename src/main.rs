mod app;
mod core;
mod model;
mod render;
mod resource;
mod sync;

use app::App;

fn main() {
    env_logger::init();
    let mut app = App::new(800, 600, "Dragonglass - Vulkan Rendering");
    app.run();
}
