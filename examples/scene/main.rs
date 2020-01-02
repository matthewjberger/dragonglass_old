use dragonglass::app::App;

fn main() {
    env_logger::init();
    let mut app = App::new(800, 600, "Dragonglass - Vulkan Rendering");
    app.run();
}
