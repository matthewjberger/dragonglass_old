use dragonglass::app::App;

fn main() {
    env_logger::init();
    let mut app = App::new(1920, 1080, "Dragonglass - Vulkan Rendering");
    app.run();
}
