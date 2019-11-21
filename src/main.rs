mod app;
mod debug;
mod error;
mod platform;

use app::VulkanApp;

fn main() {
    env_logger::init();
    match VulkanApp::new() {
        Ok(mut app) => app.run(),
        Err(error) => log::error!("Failed to create application. Cause: {}", error),
    }
}
