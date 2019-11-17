mod app;
mod platform;
use app::*;

fn main() {
    let mut vulkan_app = VulkanApp::new();
    vulkan_app.run();
}
