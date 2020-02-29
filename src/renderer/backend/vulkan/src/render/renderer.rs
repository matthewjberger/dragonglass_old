use crate::{
    core::VulkanContext,
    render::{pipeline_gltf::GltfPipeline, VulkanSwapchain},
    resource::CommandPool,
    sync::SynchronizationSet,
};
use ash::{version::DeviceV1_0, vk};
use std::sync::Arc;

pub struct Renderer {
    pub context: Arc<VulkanContext>,
    pub vulkan_swapchain: VulkanSwapchain,
    pub graphics_queue: vk::Queue,
    pub pipeline_gltf: Option<GltfPipeline>,
    pub present_queue: vk::Queue,
    pub synchronization_set: SynchronizationSet,
    pub current_frame: usize,
    pub command_pool: CommandPool,
    pub transient_command_pool: CommandPool,
}

impl Renderer {
    pub fn new(window: &winit::Window) -> Self {
        let context =
            Arc::new(VulkanContext::new(&window).expect("Failed to create VulkanContext"));

        let synchronization_set =
            SynchronizationSet::new(context.clone()).expect("Failed to create sync objects");

        unsafe {
            context
                .logical_device()
                .logical_device()
                .device_wait_idle()
                .expect("Failed to wait for the logical device to be idle!")
        };

        let graphics_queue = unsafe {
            context
                .logical_device()
                .logical_device()
                .get_device_queue(context.graphics_queue_family_index(), 0)
        };

        let present_queue = unsafe {
            context
                .logical_device()
                .logical_device()
                .get_device_queue(context.present_queue_family_index(), 0)
        };

        let command_pool = CommandPool::new(context.clone(), vk::CommandPoolCreateFlags::empty());

        let transient_command_pool =
            CommandPool::new(context.clone(), vk::CommandPoolCreateFlags::TRANSIENT);

        let logical_size = window
            .get_inner_size()
            .expect("Failed to get the window's inner size!");
        let dimensions = [logical_size.width as u32, logical_size.height as u32];

        let vulkan_swapchain =
            VulkanSwapchain::new(context.clone(), dimensions, graphics_queue, &command_pool);

        Renderer {
            context,
            graphics_queue,
            pipeline_gltf: None,
            present_queue,
            synchronization_set,
            current_frame: 0,
            vulkan_swapchain,
            command_pool,
            transient_command_pool,
        }
    }

    #[allow(dead_code)]
    pub fn recreate_swapchain(&mut self, _: Option<[u32; 2]>) {
        log::debug!("Recreating swapchain");
        // TODO: Implement swapchain recreation
    }

    pub fn wait_idle(&self) {
        unsafe {
            self.context
                .logical_device()
                .logical_device()
                .device_wait_idle()
                .expect("Failed to wait for the logical device to be idle!")
        };
    }
}
