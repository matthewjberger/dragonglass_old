use crate::{
    core::{ImageView, Swapchain, VulkanContext},
    render::{pipeline_gltf::GltfPipeline, Framebuffer, RenderPass},
    resource::{CommandPool, Texture},
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

pub struct VulkanSwapchain {
    pub swapchain: Swapchain,
    pub render_pass: RenderPass,
    pub depth_texture: Texture,
    pub depth_texture_view: ImageView,
    pub framebuffers: Vec<Framebuffer>,
}

impl VulkanSwapchain {
    pub fn new(
        context: Arc<VulkanContext>,
        dimensions: [u32; 2],
        graphics_queue: vk::Queue,
        command_pool: &CommandPool,
    ) -> Self {
        let depth_format = context.determine_depth_format(
            vk::ImageTiling::OPTIMAL,
            vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
        );

        let swapchain = Swapchain::new(context.clone(), dimensions);
        let render_pass = RenderPass::new(context.clone(), swapchain.properties(), depth_format);

        let create_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width: swapchain.properties().extent.width,
                height: swapchain.properties().extent.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .format(depth_format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(vk::SampleCountFlags::TYPE_1)
            .flags(vk::ImageCreateFlags::empty())
            .build();
        let depth_texture = Texture::new(context.clone(), create_info);

        command_pool.transition_image_layout(
            graphics_queue,
            depth_texture.image(),
            depth_format,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        );

        let create_info = vk::ImageViewCreateInfo::builder()
            .image(depth_texture.image())
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(depth_format)
            .components(vk::ComponentMapping {
                r: vk::ComponentSwizzle::IDENTITY,
                g: vk::ComponentSwizzle::IDENTITY,
                b: vk::ComponentSwizzle::IDENTITY,
                a: vk::ComponentSwizzle::IDENTITY,
            })
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .build();
        let depth_texture_view = ImageView::new(context.clone(), create_info);

        // Create one framebuffer for each image in the swapchain
        let framebuffers = swapchain
            .image_views()
            .iter()
            .map(|view| [view.view(), depth_texture_view.view()])
            .map(|attachments| {
                let create_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(render_pass.render_pass())
                    .attachments(&attachments)
                    .width(swapchain.properties().extent.width)
                    .height(swapchain.properties().extent.height)
                    .layers(1)
                    .build();
                Framebuffer::new(context.clone(), create_info)
            })
            .collect::<Vec<_>>();

        VulkanSwapchain {
            swapchain,
            render_pass,
            depth_texture,
            depth_texture_view,
            framebuffers,
        }
    }
}
