use crate::{
    core::{ImageView, Swapchain, SwapchainProperties, VulkanContext},
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
        let render_pass =
            Self::create_render_pass(context.clone(), &swapchain.properties(), depth_format);

        let swapchain_extent = swapchain.properties().extent;
        let depth_texture =
            Self::create_depth_texture(context.clone(), &swapchain_extent, depth_format);

        command_pool.transition_image_layout(
            graphics_queue,
            depth_texture.image(),
            depth_format,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        );

        let depth_texture_view =
            Self::create_depth_texture_view(context.clone(), &depth_texture, depth_format);

        let framebuffers = Self::create_framebuffers(
            context.clone(),
            &swapchain,
            &depth_texture_view,
            &render_pass,
        );

        VulkanSwapchain {
            swapchain,
            render_pass,
            depth_texture,
            depth_texture_view,
            framebuffers,
        }
    }

    pub fn create_render_pass(
        context: Arc<VulkanContext>,
        swapchain_properties: &SwapchainProperties,
        depth_format: vk::Format,
    ) -> RenderPass {
        let color_attachment_description = vk::AttachmentDescription::builder()
            .format(swapchain_properties.format.format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .build();

        let depth_attachment_description = vk::AttachmentDescription::builder()
            .format(depth_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .build();

        let attachment_descriptions = [color_attachment_description, depth_attachment_description];

        let color_attachment_reference = vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build();
        let color_attachment_references = [color_attachment_reference];

        let depth_attachment_reference = vk::AttachmentReference::builder()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .build();

        let subpass_description = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachment_references)
            .depth_stencil_attachment(&depth_attachment_reference)
            .build();
        let subpass_descriptions = [subpass_description];

        let subpass_dependency = vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            )
            .build();
        let subpass_dependencies = [subpass_dependency];

        let create_info = vk::RenderPassCreateInfo::builder()
            .attachments(&attachment_descriptions)
            .subpasses(&subpass_descriptions)
            .dependencies(&subpass_dependencies)
            .build();

        RenderPass::new(context, &create_info)
    }

    fn create_depth_texture(
        context: Arc<VulkanContext>,
        swapchain_extent: &vk::Extent2D,
        depth_format: vk::Format,
    ) -> Texture {
        let image_create_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width: swapchain_extent.width,
                height: swapchain_extent.height,
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

        let image_allocation_create_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::GpuOnly,
            ..Default::default()
        };
        Texture::new(context, &image_allocation_create_info, &image_create_info)
    }

    fn create_depth_texture_view(
        context: Arc<VulkanContext>,
        depth_texture: &Texture,
        depth_format: vk::Format,
    ) -> ImageView {
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
        ImageView::new(context.clone(), create_info)
    }

    fn create_framebuffers(
        context: Arc<VulkanContext>,
        swapchain: &Swapchain,
        depth_texture_view: &ImageView,
        render_pass: &RenderPass,
    ) -> Vec<Framebuffer> {
        swapchain
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
            .collect::<Vec<_>>()
    }
}
