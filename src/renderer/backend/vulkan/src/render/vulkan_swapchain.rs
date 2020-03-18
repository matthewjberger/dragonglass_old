use crate::{
    core::{Swapchain, SwapchainProperties, VulkanContext},
    render::{Framebuffer, RenderPass},
    resource::{CommandPool, ImageView, Texture},
};
use ash::vk;
use std::sync::Arc;

pub struct VulkanSwapchain {
    pub swapchain: Swapchain,
    pub render_pass: RenderPass,
    pub depth_texture: Texture,
    pub depth_texture_view: ImageView,
    pub color_texture: Texture,
    pub color_texture_view: ImageView,
    pub framebuffers: Vec<Framebuffer>,
}

impl VulkanSwapchain {
    pub fn new(
        context: Arc<VulkanContext>,
        dimensions: [u32; 2],
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
            Self::create_depth_texture(context.clone(), swapchain_extent, depth_format);

        Self::transition_depth_texture(&command_pool, &depth_texture, depth_format);

        let depth_texture_view =
            Self::create_depth_texture_view(context.clone(), &depth_texture, depth_format);

        let color_format = swapchain.properties().format.format;
        let color_texture =
            Self::create_color_texture(context.clone(), swapchain_extent, color_format);
        Self::transition_color_texture(&command_pool, &color_texture, color_format);
        let color_texture_view =
            Self::create_color_texture_view(context.clone(), &color_texture, color_format);

        let framebuffers = Self::create_framebuffers(
            context,
            &swapchain,
            &color_texture_view,
            &depth_texture_view,
            &render_pass,
        );

        VulkanSwapchain {
            swapchain,
            render_pass,
            depth_texture,
            depth_texture_view,
            color_texture,
            color_texture_view,
            framebuffers,
        }
    }

    pub fn create_render_pass(
        context: Arc<VulkanContext>,
        swapchain_properties: &SwapchainProperties,
        depth_format: vk::Format,
    ) -> RenderPass {
        let msaa_samples = context.max_usable_samples();

        let color_attachment_description = vk::AttachmentDescription::builder()
            .format(swapchain_properties.format.format)
            .samples(msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build();

        let depth_attachment_description = vk::AttachmentDescription::builder()
            .format(depth_format)
            .samples(msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .build();

        let resolve_attachment_description = vk::AttachmentDescription::builder()
            .format(swapchain_properties.format.format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .build();

        let attachment_descriptions = [
            color_attachment_description,
            depth_attachment_description,
            resolve_attachment_description,
        ];

        let color_attachment_reference = vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build();
        let color_attachment_references = [color_attachment_reference];

        let depth_attachment_reference = vk::AttachmentReference::builder()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .build();

        let resolve_attachment_description = vk::AttachmentReference::builder()
            .attachment(2)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build();
        let resolve_attachment_references = [resolve_attachment_description];

        let subpass_description = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachment_references)
            .resolve_attachments(&resolve_attachment_references)
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

    fn create_framebuffers(
        context: Arc<VulkanContext>,
        swapchain: &Swapchain,
        color_texture_view: &ImageView,
        depth_texture_view: &ImageView,
        render_pass: &RenderPass,
    ) -> Vec<Framebuffer> {
        swapchain
            .image_views()
            .iter()
            .map(|view| {
                [
                    color_texture_view.view(),
                    depth_texture_view.view(),
                    view.view(),
                ]
            })
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

    fn create_depth_texture(
        context: Arc<VulkanContext>,
        swapchain_extent: vk::Extent2D,
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
            .samples(context.max_usable_samples())
            .flags(vk::ImageCreateFlags::empty())
            .build();

        let image_allocation_create_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::GpuOnly,
            ..Default::default()
        };
        Texture::new(context, &image_allocation_create_info, &image_create_info)
    }

    fn transition_depth_texture(
        command_pool: &CommandPool,
        depth_texture: &Texture,
        depth_format: vk::Format,
    ) {
        let mut aspect_mask = vk::ImageAspectFlags::DEPTH;
        let has_stencil_component = depth_format == vk::Format::D32_SFLOAT_S8_UINT
            || depth_format == vk::Format::D24_UNORM_S8_UINT;

        if has_stencil_component {
            aspect_mask |= vk::ImageAspectFlags::STENCIL;
        }
        let barrier = vk::ImageMemoryBarrier::builder()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(depth_texture.image())
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            )
            .build();
        let barriers = [barrier];

        command_pool.transition_image_layout(
            &barriers,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        );
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
        ImageView::new(context, create_info)
    }

    fn create_color_texture(
        context: Arc<VulkanContext>,
        swapchain_extent: vk::Extent2D,
        color_format: vk::Format,
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
            .format(color_format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(
                vk::ImageUsageFlags::TRANSIENT_ATTACHMENT | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(context.max_usable_samples())
            .flags(vk::ImageCreateFlags::empty())
            .build();

        let image_allocation_create_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::GpuOnly,
            ..Default::default()
        };
        Texture::new(context, &image_allocation_create_info, &image_create_info)
    }

    fn transition_color_texture(
        command_pool: &CommandPool,
        color_texture: &Texture,
        color_format: vk::Format,
    ) {
        let mut aspect_mask = vk::ImageAspectFlags::COLOR;
        let has_stencil_component = color_format == vk::Format::D32_SFLOAT_S8_UINT
            || color_format == vk::Format::D24_UNORM_S8_UINT;

        if has_stencil_component {
            aspect_mask |= vk::ImageAspectFlags::STENCIL;
        }
        let barrier = vk::ImageMemoryBarrier::builder()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(color_texture.image())
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            )
            .build();
        let barriers = [barrier];

        command_pool.transition_image_layout(
            &barriers,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        );
    }

    fn create_color_texture_view(
        context: Arc<VulkanContext>,
        color_texture: &Texture,
        color_format: vk::Format,
    ) -> ImageView {
        let create_info = vk::ImageViewCreateInfo::builder()
            .image(color_texture.image())
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(color_format)
            .components(vk::ComponentMapping {
                r: vk::ComponentSwizzle::IDENTITY,
                g: vk::ComponentSwizzle::IDENTITY,
                b: vk::ComponentSwizzle::IDENTITY,
                a: vk::ComponentSwizzle::IDENTITY,
            })
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .build();
        ImageView::new(context, create_info)
    }
}
