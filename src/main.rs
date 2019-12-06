use ash::{version::DeviceV1_0, vk};
use std::time::Instant;
use winit::{dpi::LogicalSize, Event, VirtualKeyCode, WindowEvent};

mod app;
mod buffer;
mod context;
mod core;
mod render;
mod shader;
mod swapchain;
mod sync;
mod vertex;

use app::App;
use context::VulkanContext;
use std::sync::Arc;
use swapchain::VulkanSwapchain;
use sync::{SynchronizationSet, SynchronizationSetConstants};
use vertex::Vertex;

fn main() {
    env_logger::init();

    let vertices: [Vertex; 4] = [
        Vertex::new([-0.5, -0.5], [1.0, 0.0, 0.0]),
        Vertex::new([0.5, -0.5], [0.0, 1.0, 0.0]),
        Vertex::new([0.5, 0.5], [0.0, 0.0, 1.0]),
        Vertex::new([-0.5, 0.5], [1.0, 1.0, 1.0]),
    ];

    let indices: [u16; 6] = [0, 1, 2, 2, 3, 0];

    let mut app = App::new(800, 600, "Vulkan Tutorial");
    let context =
        Arc::new(VulkanContext::new(&app.window).expect("Failed to create VulkanContext"));
    let vulkan_swapchain = VulkanSwapchain::new(context.clone(), &vertices, &indices);
    let synchronization_set =
        SynchronizationSet::new(context.clone()).expect("Failed to create sync objects");

    let mut current_frame = 0;
    let start_time = Instant::now();

    let swapchain_khr_arr = [vulkan_swapchain.swapchain.swapchain_khr()];

    log::debug!("Running application.");
    let mut should_stop = false;
    loop {
        app.event_loop.poll_events(|event| match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            }
            | Event::WindowEvent {
                event:
                    WindowEvent::KeyboardInput {
                        input:
                            winit::KeyboardInput {
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    },
                ..
            } => should_stop = true,
            Event::WindowEvent {
                event:
                    WindowEvent::Resized(LogicalSize {
                        width: _width,
                        height: _height,
                    }),
                ..
            } => {
                // TODO: Handle resizing by recreating the swapchain
            }
            _ => {}
        });

        if should_stop {
            break;
        }

        let image_available_semaphore =
            *synchronization_set.image_available_semaphores()[current_frame].semaphore();
        let image_available_semaphores = [image_available_semaphore];

        let render_finished_semaphore =
            *synchronization_set.render_finished_semaphores()[current_frame].semaphore();
        let render_finished_semaphores = [render_finished_semaphore];

        let in_flight_fence = *synchronization_set.in_flight_fences()[current_frame].fence();
        let in_flight_fences = [in_flight_fence];

        unsafe {
            context
                .logical_device()
                .wait_for_fences(&in_flight_fences, true, std::u64::MAX)
                .unwrap();
            context
                .logical_device()
                .reset_fences(&in_flight_fences)
                .unwrap();
        }

        // Acquire the next image from the swapchain
        let image_index = unsafe {
            vulkan_swapchain
                .swapchain
                .swapchain()
                .acquire_next_image(
                    vulkan_swapchain.swapchain.swapchain_khr(),
                    std::u64::MAX,
                    image_available_semaphore,
                    vk::Fence::null(),
                )
                .unwrap()
                .0
        };
        let image_indices = [image_index];

        vulkan_swapchain.update_uniform_buffers(
            image_index,
            vulkan_swapchain.swapchain.properties(),
            start_time,
        );

        // Submit the command buffer
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers_to_use = [vulkan_swapchain.command_buffers[image_index as usize]];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(&image_available_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers_to_use)
            .signal_semaphores(&render_finished_semaphores)
            .build();
        let submit_info_arr = [submit_info];

        unsafe {
            context
                .logical_device()
                .queue_submit(
                    vulkan_swapchain.graphics_queue,
                    &submit_info_arr,
                    in_flight_fence,
                )
                .unwrap()
        };

        // Present the rendered image
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&render_finished_semaphores)
            .swapchains(&swapchain_khr_arr)
            .image_indices(&image_indices)
            .build();

        unsafe {
            vulkan_swapchain
                .swapchain
                .swapchain()
                .queue_present(vulkan_swapchain.present_queue, &present_info)
                .unwrap()
        };

        current_frame += (1 + current_frame) % SynchronizationSet::MAX_FRAMES_IN_FLIGHT as usize;
    }

    unsafe { context.logical_device().device_wait_idle().unwrap() };
}
