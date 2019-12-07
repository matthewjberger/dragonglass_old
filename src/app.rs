use ash::{version::DeviceV1_0, vk};
use std::time::Instant;
use winit::{dpi::LogicalSize, Event, EventsLoop, VirtualKeyCode, Window, WindowEvent};

use crate::context::VulkanContext;
use crate::render_state::RenderState;
use crate::sync::{SynchronizationSet, SynchronizationSetConstants};
use crate::vertex::Vertex;
use std::sync::Arc;

pub struct App {
    event_loop: EventsLoop,
    _window: Window, // Needs to live as long the event loop
    context: Arc<VulkanContext>,
    render_state: RenderState,
    synchronization_set: SynchronizationSet,
    should_exit: bool,
}

impl App {
    pub fn new(width: u32, height: u32, title: &str) -> Self {
        log::debug!("Initializing application.");
        let event_loop = EventsLoop::new();
        let window = winit::WindowBuilder::new()
            .with_title(title)
            .with_dimensions((width, height).into())
            .build(&event_loop)
            .expect("Failed to create window.");

        let context =
            Arc::new(VulkanContext::new(&window).expect("Failed to create VulkanContext"));

        let vertices: [Vertex; 4] = [
            Vertex::new([-0.5, -0.5], [1.0, 0.0, 0.0]),
            Vertex::new([0.5, -0.5], [0.0, 1.0, 0.0]),
            Vertex::new([0.5, 0.5], [0.0, 0.0, 1.0]),
            Vertex::new([-0.5, 0.5], [1.0, 1.0, 1.0]),
        ];
        let indices: [u16; 6] = [0, 1, 2, 2, 3, 0];
        let render_state = RenderState::new(context.clone(), &vertices, &indices);
        let synchronization_set =
            SynchronizationSet::new(context.clone()).expect("Failed to create sync objects");

        App {
            event_loop,
            _window: window,
            context,
            render_state,
            synchronization_set,
            should_exit: false,
        }
    }

    pub fn run(&mut self) {
        log::debug!("Running application.");
        let mut current_frame = 0;
        let start_time = Instant::now();
        let swapchain_khr_arr = [self.render_state.swapchain.swapchain_khr()];
        loop {
            self.process_events();

            if self.should_exit {
                break;
            }

            let current_frame_synchronization = self
                .synchronization_set
                .current_frame_synchronization(current_frame);
            let image_available_semaphores = [current_frame_synchronization.image_available()];
            let render_finished_semaphores = [current_frame_synchronization.render_finished()];
            let in_flight_fences = [current_frame_synchronization.in_flight()];

            unsafe {
                self.context
                    .logical_device()
                    .wait_for_fences(&in_flight_fences, true, std::u64::MAX)
                    .unwrap();
                self.context
                    .logical_device()
                    .reset_fences(&in_flight_fences)
                    .unwrap();
            }

            // Acquire the next image from the swapchain
            let image_index = unsafe {
                self.render_state
                    .swapchain
                    .swapchain()
                    .acquire_next_image(
                        self.render_state.swapchain.swapchain_khr(),
                        std::u64::MAX,
                        current_frame_synchronization.image_available(),
                        vk::Fence::null(),
                    )
                    .unwrap()
                    .0
            };
            let image_indices = [image_index];

            self.render_state.update_uniform_buffers(
                image_index,
                self.render_state.swapchain.properties(),
                start_time,
            );

            // Submit the command buffer
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let command_buffers_to_use =
                [self.render_state.command_pool.command_buffers()[image_index as usize]];
            let submit_info = vk::SubmitInfo::builder()
                .wait_semaphores(&image_available_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&command_buffers_to_use)
                .signal_semaphores(&render_finished_semaphores)
                .build();
            let submit_info_arr = [submit_info];

            unsafe {
                self.context
                    .logical_device()
                    .queue_submit(
                        self.render_state.graphics_queue,
                        &submit_info_arr,
                        current_frame_synchronization.in_flight(),
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
                self.render_state
                    .swapchain
                    .swapchain()
                    .queue_present(self.render_state.present_queue, &present_info)
                    .unwrap()
            };

            current_frame +=
                (1 + current_frame) % SynchronizationSet::MAX_FRAMES_IN_FLIGHT as usize;
        }

        unsafe { self.context.logical_device().device_wait_idle().unwrap() };
    }

    fn process_events(&mut self) {
        let mut should_exit = false;
        self.event_loop.poll_events(|event| match event {
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
            } => should_exit = true,
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

        self.should_exit = should_exit;
    }
}
