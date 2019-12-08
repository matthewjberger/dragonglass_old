use ash::{version::DeviceV1_0, vk};
use std::time::Instant;
use winit::{dpi::LogicalSize, Event, EventsLoop, VirtualKeyCode, Window, WindowEvent};

use crate::{
    context::VulkanContext,
    render_state::RenderState,
    sync::{SynchronizationSet, SynchronizationSetConstants},
    vertex::Vertex,
};
use std::sync::Arc;

pub struct App {
    event_loop: EventsLoop,
    _window: Window, // Needs to live as long the event loop
    context: Arc<VulkanContext>,
    render_state: RenderState,
    synchronization_set: SynchronizationSet,
    should_exit: bool,
    dimensions: [u32; 2],
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
            Vertex::new([-0.5, -0.5], [1.0, 0.0, 0.0], [0.0, 0.0]),
            Vertex::new([0.5, -0.5], [0.0, 1.0, 0.0], [1.0, 0.0]),
            Vertex::new([0.5, 0.5], [0.0, 0.0, 1.0], [1.0, 1.0]),
            Vertex::new([-0.5, 0.5], [1.0, 1.0, 1.0], [0.0, 1.0]),
        ];
        let indices: [u16; 6] = [0, 1, 2, 2, 3, 0];
        let render_state = RenderState::new(context.clone(), [width, height], &vertices, &indices);
        let synchronization_set =
            SynchronizationSet::new(context.clone()).expect("Failed to create sync objects");

        App {
            event_loop,
            _window: window,
            context,
            render_state,
            synchronization_set,
            should_exit: false,
            dimensions: [width, height],
        }
    }

    pub fn run(&mut self) {
        log::debug!("Running application.");

        let mut current_frame = 0;
        let start_time = Instant::now();
        loop {
            // TODO: Refactor this
            loop {
                self.process_events();
                let (width, height) = (self.dimensions[0], self.dimensions[1]);
                let is_minimized = width <= 0 || height <= 0;
                if !is_minimized {
                    break;
                }
            }

            if self.should_exit {
                break;
            }

            let current_frame_synchronization = self
                .synchronization_set
                .current_frame_synchronization(current_frame);

            self.context
                .logical_device()
                .wait_for_fence(&current_frame_synchronization);

            // Acquire the next image from the swapchain
            let image_index_result = self.render_state.swapchain.acquire_next_image(
                current_frame_synchronization.image_available(),
                vk::Fence::null(),
            );

            let image_index = match image_index_result {
                Ok((image_index, _)) => image_index,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.render_state.recreate_swapchain(self.dimensions);
                    continue;
                }
                Err(error) => panic!("Error while acquiring next image. Cause: {}", error),
            };
            let image_indices = [image_index];

            self.context
                .logical_device()
                .reset_fence(&current_frame_synchronization);

            self.render_state.update_uniform_buffers(
                image_index,
                self.render_state.swapchain.properties(),
                start_time,
            );

            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            self.render_state.command_pool.submit_command_buffer(
                image_index as usize,
                self.render_state.graphics_queue,
                &wait_stages,
                &current_frame_synchronization,
            );

            let swapchain_presentation_result = self.render_state.swapchain.present_rendered_image(
                &current_frame_synchronization,
                &image_indices,
                self.render_state.present_queue,
            );

            match swapchain_presentation_result {
                Ok(is_suboptimal) if is_suboptimal => {
                    self.render_state.recreate_swapchain(self.dimensions);
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.render_state.recreate_swapchain(self.dimensions);
                }
                Err(error) => panic!("Failed to present queue. Cause: {}", error),
                _ => {}
            }

            current_frame +=
                (1 + current_frame) % SynchronizationSet::MAX_FRAMES_IN_FLIGHT as usize;
        }

        unsafe {
            self.context
                .logical_device()
                .logical_device()
                .device_wait_idle()
                .unwrap()
        };
    }

    fn process_events(&mut self) {
        let extent = self.render_state.swapchain.properties().extent;
        let mut dimensions: [u32; 2] = [extent.width, extent.height];
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
                event: WindowEvent::Resized(LogicalSize { width, height }),
                ..
            } => {
                dimensions = [width as u32, height as u32];
            }
            _ => {}
        });
        self.should_exit = should_exit;
        self.dimensions = dimensions;
    }
}
