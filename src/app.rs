use crate::{renderer::Renderer, vertex::Vertex};
use nalgebra_glm as glm;
use std::time::Instant;
use winit::{dpi::LogicalSize, Event, EventsLoop, VirtualKeyCode, Window, WindowEvent};

pub struct App {
    event_loop: EventsLoop,
    _window: Window, // Needs to live as long the event loop
    renderer: Renderer,
    should_exit: bool,
    dimensions: Option<[u32; 2]>,
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

        let vertices: [Vertex; 4] = [
            Vertex::new(
                glm::vec2(-0.5, -0.5),
                glm::vec3(1.0, 0.0, 0.0),
                glm::vec2(0.0, 0.0),
            ),
            Vertex::new(
                glm::vec2(0.5, -0.5),
                glm::vec3(0.0, 1.0, 0.0),
                glm::vec2(1.0, 0.0),
            ),
            Vertex::new(
                glm::vec2(0.5, 0.5),
                glm::vec3(0.0, 0.0, 1.0),
                glm::vec2(1.0, 1.0),
            ),
            Vertex::new(
                glm::vec2(-0.5, 0.5),
                glm::vec3(1.0, 1.0, 1.0),
                glm::vec2(0.0, 1.0),
            ),
        ];
        let indices: [u16; 6] = [0, 1, 2, 2, 3, 0];
        let renderer = Renderer::new(&window, [width, height], &vertices, &indices);

        App {
            event_loop,
            _window: window,
            renderer,
            should_exit: false,
            dimensions: Some([width, height]),
        }
    }

    pub fn run(&mut self) {
        log::debug!("Running application.");
        let start_time = Instant::now();
        loop {
            self.process_events();

            if self.should_exit {
                break;
            }

            self.renderer.step(self.dimensions, start_time);
        }

        self.renderer.wait_idle();
    }

    fn process_events(&mut self) {
        let mut dimensions: Option<[u32; 2]> = None;
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
                dimensions = Some([width as u32, height as u32]);
            }
            _ => {}
        });
        self.dimensions = dimensions;
        self.should_exit = should_exit;
        self.block_while_minimized();
    }

    fn block_while_minimized(&mut self) {
        if let Some(dimensions) = self.dimensions {
            let is_minimized = dimensions[0] == 0 || dimensions[1] == 0;
            if is_minimized {
                loop {
                    self.process_events();
                    let is_minimized = dimensions[0] == 0 || dimensions[1] == 0;
                    if !is_minimized {
                        break;
                    }
                }
            }
        }
    }
}
