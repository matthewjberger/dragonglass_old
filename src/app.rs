use crate::{renderer::Renderer, vertex::Vertex};
use std::time::Instant;
use winit::{dpi::LogicalSize, Event, EventsLoop, VirtualKeyCode, Window, WindowEvent};

pub struct App {
    event_loop: EventsLoop,
    _window: Window, // Needs to live as long the event loop
    renderer: Renderer,
    should_exit: bool,
    dimensions: [u32; 2],
    resize_requested: bool,
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
            Vertex::new([-0.5, -0.5], [1.0, 0.0, 0.0], [0.0, 0.0]),
            Vertex::new([0.5, -0.5], [0.0, 1.0, 0.0], [1.0, 0.0]),
            Vertex::new([0.5, 0.5], [0.0, 0.0, 1.0], [1.0, 1.0]),
            Vertex::new([-0.5, 0.5], [1.0, 1.0, 1.0], [0.0, 1.0]),
        ];
        let indices: [u16; 6] = [0, 1, 2, 2, 3, 0];
        let renderer = Renderer::new(&window, [width, height], &vertices, &indices);

        App {
            event_loop,
            _window: window,
            renderer,
            should_exit: false,
            dimensions: [width, height],
            resize_requested: false,
        }
    }

    pub fn run(&mut self) {
        log::debug!("Running application.");
        let start_time = Instant::now();
        loop {
            // TODO: Refactor this
            loop {
                self.process_events();
                let (width, height) = (self.dimensions[0], self.dimensions[1]);
                if width != 0 && height != 0 {
                    break;
                }
            }

            if self.should_exit {
                break;
            }

            self.renderer
                .step(self.dimensions, start_time, self.resize_requested);
            self.resize_requested = false;
        }

        self.renderer.wait_idle();
    }

    fn process_events(&mut self) {
        let extent = self.renderer.swapchain.properties().extent;
        let mut dimensions: [u32; 2] = [extent.width, extent.height];
        let mut should_exit = false;
        let mut resize_requested = false;
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
                resize_requested = true;
            }
            _ => {}
        });
        self.dimensions = dimensions;
        self.should_exit = should_exit;
        self.resize_requested = resize_requested;
    }
}
