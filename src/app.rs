use winit::{EventsLoop, Window};

pub struct App {
    pub event_loop: EventsLoop,
    pub window: Window,
}

impl App {
    pub fn new(width: u32, height: u32, title: &str) -> Self {
        // Initialize the window
        let event_loop = EventsLoop::new();
        let window = winit::WindowBuilder::new()
            .with_title(title)
            .with_dimensions((width, height).into())
            .build(&event_loop)
            .expect("Failed to create window.");

        App { event_loop, window }
    }
}
