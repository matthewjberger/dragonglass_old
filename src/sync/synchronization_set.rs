use crate::{
    sync::{Fence, Semaphore},
    VulkanContext,
};
use ash::vk;
use std::sync::Arc;

pub trait SynchronizationSetConstants {
    // The maximum number of frames that can be rendered simultaneously
    const MAX_FRAMES_IN_FLIGHT: u32;
}

impl SynchronizationSetConstants for SynchronizationSet {
    const MAX_FRAMES_IN_FLIGHT: u32 = 2;
}

pub struct SynchronizationSet {
    image_available_semaphores: Vec<Semaphore>,
    render_finished_semaphores: Vec<Semaphore>,
    in_flight_fences: Vec<Fence>,
}

impl SynchronizationSet {
    pub fn new(context: Arc<VulkanContext>) -> Result<Self, Box<dyn std::error::Error>> {
        let mut image_available_semaphores = Vec::new();
        let mut render_finished_semaphores = Vec::new();
        let mut in_flight_fences = Vec::new();
        for _ in 0..SynchronizationSet::MAX_FRAMES_IN_FLIGHT {
            let image_available_semaphore = Semaphore::new(context.clone())?;
            image_available_semaphores.push(image_available_semaphore);

            let render_finished_semaphore = Semaphore::new(context.clone())?;
            render_finished_semaphores.push(render_finished_semaphore);

            let in_flight_fence = Fence::new(context.clone(), vk::FenceCreateFlags::SIGNALED)?;
            in_flight_fences.push(in_flight_fence);
        }

        Ok(SynchronizationSet {
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
        })
    }

    pub fn image_available_semaphores(&self) -> &[Semaphore] {
        &self.image_available_semaphores
    }

    pub fn render_finished_semaphores(&self) -> &[Semaphore] {
        &self.render_finished_semaphores
    }

    pub fn in_flight_fences(&self) -> &[Fence] {
        &self.in_flight_fences
    }
}
