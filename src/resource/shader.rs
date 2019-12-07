use crate::VulkanContext;
use ash::{version::DeviceV1_0, vk};
use std::sync::Arc;

// TODO: Add snafu errors

pub struct Shader {
    context: Arc<VulkanContext>,
    module: vk::ShaderModule,
}

impl Shader {
    pub fn from_file(context: Arc<VulkanContext>, path: &str) -> Self {
        let mut shader_file = std::fs::File::open(path).expect("Failed to find shader file path");
        let shader_source = ash::util::read_spv(&mut shader_file)
            .expect("Failed to read SPIR-V shader source from bytes");
        let shader_create_info = vk::ShaderModuleCreateInfo::builder()
            .code(&shader_source)
            .build();
        let module = unsafe {
            context
                .logical_device()
                .create_shader_module(&shader_create_info, None)
                .expect("Failed to create shader module")
        };
        Shader { module, context }
    }

    pub fn module(&self) -> vk::ShaderModule {
        self.module
    }
}

impl Drop for Shader {
    fn drop(&mut self) {
        unsafe {
            self.context
                .logical_device()
                .destroy_shader_module(self.module, None);
        }
    }
}
