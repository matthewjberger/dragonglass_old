use crate::core::VulkanContext;
use ash::{version::DeviceV1_0, vk};
use snafu::{ResultExt, Snafu};
use std::{ffi::CStr, sync::Arc};

type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, Snafu)]
#[snafu(visibility = "pub(crate)")]
pub enum Error {
    #[snafu(display("Failed to find shader file path '{}': {}", path, source))]
    FindShaderFilePath {
        path: String,
        source: std::io::Error,
    },

    #[snafu(display("Failed to read SPIR-V shader source from bytes: {}", source))]
    ReadShaderSourceBytes { source: std::io::Error },

    #[snafu(display("Failed to create shader module: {}", source))]
    CreateShaderModule { source: ash::vk::Result },
}

pub struct Shader {
    context: Arc<VulkanContext>,
    module: vk::ShaderModule,
    state_info: vk::PipelineShaderStageCreateInfo,
}

impl Shader {
    // TODO: Refactor this to have less parameters
    pub fn from_file(
        context: Arc<VulkanContext>,
        path: &str,
        flags: vk::ShaderStageFlags,
        entry_point_name: &CStr,
    ) -> Result<Self> {
        let mut shader_file = std::fs::File::open(path).context(FindShaderFilePath { path })?;
        let shader_source = ash::util::read_spv(&mut shader_file).context(ReadShaderSourceBytes)?;
        let shader_create_info = vk::ShaderModuleCreateInfo::builder()
            .code(&shader_source)
            .build();
        let module = unsafe {
            context
                .logical_device()
                .logical_device()
                .create_shader_module(&shader_create_info, None)
                .context(CreateShaderModule)?
        };

        let state_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(flags)
            .module(module)
            .name(entry_point_name)
            .build();

        let shader = Shader {
            module,
            context,
            state_info,
        };

        Ok(shader)
    }

    pub fn state_info(&self) -> vk::PipelineShaderStageCreateInfo {
        self.state_info
    }
}

impl Drop for Shader {
    fn drop(&mut self) {
        unsafe {
            self.context
                .logical_device()
                .logical_device()
                .destroy_shader_module(self.module, None);
        }
    }
}
