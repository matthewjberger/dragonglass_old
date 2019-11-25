use ash::{version::DeviceV1_0, vk};

pub fn create_shader_from_file<D>(path: &str, logical_device: &D) -> vk::ShaderModule
where
    D: DeviceV1_0,
{
    let mut shader_file = std::fs::File::open(path).expect("Failed to find shader file path");
    let shader_source = ash::util::read_spv(&mut shader_file)
        .expect("Failed to read SPIR-V shader source from bytes");
    let shader_create_info = vk::ShaderModuleCreateInfo::builder()
        .code(&shader_source)
        .build();
    unsafe {
        logical_device
            .create_shader_module(&shader_create_info, None)
            .expect("Failed to create shader module")
    }
}
