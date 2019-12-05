use crate::core::{surface::surface_extension_names, DebugLayer, LayerNameVec};
use ash::{
    extensions::ext::DebugUtils,
    version::{EntryV1_0, InstanceV1_0},
    vk, vk_make_version,
};
use snafu::ResultExt;
use std::ffi::{CStr, CString};

use snafu::Snafu;

type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, Snafu)]
#[snafu(visibility = "pub(crate)")]
pub enum Error {
    #[snafu(display("Failed to create entry: {}", source))]
    EntryLoading { source: ash::LoadingError },

    #[snafu(display("Failed to create instance: {}", source))]
    InstanceCreation { source: ash::InstanceError },

    #[snafu(display("Failed to create a c-string from the application name: {}", source))]
    AppNameCreation { source: std::ffi::NulError },

    #[snafu(display("Failed to create a c-string from the engine name: {}", source))]
    EngineNameCreation { source: std::ffi::NulError },
}

trait ApplicationDescription {
    const APPLICATION_NAME: &'static str;
    const APPLICATION_VERSION: u32;
    const API_VERSION: u32;
    const ENGINE_VERSION: u32;
    const ENGINE_NAME: &'static str;
}

impl ApplicationDescription for Instance {
    const APPLICATION_NAME: &'static str = "Vulkan Tutorial";
    const APPLICATION_VERSION: u32 = vk_make_version!(1, 0, 0);
    const API_VERSION: u32 = vk_make_version!(1, 0, 0);
    const ENGINE_VERSION: u32 = vk_make_version!(1, 0, 0);
    const ENGINE_NAME: &'static str = "Sepia Engine";
}

pub struct Instance {
    entry: ash::Entry,
    instance: ash::Instance,
}

impl Instance {
    pub fn new() -> Result<Self> {
        let entry = ash::Entry::new().context(EntryLoading)?;
        Self::check_required_layers_supported(&entry);
        let app_info = Self::build_application_creation_info()?;
        let instance_extensions = Self::required_instance_extension_names();
        let layer_name_vec = Self::required_layers();
        let layer_name_pointers = layer_name_vec.layer_name_pointers();
        let instance_create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&instance_extensions)
            .enabled_layer_names(&layer_name_pointers);
        let instance = unsafe {
            entry
                .create_instance(&instance_create_info, None)
                .context(InstanceCreation)?
        };
        Ok(Instance { entry, instance })
    }

    pub fn entry(&self) -> &ash::Entry {
        &self.entry
    }

    pub fn instance(&self) -> &ash::Instance {
        &self.instance
    }

    fn build_application_creation_info() -> Result<vk::ApplicationInfo> {
        let app_name = CString::new(Instance::APPLICATION_NAME).context(AppNameCreation)?;
        let engine_name = CString::new(Instance::ENGINE_NAME).context(EngineNameCreation)?;
        let app_info = vk::ApplicationInfo::builder()
            .application_name(&app_name)
            .engine_name(&engine_name)
            .api_version(Instance::API_VERSION)
            .application_version(Instance::APPLICATION_VERSION)
            .engine_version(Instance::ENGINE_VERSION)
            .build();
        Ok(app_info)
    }

    fn required_instance_extension_names() -> Vec<*const i8> {
        let mut instance_extension_names = surface_extension_names();
        if DebugLayer::validation_layers_enabled() {
            instance_extension_names.push(DebugUtils::name().as_ptr());
        }
        instance_extension_names
    }

    pub fn required_layers() -> LayerNameVec {
        let mut layer_name_vec = LayerNameVec::new();
        if DebugLayer::validation_layers_enabled() {
            layer_name_vec
                .layer_names
                // TODO: Improve naming here
                .extend(DebugLayer::debug_layer_names().layer_names);
        }
        layer_name_vec
    }

    fn check_required_layers_supported(entry: &ash::Entry) {
        let layer_name_vec = Self::required_layers();
        for layer_name in layer_name_vec.layer_names.iter() {
            let all_layers_supported = entry
                .enumerate_instance_layer_properties()
                .expect("Couldn't enumerate instance layer properties")
                .iter()
                .any(|layer| {
                    let name = unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) };
                    let name = name.to_str().expect("Failed to get layer name pointer");
                    (*layer_name).name() == name
                });

            if !all_layers_supported {
                panic!("Validation layer not supported: {}", layer_name.name());
            }
        }
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}
