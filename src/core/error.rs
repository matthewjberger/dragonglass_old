use snafu::Snafu;

pub type Result<T, E = CoreError> = std::result::Result<T, E>;

#[derive(Debug, Snafu)]
#[snafu(visibility = "pub(crate)")]
pub enum CoreError {
    #[snafu(display("Failed to create the physical device"))]
    PhysicalDeviceCreation,

    #[snafu(display("Failed to create entry: {}", source))]
    EntryLoading { source: ash::LoadingError },

    #[snafu(display("Failed to create instance: {}", source))]
    InstanceCreation { source: ash::InstanceError },

    #[snafu(display("Failed to create a c-string from the application name: {}", source))]
    AppNameCreation { source: std::ffi::NulError },

    #[snafu(display("Failed to create a c-string from the engine name: {}", source))]
    EngineNameCreation { source: std::ffi::NulError },
}
