use support::shader_compilation::{compile_shaders, Result};

fn main() -> Result<()> {
    let shader_directory = "../../../../examples/assets/shaders";
    let shader_glob = shader_directory.to_owned() + "/**/*.glsl";
    compile_shaders(&shader_glob)
}
