use glob::glob;
use std::{
    error::Error,
    io,
    path::Path,
    process::{Command, Output},
};

type Result<T, E = Box<dyn Error>> = std::result::Result<T, E>;

const SHADER_COMPILER_NAME: &str = "glslangValidator";

pub fn compile_shaders(shader_glob: &str) -> Result<()> {
    for entry in glob(&shader_glob)? {
        if let Ok(shader_path) = entry {
            compile_shader(&shader_path)?;
        }
    }
    Ok(())
}

fn compile_shader(shader_path: &Path) -> Result<()> {
    let parent_name = shader_path
        .parent()
        .ok_or("Failed to get shader parent directory name")?;

    let file_name = shader_path.file_name().ok_or("Failed to get file_name")?;

    let output_name = file_name
        .to_str()
        .ok_or("Failed to convert file_name os_str to string")?
        .replace("glsl", "spv");

    println!("Compiling {:?} -> {:?}", file_name, output_name);
    let result = Command::new(SHADER_COMPILER_NAME)
        .current_dir(&parent_name)
        .arg("-V")
        .arg(&file_name)
        .arg("-o")
        .arg(output_name)
        .output();

    display_result(result);

    Ok(())
}

fn display_result(result: std::io::Result<Output>) {
    match result {
        Ok(output) if !output.status.success() => {
            eprint!(
                "Shader compilation output: {}",
                String::from_utf8(output.stdout).unwrap_or_else(|_| {
                    "Failed to convert stdout bytes to UTF-8 string".to_string()
                })
            );
            eprintln!("Failed to compile shader: {}", output.status)
        }
        Ok(_) => println!("Shader compilation succeeded"),
        Err(error) if error.kind() == io::ErrorKind::NotFound => panic!(
            "Failed to find the shader compiler program: '{}'",
            SHADER_COMPILER_NAME,
        ),
        Err(error) => eprintln!("Failed to compile shader: {}", error),
    }
}
