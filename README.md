# Dragonglass

Rendering gltf models with Vulkan and Rust

![Damaged Helmet Model](/screenshots/damaged_helmet.png?raw=true "Damaged Helmet GLTF Model")

## Running

```
# bash
RUST_LOG=dragonglass cargo run > logfile 2>&1

# fish
env RUST_LOG=dragonglass cargo run > logfile 2>&1

# powershell
$env:RUST_LOG="dragonglass"; cargo run
```

