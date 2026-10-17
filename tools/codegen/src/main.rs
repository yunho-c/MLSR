use std::{env, error::Error, fs, path::Path};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let check = match args.as_slice() {
        [] => false,
        [arg] if arg == "--check" => true,
        _ => return Err("usage: cargo run -p mlsr-codegen -- [--check]".into()),
    };
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let ffi = root.join("crates/mlsr-ffi");
    let config = cbindgen::Config::from_file(ffi.join("cbindgen.toml"))?;
    let bindings = cbindgen::Builder::new()
        .with_crate(&ffi)
        .with_config(config)
        .generate()?;
    let mut generated = Vec::new();
    bindings.write(&mut generated);
    let header = root.join("include/mlsr.h");
    if check {
        if fs::read(&header)? != generated {
            return Err("include/mlsr.h is stale; run cargo run -p mlsr-codegen".into());
        }
        println!("include/mlsr.h is current");
    } else {
        fs::create_dir_all(header.parent().ok_or("missing include directory")?)?;
        fs::write(&header, generated)?;
        println!("Generated include/mlsr.h");
    }
    Ok(())
}
