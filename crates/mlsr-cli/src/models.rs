//! Deterministic, offline Flutter asset-package preparation.
use crate::Result;
use clap::Subcommand;
use mlsr::models::{
    catalog::{ModelCatalog, ModelDescriptor},
    supertonic3::{BUNDLE_MANIFEST_NAME, BundleManifest},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
};

#[derive(Subcommand)]
pub(crate) enum Models {
    /// Generate a local Flutter package with verified regular-file model assets.
    PrepareFlutter {
        /// YAML configuration; relative paths resolve beside this file.
        #[arg(long, default_value = "mlsr-models.yaml")]
        config: PathBuf,
    },
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    schema_version: u32,
    package: String,
    output: PathBuf,
    #[serde(default)]
    catalog: Option<PathBuf>,
    #[serde(default)]
    mlsr_package: Option<PathBuf>,
    models: Vec<Selection>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Selection {
    id: String,
    dart_name: String,
    source: PathBuf,
    #[serde(default)]
    manifest: Option<PathBuf>,
    #[serde(default)]
    platforms: Vec<String>,
}
#[derive(Clone, Serialize)]
struct AssetFile {
    path: String,
    sha256: String,
    size_bytes: u64,
}
#[derive(Serialize)]
struct PackageModel {
    id: String,
    asset_path: String,
    platforms: Vec<String>,
    files: Vec<AssetFile>,
}
#[derive(Serialize)]
struct PackageIndex {
    schema_version: u32,
    package: String,
    models: Vec<PackageModel>,
}
#[derive(Serialize)]
struct CatalogOutput {
    schema_version: u32,
    models: Vec<ModelDescriptor>,
}

pub(crate) fn run(command: Models) -> Result<()> {
    match command {
        Models::PrepareFlutter { config } => println!("Prepared {}", prepare(&config)?.display()),
    }
    Ok(())
}
fn limited(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let mut data = Vec::new();
    File::open(path)?.take(limit + 1).read_to_end(&mut data)?;
    if data.len() as u64 > limit {
        return Err(format!("{} exceeds its size limit", path.display()).into());
    }
    Ok(data)
}
fn path(base: &Path, value: &Path) -> Result<PathBuf> {
    if let Some(value) = value.to_str().and_then(|s| s.strip_prefix("~/")) {
        let home = std::env::var_os("HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .ok_or("cannot expand ~/ without a home directory")?;
        return Ok(PathBuf::from(home).join(value));
    }
    Ok(base.join(value))
}
fn dart_name(name: &str) -> bool {
    const RESERVED: &[&str] = &[
        "abstract",
        "as",
        "assert",
        "async",
        "await",
        "base",
        "break",
        "case",
        "catch",
        "class",
        "const",
        "continue",
        "covariant",
        "default",
        "deferred",
        "do",
        "dynamic",
        "else",
        "enum",
        "export",
        "extends",
        "extension",
        "external",
        "factory",
        "false",
        "final",
        "finally",
        "for",
        "Function",
        "get",
        "hide",
        "if",
        "implements",
        "import",
        "in",
        "interface",
        "is",
        "late",
        "library",
        "mixin",
        "new",
        "null",
        "of",
        "on",
        "operator",
        "part",
        "required",
        "rethrow",
        "return",
        "sealed",
        "set",
        "show",
        "static",
        "super",
        "switch",
        "sync",
        "this",
        "throw",
        "true",
        "try",
        "type",
        "typedef",
        "var",
        "void",
        "when",
        "while",
        "with",
        "yield",
    ];
    !name.is_empty()
        && name.len() <= 64
        && name.as_bytes()[0].is_ascii_lowercase()
        && name.bytes().all(|b| b.is_ascii_alphanumeric())
        && !RESERVED.contains(&name)
}
fn package_name(name: &str) -> bool {
    (3..=64).contains(&name.len())
        && name.as_bytes()[0].is_ascii_lowercase()
        && name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
        && !["mlsr", "mlsr_flutter", "flutter"].contains(&name)
}
fn write(root: &Path, relative: &str, bytes: &[u8]) -> Result<()> {
    let path = root.join(relative);
    fs::create_dir_all(path.parent().ok_or("file has no parent")?)?;
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    Ok(())
}
fn copy_payload(source: &Path, target: &Path, expected: u64) -> Result<()> {
    let file = File::open(source)?;
    if !file.metadata()?.is_file() {
        return Err("source payload is not a file".into());
    }
    fs::create_dir_all(target.parent().ok_or("payload has no parent")?)?;
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(target)?;
    let count = std::io::copy(
        &mut file.take(expected.checked_add(1).ok_or("payload size overflow")?),
        &mut output,
    )?;
    if count != expected {
        return Err(format!("source size changed: {}", source.display()).into());
    }
    Ok(())
}
fn quoted(value: &str) -> String {
    serde_json::to_string(value)
        .expect("string serialization")
        .replace('$', "\\$")
}

fn prepare(config_path: &Path) -> Result<PathBuf> {
    let config_path = fs::canonicalize(config_path)?;
    let base = config_path.parent().ok_or("config has no parent")?;
    // Parse through Value to reject duplicate YAML keys before struct decoding.
    let value: serde_yaml_ng::Value = serde_yaml_ng::from_slice(&limited(&config_path, 65536)?)?;
    let mut config: Config = serde_yaml_ng::from_value(value)?;
    if config.schema_version != 1
        || !package_name(&config.package)
        || config.models.is_empty()
        || config.models.len() > 64
    {
        return Err("expected schema 1, a valid distinct package name, and 1..=64 models".into());
    }
    let mut names = BTreeSet::new();
    let mut ids = BTreeSet::new();
    for selection in &config.models {
        if !dart_name(&selection.dart_name)
            || !names.insert(selection.dart_name.clone())
            || !ids.insert(selection.id.clone())
        {
            return Err("model IDs and valid Dart names must be unique".into());
        }
    }
    config.models.sort_by(|a, b| a.id.cmp(&b.id));
    let catalog = match &config.catalog {
        Some(p) => ModelCatalog::parse(&limited(&path(base, p)?, 65536)?)?,
        None => ModelCatalog::builtin(),
    };
    let output = path(base, &config.output)?;
    let parent = output.parent().ok_or("output has no parent")?;
    let name = output
        .file_name()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .ok_or("output needs a directory name")?;
    fs::create_dir_all(parent)?;
    let parent = fs::canonicalize(parent)?;
    let output = parent.join(name);
    let _lock = package_lock(&parent.join(format!(".{name}.mlsr-prepare.lock")))?;
    let staging = tempfile::Builder::new()
        .prefix(".mlsr-prepare-")
        .tempdir_in(&parent)?;
    let mut descriptors = Vec::new();
    let mut entries = Vec::new();
    let mut asset_entries = Vec::new();
    for selection in &config.models {
        let descriptor = catalog
            .models()
            .find(|m| m.id() == selection.id)
            .ok_or_else(|| format!("unknown model {}", selection.id))?;
        let mut platforms = if selection.platforms.is_empty() {
            descriptor.platforms().map(str::to_owned).collect()
        } else {
            selection.platforms.clone()
        };
        platforms.sort();
        if platforms.windows(2).any(|p| p[0] == p[1]) {
            return Err("duplicate requested platform".into());
        }
        for platform in &platforms {
            catalog.resolve(&selection.id, platform)?;
        }
        let source = fs::canonicalize(path(base, &selection.source)?)?;
        if source.starts_with(&output) || output.starts_with(&source) {
            return Err("output and source directories must not overlap".into());
        }
        let manifest_path = match &selection.manifest {
            Some(p) => path(base, p)?,
            None => source.join(BUNDLE_MANIFEST_NAME),
        };
        let manifest = BundleManifest::read(manifest_path)?;
        descriptor.verify_manifest(&manifest)?;
        manifest.verify(&source)?;
        let model_root = staging.path().join(descriptor.asset_path());
        fs::create_dir_all(&model_root)?;
        write(&model_root, BUNDLE_MANIFEST_NAME, manifest.bytes())?;
        let mut files = vec![AssetFile {
            path: BUNDLE_MANIFEST_NAME.into(),
            sha256: manifest.sha256().into(),
            size_bytes: manifest.bytes().len() as u64,
        }];
        for file in manifest.files() {
            copy_payload(
                &source.join(&file.path),
                &model_root.join(&file.path),
                file.size_bytes,
            )?;
            files.push(AssetFile {
                path: file.path.clone(),
                sha256: file.sha256.clone(),
                size_bytes: file.size_bytes,
            });
        }
        // Reverify the produced bytes, catching source mutation during copying.
        manifest.verify(&model_root)?;
        files.sort_by(|a, b| a.path.cmp(&b.path));
        let flutter_platforms: BTreeSet<_> = platforms
            .iter()
            .map(|p| p.split('-').next().expect("validated platform"))
            .collect();
        for file in &files {
            asset_entries.push(serde_json::json!({"path": format!("{}/{}", descriptor.asset_path(), file.path), "platforms": flutter_platforms}));
        }
        entries.push(PackageModel {
            id: selection.id.clone(),
            asset_path: descriptor.asset_path().into(),
            platforms,
            files,
        });
        descriptors.push(descriptor.clone());
    }
    let catalog_yaml = ModelCatalog::parse(
        serde_yaml_ng::to_string(&CatalogOutput {
            schema_version: 1,
            models: descriptors,
        })?
        .as_bytes(),
    )?
    .to_yaml()?;
    write(staging.path(), "catalog.yaml", catalog_yaml.as_bytes())?;
    let index = PackageIndex {
        schema_version: 1,
        package: config.package.clone(),
        models: entries,
    };
    write(
        staging.path(),
        "package-index.yaml",
        serde_yaml_ng::to_string(&index)?.as_bytes(),
    )?;
    write(
        staging.path(),
        ".mlsr-generated",
        b"MLSR Flutter model assets v1\n",
    )?;
    for (relative, bytes) in [
        ("LICENSE", include_bytes!("../../../LICENSE").as_slice()),
        (
            "notices/supertonic3-OpenRAIL-M.txt",
            include_bytes!("../../../model-manifests/licenses/supertonic3-OpenRAIL-M.txt")
                .as_slice(),
        ),
        (
            "notices/supertonic-MIT.txt",
            include_bytes!("../../../model-manifests/licenses/supertonic-MIT.txt").as_slice(),
        ),
        (
            "notices/SOURCES.md",
            include_bytes!("../../../model-manifests/licenses/README.md").as_slice(),
        ),
        (
            "notices/MLSR-THIRD-PARTY.md",
            include_bytes!("../../mlsr/THIRD_PARTY_NOTICES.md").as_slice(),
        ),
    ] {
        write(staging.path(), relative, bytes)?;
        asset_entries.push(serde_json::json!(relative));
    }
    for relative in ["catalog.yaml", "package-index.yaml"] {
        asset_entries.push(serde_json::json!(relative));
    }
    let mlsr_dependency = match &config.mlsr_package {
        Some(p) => {
            let dependency = fs::canonicalize(path(base, p)?)?;
            if !dependency.join("pubspec.yaml").is_file() {
                return Err("mlsr_package is not a Dart package".into());
            }
            serde_json::json!({"path": dependency.to_str().ok_or("Dart dependency path is not UTF-8")?})
        }
        None => serde_json::json!("^0.1.0"),
    };
    let pubspec = serde_json::json!({
        "name": config.package, "description": "Verified MLSR model assets generated from a pinned catalog.", "version": "0.1.0", "publish_to": "none",
        "environment": {"sdk": ">=3.11.0 <4.0.0", "flutter": ">=3.41.0"},
        "dependencies": {"flutter": {"sdk": "flutter"}, "mlsr": mlsr_dependency},
        "flutter": {"assets": asset_entries},
    });
    write(
        staging.path(),
        "pubspec.yaml",
        serde_yaml_ng::to_string(&pubspec)?.as_bytes(),
    )?;
    let mut dart = String::from(
        "// Generated by mlsr models prepare-flutter. Do not edit.\nimport 'package:mlsr/mlsr.dart';\n\nabstract final class AppModels {\n",
    );
    dart.push_str(&format!(
        "  static const _catalog = {};\n",
        quoted(&catalog_yaml)
    ));
    for (selection, model) in config.models.iter().zip(&index.models) {
        dart.push_str(&format!("  static const {} = BundledModel(\n    package: {}, id: {}, assetPath: {}, catalogYaml: _catalog,\n    platforms: [{}],\n    files: [\n", selection.dart_name, quoted(&config.package), quoted(&model.id), quoted(&model.asset_path), model.platforms.iter().map(|p| quoted(p)).collect::<Vec<_>>().join(", ")));
        for file in &model.files {
            dart.push_str(&format!(
                "      ModelAssetFile(path: {}, sha256: {}, sizeBytes: {}),\n",
                quoted(&file.path),
                quoted(&file.sha256),
                file.size_bytes
            ));
        }
        dart.push_str("    ],\n  );\n");
    }
    dart.push_str("}\n");
    write(staging.path(), "lib/app_models.dart", dart.as_bytes())?;
    write(staging.path(), "README.md", b"# Generated MLSR model assets\n\nAdd this local package as a path dependency before Flutter pub get.\nImport lib/app_models.dart and open AppModels selections through mlsr_flutter.\nModel weights retain their upstream licenses in notices/.\nPreparation never downloads or modifies source files.\n")?;
    if output.symlink_metadata().is_ok() {
        if output.symlink_metadata()?.file_type().is_symlink()
            || !output.join(".mlsr-generated").is_file()
            || tree_hashes(&output)? != tree_hashes(staging.path())?
        {
            return Err(format!("output already exists with different contents: {}; choose a fresh output directory", output.display()).into());
        }
        return Ok(output);
    }
    fs::rename(staging.path(), &output)?;
    Ok(output)
}
fn tree_hashes(root: &Path) -> Result<BTreeMap<PathBuf, (u64, String)>> {
    fn visit(root: &Path, dir: &Path, result: &mut BTreeMap<PathBuf, (u64, String)>) -> Result<()> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            // Pub/analyze may create these beside generated package sources.
            // They are never model assets and do not change preparation identity.
            if dir == root
                && [".dart_tool", "build", "pubspec.lock"]
                    .iter()
                    .any(|name| entry.file_name() == *name)
            {
                continue;
            }
            if entry.file_type()?.is_dir() {
                visit(root, &entry.path(), result)?;
            } else if entry.file_type()?.is_file() {
                let mut file = File::open(entry.path())?;
                let mut hash = Sha256::new();
                let bytes = std::io::copy(&mut file, &mut hash)?;
                result.insert(
                    entry.path().strip_prefix(root)?.into(),
                    (bytes, format!("{:x}", hash.finalize())),
                );
            } else {
                return Err("generated package contains non-regular entries".into());
            }
        }
        Ok(())
    }
    let mut result = BTreeMap::new();
    visit(root, root, &mut result)?;
    Ok(result)
}
fn package_lock(path: &Path) -> Result<File> {
    if let Ok(metadata) = path.symlink_metadata()
        && !metadata.file_type().is_file()
    {
        return Err("package lock is not a regular file".into());
    }
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)?;
    #[cfg(any(unix, windows))]
    {
        fs2::FileExt::try_lock_exclusive(&file)
            .map_err(|e| format!("package preparation is locked: {e}"))?;
        Ok(file)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = file;
        Err("package locking is unsupported on this platform".into())
    }
}

#[cfg(test)]
mod tests;
