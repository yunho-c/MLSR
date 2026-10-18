use crate::{
    error::{Error, Result},
    runtime::{HostTensor, TensorData},
};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
};

const BASELINE: &str = include_str!("../../../../../model-manifests/supertonic3-coreml.yaml");
/// Filename at the root of a self-contained model bundle.
pub const BUNDLE_MANIFEST_NAME: &str = "mlsr-bundle.yaml";
const MANIFEST_LIMIT: u64 = 1024 * 1024;
const METADATA_LIMIT: u64 = 4 * 1024 * 1024;
const VARIANTS: &str =
    include_str!("../../../../../model-manifests/supertonic3-coreml-variants.yaml");

// This description is compiled into the library, not read from untrusted input.
// Provenance/routing documentation is retained in YAML; code owns routing policy.
#[derive(Deserialize)]
struct VariantManifest {
    base_manifest_sha256: String,
    artifacts: BTreeMap<String, Artifact>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: u32,
    id: String,
    model: String,
    model_version: String,
    source: Source,
    provenance: Value,
    runtime: Value,
    contract: Value,
    integrity: Value,
    artifacts: BTreeMap<String, Artifact>,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Source {
    repository: String,
    revision: String,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Artifact {
    kind: String,
    path: String,
    #[serde(default)]
    precision: Value,
    #[serde(default)]
    latent_bucket: Option<usize>,
    #[serde(default)]
    inputs: Value,
    #[serde(default)]
    outputs: Value,
    files: Vec<LockedFile>,
}
/// One immutable payload file named by a validated bundle manifest.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LockedFile {
    /// Validated path relative to the model bundle root.
    pub path: String,
    /// Lowercase SHA-256 of the exact payload bytes.
    pub sha256: String,
    /// Expected payload length.
    pub size_bytes: u64,
}

/// Parsed model contract retaining the original YAML bytes and their identity.
/// Parsing checks interfaces and paths; verification additionally checks payloads.
#[derive(Debug)]
pub struct BundleManifest {
    bytes: Vec<u8>,
    sha256: String,
    parsed: Manifest,
}
impl BundleManifest {
    /// Parse a bounded YAML contract. No model files or runtime are opened.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() as u64 > MANIFEST_LIMIT {
            return Err(Error::Manifest("manifest exceeds 1 MiB".into()));
        }
        Ok(Self {
            parsed: parse_manifest(bytes)?,
            sha256: format!("{:x}", Sha256::digest(bytes)),
            bytes: bytes.to_vec(),
        })
    }
    /// Read a bounded YAML contract, retaining its exact bytes.
    pub fn read(path: impl AsRef<Path>) -> Result<Self> {
        Self::parse(&read_limited(path.as_ref(), MANIFEST_LIMIT)?)
    }
    /// The baseline contract embedded in this library; weights remain external.
    pub fn baseline() -> Self {
        Self::parse(BASELINE.as_bytes()).expect("checked-in baseline manifest")
    }
    /// Original YAML bytes. Copy these when packaging; do not reserialize them.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    /// SHA-256 of the original YAML file, distinct from a semantic/canonical hash.
    pub fn sha256(&self) -> &str {
        &self.sha256
    }
    /// Bundle identifier declared by this contract.
    pub fn id(&self) -> &str {
        &self.parsed.id
    }
    /// Exact source revision declared by this manifest.
    pub fn revision(&self) -> &str {
        &self.parsed.source.revision
    }
    /// Voice declarations without reading payloads or loading inference graphs.
    pub fn voice_catalog(&self) -> crate::voice::VoiceCatalog {
        crate::voice::VoiceCatalog::supertonic3(
            self.id(),
            self.revision(),
            self.sha256(),
            crate::voice::VoiceReadiness::CatalogOnly,
        )
    }
    /// Every payload file in the validated contract.
    pub fn files(&self) -> impl Iterator<Item = &LockedFile> {
        self.parsed.artifacts.values().flat_map(|a| &a.files)
    }
    /// Check a trusted expected file hash before opening any payloads.
    pub fn require_sha256(&self, expected: &str) -> Result<()> {
        if !hex(expected, 64) || expected != self.sha256 {
            return Err(Error::Manifest(
                "manifest SHA-256 differs from the pinned identity".into(),
            ));
        }
        Ok(())
    }
    /// Verify all payloads and parse owned conditioning/tokenizer data.
    pub fn verify(&self, root: impl AsRef<Path>) -> Result<Bundle> {
        Bundle::verify_manifest(self, root.as_ref())
    }
}

/// A fully checked local Supertonic 3 bundle. Loading never uses the network.
///
/// SHA-256 verifies consistency with the supplied manifest, not the manifest's
/// authenticity. Use a trusted manifest. Keep model files and symlink targets
/// immutable for the lifetime of this bundle and every engine created from it.
#[derive(Debug)]
pub struct Bundle {
    id: String,
    root: PathBuf,
    models: BTreeMap<String, PathBuf>,
    indexer: Vec<i32>,
    voices: BTreeMap<String, Voice>,
    manifest_sha256: String,
    revision: String,
}

/// Owned, shape-checked conditioning tensors for one bundled voice.
#[derive(Debug)]
pub struct Voice {
    id: String,
    ttl: HostTensor,
    dp: HostTensor,
}
impl Voice {
    /// Manifest voice identifier, such as M1 or F1.
    pub fn id(&self) -> &str {
        &self.id
    }
    /// Text/latent conditioning, Float32 [1, 50, 256].
    pub fn style_ttl(&self) -> &HostTensor {
        &self.ttl
    }
    /// Duration conditioning, Float32 [1, 8, 16].
    pub fn style_dp(&self) -> &HostTensor {
        &self.dp
    }
}

impl Bundle {
    // Only the store calls this, after verifying the staging tree and atomically
    // renaming it. Rebase known graph paths without reading the payload twice.
    pub(crate) fn relocated(mut self, root: PathBuf) -> Self {
        for path in self.models.values_mut() {
            *path = root.join(path.strip_prefix(&self.root).expect("verified graph path"));
        }
        self.root = root;
        self
    }
    /// Open a self-contained local bundle using its conventional manifest name.
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        Self::load(root.as_ref().join(BUNDLE_MANIFEST_NAME), root)
    }
    /// Open a self-contained bundle after checking a trusted manifest file hash.
    pub fn open_pinned(root: impl AsRef<Path>, manifest_sha256: &str) -> Result<Self> {
        let manifest = BundleManifest::read(root.as_ref().join(BUNDLE_MANIFEST_NAME))?;
        manifest.require_sha256(manifest_sha256)?;
        manifest.verify(root)
    }
    /// Verify a YAML manifest and every listed payload, then parse small assets.
    /// File symlinks are supported and checked by their target bytes. Unexpected
    /// files and nested directory symlinks inside compiled graphs are rejected;
    /// regular `.DS_Store` files are ignored. No inference model is loaded yet.
    pub fn load(manifest_path: impl AsRef<Path>, root: impl AsRef<Path>) -> Result<Self> {
        BundleManifest::read(manifest_path)?.verify(root)
    }
    fn verify_manifest(description: &BundleManifest, root: &Path) -> Result<Self> {
        let manifest = &description.parsed;
        let root = fs::canonicalize(root).map_err(|source| io(root, source))?;
        if !root.is_dir() {
            return Err(asset(&root, "expected a bundle directory"));
        }
        let mut metadata = BTreeMap::new();
        let mut models = BTreeMap::new();
        for (name, artifact) in &manifest.artifacts {
            let graph = artifact.kind == "coreml_compiled";
            for entry in &artifact.files {
                let bytes = verify_file(&root.join(&entry.path), entry, !graph)?;
                if !graph {
                    metadata.insert(name.as_str(), bytes);
                }
            }
            if graph {
                verify_inventory(&root, artifact)?;
                models.insert(name.clone(), root.join(&artifact.path));
            }
        }
        let config: Value = json(&metadata["tts_config"], "tts_config")?;
        for (pointer, expected) in [
            ("/ae/sample_rate", 44100),
            ("/ae/base_chunk_size", 512),
            ("/ttl/chunk_compress_factor", 6),
            ("/ttl/latent_dim", 24),
        ] {
            if config.pointer(pointer).and_then(Value::as_u64) != Some(expected) {
                return Err(asset("tts_config", format!("unsupported {pointer}")));
            }
        }
        let indexer: Vec<i32> = json(&metadata["unicode_indexer"], "unicode_indexer")?;
        if indexer.len() != 65536 || indexer.iter().any(|&v| !(-1..=8320).contains(&v)) {
            return Err(asset(
                "unicode_indexer",
                "expected 65536 entries in -1..=8320",
            ));
        }
        let mut voices = BTreeMap::new();
        for id in ["M1", "F1"] {
            let name = format!("voice_{id}");
            let data: VoiceFile = json(&metadata[name.as_str()], &name)?;
            voices.insert(
                id.into(),
                Voice {
                    id: id.into(),
                    ttl: data.style_ttl.tensor(&name, &[1, 50, 256])?,
                    dp: data.style_dp.tensor(&name, &[1, 8, 16])?,
                },
            );
        }
        Ok(Self {
            id: manifest.id.clone(),
            root,
            models,
            indexer,
            voices,
            manifest_sha256: description.sha256.clone(),
            revision: manifest.source.revision.clone(),
        })
    }
    /// Bundle identifier recorded by the manifest.
    pub fn id(&self) -> &str {
        &self.id
    }
    /// SHA-256 of the original manifest bytes used to verify this bundle.
    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }
    /// Verified local assets, without claiming a loaded or available runtime.
    pub fn voice_catalog(&self) -> crate::voice::VoiceCatalog {
        crate::voice::VoiceCatalog::supertonic3(
            self.id(),
            &self.revision,
            self.manifest_sha256(),
            crate::voice::VoiceReadiness::AssetsVerified,
        )
    }
    /// Canonical local bundle directory.
    pub fn root(&self) -> &Path {
        &self.root
    }
    /// Available voice identifiers, in deterministic order.
    pub fn voices(&self) -> impl Iterator<Item = &str> {
        self.voices.keys().map(String::as_str)
    }
    /// Look up an owned, validated voice style.
    pub fn voice(&self, id: &str) -> Result<&Voice> {
        self.voices
            .get(id)
            .ok_or_else(|| asset(id, "unknown voice; expected M1 or F1"))
    }
    /// Unicode scalar to token mapping. Negative entries are unsupported.
    pub fn unicode_indexer(&self) -> &[i32] {
        &self.indexer
    }
    /// Locate a verified compiled graph by its manifest artifact name.
    pub fn model_path(&self, name: &str) -> Result<&Path> {
        self.models
            .get(name)
            .map(PathBuf::as_path)
            .ok_or_else(|| asset(name, "unknown graph"))
    }
    pub(super) fn variant_models(
        &self,
        variant: crate::config::ExecutionVariant,
    ) -> Result<BTreeMap<String, PathBuf>> {
        use crate::config::ExecutionVariant;
        let mut models = BTreeMap::new();
        if variant == ExecutionVariant::Baseline {
            return Ok(models);
        }
        let manifest: VariantManifest =
            serde_yaml_ng::from_str(VARIANTS).map_err(|e| Error::Manifest(e.to_string()))?;
        if self.manifest_sha256 != manifest.base_manifest_sha256 {
            return Err(Error::Manifest(
                "execution variants require the exact pinned baseline manifest".into(),
            ));
        }
        for (name, artifact) in manifest.artifacts {
            if variant == ExecutionVariant::Fused8 && name != "ve_fused8_128" {
                continue;
            }
            for entry in &artifact.files {
                verify_file(&self.root.join(&entry.path), entry, false)?;
            }
            verify_inventory(&self.root, &artifact)?;
            models.insert(name, self.root.join(artifact.path));
        }
        Ok(models)
    }
}

fn parse_manifest(bytes: &[u8]) -> Result<Manifest> {
    // Value's mapping deserializer rejects duplicate keys even in nested maps;
    // direct deserialization into a BTreeMap would silently keep the last value.
    let value: serde_yaml_ng::Value =
        serde_yaml_ng::from_slice(bytes).map_err(|e| Error::Manifest(e.to_string()))?;
    let manifest: Manifest =
        serde_yaml_ng::from_value(value).map_err(|e| Error::Manifest(e.to_string()))?;
    let baseline: Manifest = serde_yaml_ng::from_str(BASELINE).expect("checked-in manifest schema");
    validate_manifest(&manifest, &baseline)?;
    Ok(manifest)
}

fn validate_manifest(m: &Manifest, baseline: &Manifest) -> Result<()> {
    let require = |ok, message: &str| {
        if ok {
            Ok(())
        } else {
            Err(Error::Manifest(message.into()))
        }
    };
    require(
        m.schema_version == 1
            && m.model == baseline.model
            && m.model_version == baseline.model_version,
        "unsupported schema or model version",
    )?;
    require(
        !m.id.is_empty() && m.id.len() <= 128,
        "invalid bundle identifier",
    )?;
    require(
        hex(&m.source.revision, 40) && m.source.repository.starts_with("https://"),
        "source must pin an HTTPS repository and full revision",
    )?;
    require(m.provenance.is_object(), "missing provenance mapping")?;
    require(
        m.contract == baseline.contract && m.integrity == baseline.integrity,
        "unsupported tensor, audio, preprocessing, or integrity contract",
    )?;
    for key in [
        "kind",
        "format",
        "specification_version",
        "deployment_target",
        "minimum_os",
        "compute_policy",
    ] {
        require(
            m.runtime.get(key) == baseline.runtime.get(key),
            &format!("unsupported runtime.{key}"),
        )?;
    }
    require(
        m.artifacts.keys().eq(baseline.artifacts.keys()),
        "expected the ten Supertonic baseline artifacts",
    )?;
    let mut seen = BTreeSet::new();
    let mut roots = BTreeSet::new();
    for (name, a) in &m.artifacts {
        let expected = &baseline.artifacts[name];
        relative(&a.path)?;
        require(roots.insert(&a.path), "duplicate artifact root")?;
        require(
            a.kind == expected.kind
                && a.inputs == expected.inputs
                && a.outputs == expected.outputs
                && a.latent_bucket == expected.latent_bucket,
            &format!("unsupported tensor interface for {name}"),
        )?;
        require(
            a.precision.is_null() || a.precision.is_object(),
            "invalid precision metadata",
        )?;
        let graph = a.kind == "coreml_compiled";
        require(
            !a.files.is_empty() && a.files.len() <= 256,
            "invalid locked file count",
        )?;
        if graph {
            require(
                a.path.ends_with(".mlmodelc"),
                "expected a compiled Core ML directory",
            )?;
        } else {
            require(
                a.files.len() == 1 && a.files[0].path == a.path,
                "metadata artifact must lock exactly its own file",
            )?;
        }
        for entry in &a.files {
            relative(&entry.path)?;
            require(
                if graph {
                    entry.path.starts_with(&(a.path.clone() + "/"))
                } else {
                    entry.path == a.path
                },
                "file lies outside its artifact",
            )?;
            require(seen.insert(&entry.path), "duplicate payload path")?;
            require(
                hex(&entry.sha256, 64) && entry.size_bytes > 0,
                "invalid SHA-256 or byte count",
            )?;
            require(
                graph || entry.size_bytes <= METADATA_LIMIT,
                "metadata exceeds 4 MiB",
            )?;
        }
    }
    for a in &roots {
        require(
            !roots
                .iter()
                .any(|b| a != b && a.starts_with(&(b.to_string() + "/"))),
            "overlapping artifact roots",
        )?;
    }
    Ok(())
}
fn hex(s: &str, size: usize) -> bool {
    s.len() == size
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn relative(s: &str) -> Result<()> {
    if s.is_empty()
        || s.contains(['\\', ':', '\0'])
        || s.split('/').any(|p| matches!(p, "" | "." | ".."))
    {
        Err(Error::Manifest(format!("unsafe relative path: {s:?}")))
    } else {
        Ok(())
    }
}
fn io(path: impl AsRef<Path>, source: std::io::Error) -> Error {
    Error::Io {
        path: path.as_ref().into(),
        source,
    }
}
fn asset(path: impl AsRef<Path>, message: impl Into<String>) -> Error {
    Error::Asset {
        path: path.as_ref().into(),
        message: message.into(),
    }
}
fn json<T: serde::de::DeserializeOwned>(bytes: &[u8], name: &str) -> Result<T> {
    serde_json::from_slice(bytes).map_err(|e| asset(name, e.to_string()))
}
fn read_limited(path: &Path, limit: u64) -> Result<Vec<u8>> {
    if !fs::metadata(path).map_err(|e| io(path, e))?.is_file() {
        return Err(asset(path, "expected a regular file"));
    }
    let file = File::open(path).map_err(|e| io(path, e))?;
    if !file.metadata().map_err(|e| io(path, e))?.is_file() {
        return Err(asset(path, "expected a regular file"));
    }
    let mut bytes = Vec::new();
    file.take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| io(path, e))?;
    if bytes.len() as u64 > limit {
        return Err(asset(path, "file exceeds size limit"));
    }
    Ok(bytes)
}
fn verify_file(path: &Path, entry: &LockedFile, keep: bool) -> Result<Vec<u8>> {
    if !fs::metadata(path).map_err(|e| io(path, e))?.is_file() {
        return Err(asset(path, "expected a regular file"));
    }
    let mut file = File::open(path).map_err(|e| io(path, e))?;
    let info = file.metadata().map_err(|e| io(path, e))?;
    if !info.is_file() || info.len() != entry.size_bytes {
        return Err(asset(path, "file type or byte count differs from manifest"));
    }
    let mut hasher = Sha256::new();
    let mut buffer = [0; 64 * 1024];
    let mut bytes = Vec::new();
    let mut count = 0u64;
    loop {
        let n = file.read(&mut buffer).map_err(|e| io(path, e))?;
        if n == 0 {
            break;
        }
        count = count
            .checked_add(n as u64)
            .ok_or_else(|| asset(path, "file size overflow"))?;
        if count > entry.size_bytes {
            return Err(asset(path, "file grew during verification"));
        }
        hasher.update(&buffer[..n]);
        if keep {
            bytes.extend_from_slice(&buffer[..n]);
        }
    }
    if count != entry.size_bytes || format!("{:x}", hasher.finalize()) != entry.sha256 {
        return Err(asset(path, "size or SHA-256 differs from manifest"));
    }
    Ok(bytes)
}
fn verify_inventory(root: &Path, artifact: &Artifact) -> Result<()> {
    let expected: BTreeSet<_> = artifact.files.iter().map(|f| root.join(&f.path)).collect();
    let mut directories = vec![root.join(&artifact.path)];
    let mut actual = BTreeSet::new();
    while let Some(directory) = directories.pop() {
        for entry in fs::read_dir(&directory).map_err(|e| io(&directory, e))? {
            let path = entry.map_err(|e| io(&directory, e))?.path();
            let info = fs::symlink_metadata(&path).map_err(|e| io(&path, e))?;
            if info.is_dir() {
                directories.push(path);
            } else if path.is_file() {
                if path.file_name().is_some_and(|n| n == ".DS_Store") && !info.is_symlink() {
                    continue;
                }
                actual.insert(path);
            } else {
                return Err(asset(
                    path,
                    "expected a regular file or directory; directory symlinks are unsupported",
                ));
            }
        }
    }
    if actual != expected {
        return Err(asset(
            root.join(&artifact.path),
            "unexpected or missing compiled-model files",
        ));
    }
    Ok(())
}

#[derive(Deserialize)]
struct VoiceFile {
    style_ttl: Style,
    style_dp: Style,
}
#[derive(Deserialize)]
struct Style {
    dims: Vec<usize>,
    r#type: String,
    data: Vec<Vec<Vec<f32>>>,
}
impl Style {
    fn tensor(self, name: &str, shape: &[usize; 3]) -> Result<HostTensor> {
        if self.dims != shape
            || self.r#type != "float32"
            || self.data.len() != shape[0]
            || self.data.iter().any(|b| {
                b.len() != shape[1]
                    || b.iter()
                        .any(|r| r.len() != shape[2] || r.iter().any(|v| !v.is_finite()))
            })
        {
            return Err(asset(
                name,
                "voice style shape, dtype, or values are invalid",
            ));
        }
        Ok(HostTensor::new(
            shape.to_vec(),
            TensorData::F32(self.data.into_iter().flatten().flatten().collect()),
        )?)
    }
}

#[cfg(test)]
pub(crate) mod tests;
