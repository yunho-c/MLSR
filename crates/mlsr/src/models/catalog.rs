//! Small, pinned model identities shared by packaging tools and host adapters.
use super::supertonic3::{Bundle, BundleManifest};
use crate::{
    error::{Error, Result},
    runtime::BackendKind,
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, path::Path};

const BUILTIN: &str = include_str!("../../../../model-manifests/catalog.yaml");
const LIMIT: usize = 64 * 1024;
const PLATFORMS: &[&str] = &[
    "macos-arm64",
    "macos-x64",
    "ios-arm64",
    "ios-sim-arm64",
    "ios-sim-x64",
    "android-arm64",
    "android-x64",
    "linux-arm64",
    "linux-x64",
    "windows-arm64",
    "windows-x64",
];

/// A deterministic catalog of exact releases. Model IDs are unique within it.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelCatalog {
    schema_version: u32,
    models: Vec<ModelDescriptor>,
}

/// Trusted identity and package-relative location of one model release.
/// A manifest's own checksums are insufficient: its exact hash is pinned here.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelDescriptor {
    id: String,
    backend: String,
    platforms: Vec<String>,
    manifest_sha256: String,
    asset_path: String,
}
impl ModelCatalog {
    /// Parse a bounded catalog, rejecting duplicate keys, IDs and unsafe paths.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() > LIMIT {
            return Err(invalid("catalog exceeds 64 KiB"));
        }
        let value: serde_yaml_ng::Value =
            serde_yaml_ng::from_slice(bytes).map_err(|e| invalid(e.to_string()))?;
        let mut catalog: Self =
            serde_yaml_ng::from_value(value).map_err(|e| invalid(e.to_string()))?;
        catalog.validate()?;
        catalog.models.sort_by(|a, b| a.id.cmp(&b.id));
        for model in &mut catalog.models {
            model.platforms.sort();
        }
        Ok(catalog)
    }
    /// Catalog qualified and embedded with this library. No weights are embedded.
    pub fn builtin() -> Self {
        Self::parse(BUILTIN.as_bytes()).expect("checked-in model catalog")
    }
    /// Stable YAML serialization for generated package metadata.
    pub fn to_yaml(&self) -> Result<String> {
        self.validate()?;
        serde_yaml_ng::to_string(self).map_err(|e| invalid(e.to_string()))
    }
    /// Models sorted by identifier.
    pub fn models(&self) -> impl Iterator<Item = &ModelDescriptor> {
        self.models.iter()
    }
    /// Resolve an exact model ID for a platform. No filesystem search or network.
    pub fn resolve(&self, id: &str, platform: &str) -> Result<&ModelDescriptor> {
        self.validate()?;
        self.models
            .iter()
            .find(|m| m.id == id && m.supports(platform))
            .ok_or_else(|| invalid(format!("model {id:?} is unavailable for {platform:?}")))
    }
    fn validate(&self) -> Result<()> {
        if self.schema_version != 1 || self.models.is_empty() || self.models.len() > 64 {
            return Err(invalid("expected catalog schema 1 and 1..=64 models"));
        }
        let mut ids = BTreeSet::new();
        let mut paths = BTreeSet::new();
        for model in &self.models {
            model.validate()?;
            if !ids.insert(&model.id) || !paths.insert(&model.asset_path) {
                return Err(invalid("duplicate model ID or asset path"));
            }
        }
        for a in &paths {
            if paths
                .iter()
                .any(|b| a != b && a.starts_with(&(b.to_string() + "/")))
            {
                return Err(invalid("overlapping model asset paths"));
            }
        }
        Ok(())
    }
}
impl ModelDescriptor {
    /// Exact bundle ID, also used as the application's stable selection key.
    pub fn id(&self) -> &str {
        &self.id
    }
    /// Runtime backend identifier, such as `coreml` or `qnn`.
    pub fn backend(&self) -> &str {
        &self.backend
    }
    /// Original manifest file hash; never a hash of reserialized YAML.
    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }
    /// Root-relative asset directory inside a generated package.
    pub fn asset_path(&self) -> &str {
        &self.asset_path
    }
    /// Qualified platform keys, such as `macos-arm64`.
    pub fn platforms(&self) -> impl Iterator<Item = &str> {
        self.platforms.iter().map(String::as_str)
    }
    /// Whether the catalog allows this release on the given platform.
    pub fn supports(&self, platform: &str) -> bool {
        self.platforms.iter().any(|p| p == platform)
    }
    /// Check a parsed manifest against this identity before reading its payload.
    pub fn verify_manifest(&self, manifest: &BundleManifest) -> Result<()> {
        self.validate()?;
        let backend = match manifest.backend() {
            BackendKind::CoreMl => "coreml",
            BackendKind::QnnHtp => "qnn",
        };
        if self.backend != backend || manifest.id() != self.id {
            return Err(invalid(
                "model ID or backend differs from its catalog entry",
            ));
        }
        manifest.require_sha256(&self.manifest_sha256)
    }
    /// Verify an already located directory. The host resolves its asset location.
    pub fn open(&self, root: impl AsRef<Path>) -> Result<Bundle> {
        let manifest =
            BundleManifest::read(root.as_ref().join(super::supertonic3::BUNDLE_MANIFEST_NAME))?;
        self.verify_manifest(&manifest)?;
        manifest.verify(root)
    }
    pub(crate) fn validate(&self) -> Result<()> {
        if !component(&self.id) || !component(&self.backend) {
            return Err(invalid(
                "model ID/backend must be lowercase path components",
            ));
        }
        if self.manifest_sha256.len() != 64
            || !self
                .manifest_sha256
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err(invalid("expected lowercase manifest SHA-256"));
        }
        if self.asset_path.len() > 512 || self.asset_path.split('/').any(|s| !component(s)) {
            return Err(invalid("unsafe package-relative model asset path"));
        }
        let unique: BTreeSet<_> = self.platforms.iter().collect();
        if unique.is_empty()
            || unique.len() != self.platforms.len()
            || self
                .platforms
                .iter()
                .any(|p| !PLATFORMS.contains(&p.as_str()))
        {
            return Err(invalid("unknown, empty or duplicated model platform list"));
        }
        Ok(())
    }
}
fn component(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 128
        && s.as_bytes()[0].is_ascii_lowercase()
        && s.bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'_')
}
fn invalid(message: impl Into<String>) -> Error {
    Error::Manifest(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn builtin_pins_exact_baseline_and_qualified_platform() {
        let catalog = ModelCatalog::builtin();
        let recommended = BundleManifest::recommended();
        catalog
            .resolve(recommended.id(), "macos-arm64")
            .unwrap()
            .verify_manifest(&recommended)
            .unwrap();
        assert!(catalog.resolve(recommended.id(), "ios-arm64").is_err());
        let manifest = BundleManifest::baseline();
        let model = catalog.resolve(manifest.id(), "macos-arm64").unwrap();
        model.verify_manifest(&manifest).unwrap();
        assert!(catalog.resolve(manifest.id(), "ios-arm64").is_err());
        assert!(catalog.resolve("latest", "macos-arm64").is_err());
        assert_eq!(
            ModelCatalog::parse(catalog.to_yaml().unwrap().as_bytes())
                .unwrap()
                .to_yaml()
                .unwrap(),
            catalog.to_yaml().unwrap()
        );
        let changed = BundleManifest::parse(
            format!(
                "# changed bytes\n{}",
                String::from_utf8_lossy(manifest.bytes())
            )
            .as_bytes(),
        )
        .unwrap();
        assert!(model.verify_manifest(&changed).is_err());
    }
    #[test]
    fn qnn_catalog_requires_explicit_opt_in_and_matching_backend() {
        let bytes = include_bytes!("../../../../model-manifests/supertonic3-qnn-superfast.yaml");
        let manifest = BundleManifest::parse(bytes).unwrap();
        assert_eq!(manifest.backend(), BackendKind::QnnHtp);
        let catalog = ModelCatalog::parse(include_bytes!(
            "../../../../model-manifests/catalog-qnn-experimental.yaml"
        ))
        .unwrap();
        let model = catalog.resolve(manifest.id(), "windows-arm64").unwrap();
        model.verify_manifest(&manifest).unwrap();
        assert!(catalog.resolve(manifest.id(), "macos-arm64").is_err());
        assert!(
            ModelCatalog::builtin()
                .resolve(manifest.id(), "windows-arm64")
                .is_err()
        );

        let mut mismatched = model.clone();
        mismatched.backend = "coreml".into();
        assert!(mismatched.verify_manifest(&manifest).is_err());
        let baseline = BundleManifest::baseline();
        let mut mismatched = ModelCatalog::builtin().models().next().unwrap().clone();
        mismatched.backend = "qnn".into();
        assert!(mismatched.verify_manifest(&baseline).is_err());
    }

    #[test]
    fn rejects_ambiguous_or_unsafe_catalogs() {
        for (before, after) in [
            ("schema_version: 1", "schema_version: 2"),
            ("backend: coreml", "backend: ../coreml"),
            ("platforms: [macos-arm64]", "platforms: []"),
            (
                "platforms: [macos-arm64]",
                "platforms: [macos-arm64, macos-arm64]",
            ),
            ("platforms: [macos-arm64]", "platforms: [macos-typo]"),
            ("asset_path: models/", "asset_path: ../models/"),
            ("asset_path: models/", "asset_path: /models/"),
        ] {
            assert!(
                ModelCatalog::parse(BUILTIN.replace(before, after).as_bytes()).is_err(),
                "{after}"
            );
        }
        assert!(ModelCatalog::parse(format!("schema_version: 1\n{BUILTIN}").as_bytes()).is_err());
        let mut catalog = ModelCatalog::builtin();
        catalog.models.push(catalog.models[0].clone());
        assert!(catalog.to_yaml().is_err());
    }
}
