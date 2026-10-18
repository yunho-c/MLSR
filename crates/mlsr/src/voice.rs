//! Model-scoped voice identity, explicit locale resolution and readiness snapshots.
use crate::{
    error::{Error, Result},
    models::supertonic3::SUPPORTED_LANGUAGES,
};
use serde::Serialize;
use std::sync::OnceLock;

/// What has actually been established when this metadata snapshot was taken.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VoiceReadiness {
    /// Declared in a manifest; payload presence and runtime are not established.
    CatalogOnly,
    /// All model assets verified locally; inference graphs are not loaded.
    AssetsVerified,
    /// This engine loaded the graphs. It can still be closed after this snapshot.
    EngineReady,
}
/// Locale fallback is opt-in; it never invents a regional accent.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VoiceFallback {
    /// Match an explicitly supported language tag exactly, ignoring ASCII case.
    #[default]
    Strict,
    /// An unsupported regional/script tag may use its supported base language.
    SameLanguage,
}
/// Timing and synthesis capabilities declared by the model implementation.
#[derive(Clone, Debug, Serialize)]
pub struct VoiceCapabilities {
    /// The engine emits owned PCM, independent of an audio output device.
    pub pcm: bool,
    /// The model can predict segment duration before waveform generation.
    pub duration_prediction: bool,
    /// Generated segments retain their original source-text ranges.
    pub segment_alignment: bool,
    /// Exact word timestamps are currently unavailable.
    pub word_alignment: bool,
}
/// Immutable voice metadata. Identity does not contain an execution backend.
#[derive(Clone, Debug, Serialize)]
pub struct VoiceDescriptor {
    id: String,
    model_voice_id: String,
    display_name: String,
    model: String,
    bundle_id: String,
    revision: String,
    manifest_sha256: String,
    backend: String,
    languages: Vec<String>,
    qualified_languages: Vec<String>,
    requires_network: bool,
    readiness: VoiceReadiness,
    capabilities: VoiceCapabilities,
}
impl VoiceDescriptor {
    /// Backend-independent stable identity, for example `supertonic3:M1`.
    pub fn id(&self) -> &str {
        &self.id
    }
    /// Model-local conditioning ID used by the low-level synthesis API.
    pub fn model_voice_id(&self) -> &str {
        &self.model_voice_id
    }
    /// Display label, without an invented regional accent.
    pub fn display_name(&self) -> &str {
        &self.display_name
    }
    /// Model family/version identity.
    pub fn model(&self) -> &str {
        &self.model
    }
    /// Exact bundle release identity.
    pub fn bundle_id(&self) -> &str {
        &self.bundle_id
    }
    /// Declared source revision of this model release.
    pub fn revision(&self) -> &str {
        &self.revision
    }
    /// Original model manifest SHA-256, identifying the exact asset contract.
    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }
    /// Execution backend for this catalog entry, distinct from voice identity.
    pub fn backend(&self) -> &str {
        &self.backend
    }
    /// Frontend/model-supported tags; not all have been quality-qualified.
    pub fn languages(&self) -> &[String] {
        &self.languages
    }
    /// Languages exercised by the accepted real-model qualification corpus.
    pub fn qualified_languages(&self) -> &[String] {
        &self.qualified_languages
    }
    /// Whether inference requires a network connection once assets are present.
    pub fn requires_network(&self) -> bool {
        self.requires_network
    }
    /// Asset/runtime readiness at snapshot time, never a promise of future state.
    pub fn readiness(&self) -> VoiceReadiness {
        self.readiness
    }
    /// Model capabilities, independent of optional host playback support.
    pub fn capabilities(&self) -> &VoiceCapabilities {
        &self.capabilities
    }
}
/// Resolved voice and language for an immutable synthesis request.
#[derive(Clone, Debug, Serialize)]
pub struct ResolvedVoice {
    voice: VoiceDescriptor,
    requested_locale: String,
    language: String,
    used_fallback: bool,
}
impl ResolvedVoice {
    /// Selected voice metadata snapshot.
    pub fn voice(&self) -> &VoiceDescriptor {
        &self.voice
    }
    /// Caller locale, with original casing retained for diagnostics.
    pub fn requested_locale(&self) -> &str {
        &self.requested_locale
    }
    /// Canonical model language tag passed to the frontend.
    pub fn language(&self) -> &str {
        &self.language
    }
    /// Whether a regional/script tag resolved through same-language fallback.
    pub fn used_fallback(&self) -> bool {
        self.used_fallback
    }
}
/// Ordered immutable voice metadata. Selection order is deterministic (M1, F1).
#[derive(Clone, Debug, Serialize)]
pub struct VoiceCatalog {
    voices: Vec<VoiceDescriptor>,
}
impl VoiceCatalog {
    pub(crate) fn supertonic3(
        bundle: &str,
        revision: &str,
        manifest: &str,
        readiness: VoiceReadiness,
    ) -> Self {
        static QUALIFIED_MANIFEST: OnceLock<String> = OnceLock::new();
        let qualified_languages = if manifest
            == QUALIFIED_MANIFEST.get_or_init(|| {
                crate::models::supertonic3::BundleManifest::baseline()
                    .sha256()
                    .into()
            }) {
            vec!["en".into(), "ko".into()]
        } else {
            Vec::new()
        };
        Self {
            voices: ["M1", "F1"]
                .into_iter()
                .map(|id| VoiceDescriptor {
                    id: format!("supertonic3:{id}"),
                    model_voice_id: id.into(),
                    display_name: format!("Supertonic 3 {id}"),
                    model: "supertonic3".into(),
                    bundle_id: bundle.into(),
                    revision: revision.into(),
                    manifest_sha256: manifest.into(),
                    backend: "coreml".into(),
                    languages: SUPPORTED_LANGUAGES.iter().map(|s| (*s).into()).collect(),
                    qualified_languages: qualified_languages.clone(),
                    requires_network: false,
                    readiness,
                    capabilities: VoiceCapabilities {
                        pcm: true,
                        duration_prediction: true,
                        segment_alignment: true,
                        word_alignment: false,
                    },
                })
                .collect(),
        }
    }
    pub(crate) fn with_readiness(mut self, readiness: VoiceReadiness) -> Self {
        for voice in &mut self.voices {
            voice.readiness = readiness;
        }
        self
    }
    pub(crate) fn retain_voices(mut self, ids: &[String]) -> Self {
        self.voices.retain(|v| ids.contains(&v.model_voice_id));
        self
    }
    /// Entries in explicit default-selection order. No runtime is loaded here.
    pub fn voices(&self) -> &[VoiceDescriptor] {
        &self.voices
    }
    /// Exact stable ID lookup. Raw M1/F1 aliases belong to the low-level engine.
    pub fn get(&self, id: &str) -> Result<&VoiceDescriptor> {
        self.voices
            .iter()
            .find(|v| v.id == id)
            .ok_or_else(|| Error::VoiceUnavailable(id.into()))
    }
    /// Resolve a locale and optional stable voice ID. No cross-language fallback,
    /// filesystem search, inference or downloads occur. A bare language tag does
    /// not promise any particular regional accent.
    pub fn resolve(
        &self,
        locale: &str,
        voice_id: Option<&str>,
        fallback: VoiceFallback,
    ) -> Result<ResolvedVoice> {
        let mut subtags = locale.split('-');
        let base = subtags.next().unwrap_or_default();
        if locale.len() > 63
            || !(2..=3).contains(&base.len())
            || !base.bytes().all(|c| c.is_ascii_alphabetic())
            || subtags.any(|s| {
                s.is_empty() || s.len() > 8 || !s.bytes().all(|c| c.is_ascii_alphanumeric())
            })
        {
            return Err(Error::UnsupportedLocale(locale.into()));
        }
        let requested = locale.to_ascii_lowercase();
        let base = base.to_ascii_lowercase();
        let selected = match voice_id {
            Some(id) => vec![self.get(id)?],
            None => self.voices.iter().collect(),
        };
        // Prefer an exact supported tag across the catalog before falling back.
        for language in [
            Some(requested.as_str()),
            (fallback == VoiceFallback::SameLanguage && requested != base).then_some(base.as_str()),
        ]
        .into_iter()
        .flatten()
        {
            if let Some(voice) = selected
                .iter()
                .find(|v| v.languages.iter().any(|tag| tag == language))
            {
                return Ok(ResolvedVoice {
                    voice: (*voice).clone(),
                    requested_locale: locale.into(),
                    language: language.into(),
                    used_fallback: requested != language,
                });
            }
        }
        Err(Error::UnsupportedLocale(locale.into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::supertonic3::{BundleManifest, Fixture};
    #[test]
    fn metadata_and_readiness_are_distinct_from_loading() {
        let catalog = BundleManifest::baseline().voice_catalog();
        let voice = catalog.get("supertonic3:M1").unwrap();
        assert_eq!(voice.readiness(), VoiceReadiness::CatalogOnly);
        assert_eq!(voice.languages().len(), 31);
        assert_eq!(voice.qualified_languages(), &["en", "ko"]);
        assert!(!voice.requires_network());
        assert!(!voice.capabilities().word_alignment);
        let fixture = Fixture::new();
        let verified = fixture.load().unwrap().voice_catalog();
        assert_eq!(
            verified.get(voice.id()).unwrap().readiness(),
            VoiceReadiness::AssetsVerified
        );
        assert_eq!(verified.get(voice.id()).unwrap().model_voice_id(), "M1");
    }
    #[test]
    fn explicit_fallback_never_silently_changes_language_or_voice() {
        let catalog = BundleManifest::baseline().voice_catalog();
        let voice = catalog.resolve("EN", None, VoiceFallback::Strict).unwrap();
        assert_eq!(voice.voice().id(), "supertonic3:M1");
        assert_eq!(voice.language(), "en");
        assert!(!voice.used_fallback());
        assert!(
            catalog
                .resolve("en-US", None, VoiceFallback::Strict)
                .is_err()
        );
        let voice = catalog
            .resolve("ko-KR", Some("supertonic3:F1"), VoiceFallback::SameLanguage)
            .unwrap();
        assert_eq!(voice.voice().id(), "supertonic3:F1");
        assert_eq!(voice.language(), "ko");
        assert!(voice.used_fallback());
        assert_eq!(
            catalog
                .resolve("en", Some("M1"), VoiceFallback::Strict)
                .unwrap_err()
                .code(),
            "voice_unavailable"
        );
        for locale in ["", "en_US", "en-", "en-!", "en--US", "zz-ZZ", "中文"] {
            assert_eq!(
                catalog
                    .resolve(locale, None, VoiceFallback::SameLanguage)
                    .unwrap_err()
                    .code(),
                "unsupported_locale"
            );
        }
    }
}
