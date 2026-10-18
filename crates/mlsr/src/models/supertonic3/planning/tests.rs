use super::*;
use serde::Deserialize;

#[test]
fn matches_captured_duration_bits_frames_and_bucket_boundaries() {
    #[derive(Deserialize)]
    struct Fixture {
        cases: Vec<Case>,
    }
    #[derive(Deserialize)]
    struct Case {
        id: String,
        raw_duration_bits: u32,
        speed_bits: u32,
        duration_bits: u32,
        requested_frames: usize,
        latent_length: usize,
        latent_bucket: usize,
    }
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/supertonic3/planning.yaml");
    let fixture: Fixture = serde_yaml_ng::from_slice(&std::fs::read(path).unwrap()).unwrap();
    assert_eq!(fixture.cases.len(), 14);
    for case in fixture.cases {
        let options = SynthesisOptions::new(8, f32::from_bits(case.speed_bits), 0).unwrap();
        let plan =
            SynthesisPlan::from_duration(f32::from_bits(case.raw_duration_bits), options).unwrap();
        assert_eq!(
            plan.duration_seconds().to_bits(),
            case.duration_bits,
            "{}",
            case.id
        );
        assert_eq!(
            plan.requested_frames(),
            case.requested_frames,
            "{}",
            case.id
        );
        assert_eq!(plan.latent_length(), case.latent_length, "{}", case.id);
        assert_eq!(plan.latent_bucket(), case.latent_bucket, "{}", case.id);
    }
}
#[test]
fn validates_options_duration_bounds_and_noise_before_allocation() {
    for speed in [0., -1., f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(SynthesisOptions::new(8, speed, 0).is_err());
    }
    for steps in [0, 65, u32::MAX] {
        assert!(SynthesisOptions::new(steps, 1., 0).is_err());
    }
    for raw in [f32::NAN, f32::INFINITY, -1., 0., f32::MAX, 40.] {
        assert!(
            SynthesisPlan::from_duration(raw, SynthesisOptions::new(8, 1., 0).unwrap()).is_err()
        );
    }
    let clamped =
        SynthesisPlan::from_duration(0.1, SynthesisOptions::new(8, 0.001, 0).unwrap()).unwrap();
    assert_eq!(clamped.duration_seconds(), 2.);
    let plan = SynthesisPlan::from_duration(1., SynthesisOptions::new(8, 1., 0).unwrap()).unwrap();
    assert!(
        plan.prepare_saved_noise(HostTensor::new(vec![1], TensorData::F32(vec![0.])).unwrap())
            .is_err()
    );
    assert!(
        plan.prepare_saved_noise(
            HostTensor::new(
                vec![1, 144, plan.length],
                TensorData::I32(vec![0; 144 * plan.length])
            )
            .unwrap()
        )
        .is_err()
    );
    assert!(
        plan.prepare_saved_noise(
            HostTensor::new(
                vec![1, 144, plan.length],
                TensorData::F32(vec![f32::NAN; 144 * plan.length])
            )
            .unwrap()
        )
        .is_err()
    );
}
#[test]
fn pads_each_channel_preserving_noise_and_masks_exactly() {
    let plan = SynthesisPlan::from_duration(1., SynthesisOptions::new(8, 1., 0).unwrap()).unwrap();
    let values = (0..144 * plan.length)
        .map(|i| if i == 0 { -0.0 } else { i as f32 / 100. })
        .collect::<Vec<_>>();
    let prepared = plan
        .prepare_saved_noise(
            HostTensor::new(vec![1, 144, plan.length], TensorData::F32(values.clone())).unwrap(),
        )
        .unwrap();
    let TensorData::F32(padded) = prepared.padded_noise().data() else {
        unreachable!()
    };
    for (source, output) in values
        .chunks_exact(plan.length)
        .zip(padded.chunks_exact(plan.bucket))
    {
        assert!(
            source
                .iter()
                .zip(output)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        assert!(output[plan.length..].iter().all(|v| v.to_bits() == 0));
    }
    let TensorData::F32(mask) = prepared.padded_mask().data() else {
        unreachable!()
    };
    assert!(mask[..plan.length].iter().all(|&v| v == 1.));
    assert!(mask[plan.length..].iter().all(|v| v.to_bits() == 0));
}
#[test]
fn seeded_noise_is_request_local_and_has_gaussian_scale() {
    let plan = SynthesisPlan::from_duration(20., SynthesisOptions::new(8, 1., 0).unwrap()).unwrap();
    let a = plan.prepare_noise(1729).unwrap();
    let b = plan.prepare_noise(1730).unwrap();
    let again = plan.prepare_noise(1729).unwrap();
    assert_eq!(a.noise(), again.noise());
    assert_ne!(a.noise(), b.noise());
    let TensorData::F32(values) = a.noise().data() else {
        unreachable!()
    };
    assert!(values.iter().all(|v| v.is_finite()));
    let mean = values.iter().map(|&v| f64::from(v)).sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|&v| (f64::from(v) - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    assert!(mean.abs() < 0.04 && (0.9..1.1).contains(&variance));
}
