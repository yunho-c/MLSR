use super::*;

#[cfg(not(all(feature = "coreml", any(target_os = "macos", target_os = "ios"))))]
#[test]
fn unavailable_backend_is_reported_without_spawning_a_worker() {
    assert!(matches!(
        Session::coreml(ComputePolicy::default()),
        Err(Error::BackendUnavailable)
    ));
}

#[cfg(all(feature = "coreml", any(target_os = "macos", target_os = "ios")))]
mod apple {
    use super::*;
    use crate::runtime::{TensorData, coreml::tests::Fixture};

    fn host(value: f32) -> HostTensor {
        HostTensor::new(vec![1, 2, 3], TensorData::F32(vec![value; 6])).unwrap()
    }

    #[test]
    fn persists_models_releases_tensors_and_joins_on_shutdown() {
        let fixture = Fixture::compile();
        let session = Session::coreml(ComputePolicy::CpuOnly).unwrap();
        let model = session.load(fixture.model()).unwrap();
        let cached = session.load(fixture.model()).unwrap();
        assert_eq!(model.id, cached.id);
        let mut input = session.upload(host(1.)).unwrap();
        session.warm_up(&model, &[("x", &input)], &["y"]).unwrap();
        assert_eq!(session.stats().unwrap().live_tensors, 1);
        for value in 0..16 {
            session.write(&mut input, host(value as f32)).unwrap();
            let outputs = session.predict(&model, &[("x", &input)], &["y"]).unwrap();
            assert_eq!(
                session.read(&outputs[0]).unwrap(),
                host(value as f32 * 2. + 1.)
            );
        }
        assert_eq!(
            session.stats().unwrap(),
            SessionStats {
                model_loads: 1,
                live_tensors: 1,
                predictions: 17
            }
        );
        assert!(session.predict(&model, &[], &["y"]).is_err());
        let outputs = session.predict(&model, &[("x", &input)], &["y"]).unwrap();
        assert_eq!(session.read(&outputs[0]).unwrap(), host(31.));
        drop(outputs);
        drop(input);
        assert_eq!(session.stats().unwrap().live_tensors, 0);
        let other = session.clone();
        session.shutdown().unwrap();
        other.shutdown().unwrap();
        assert!(matches!(other.stats(), Err(Error::SessionClosed)));
        assert!(session.inner.join.lock().unwrap().is_none());
    }

    #[test]
    fn rejects_cross_session_handles_and_allows_concurrent_callers() {
        let fixture = Fixture::compile();
        let session = Session::coreml(ComputePolicy::CpuOnly).unwrap();
        let other = Session::coreml(ComputePolicy::CpuOnly).unwrap();
        let model = session.load(fixture.model()).unwrap();
        let foreign_model = other.load(fixture.model()).unwrap();
        let foreign_tensor = other.upload(host(1.)).unwrap();
        assert!(matches!(
            session.read(&foreign_tensor),
            Err(Error::ForeignHandle)
        ));
        assert!(matches!(
            session.predict(&model, &[("x", &foreign_tensor)], &["y"]),
            Err(Error::ForeignHandle)
        ));
        assert!(matches!(
            session.predict(&foreign_model, &[], &["y"]),
            Err(Error::ForeignHandle)
        ));
        let threads = (0..8)
            .map(|value| {
                let session = session.clone();
                let model = model.clone();
                std::thread::spawn(move || {
                    let input = session.upload(host(value as f32)).unwrap();
                    let out = session.predict(&model, &[("x", &input)], &["y"]).unwrap();
                    assert_eq!(session.read(&out[0]).unwrap(), host(2. * value as f32 + 1.));
                })
            })
            .collect::<Vec<_>>();
        for thread in threads {
            thread.join().unwrap();
        }
        assert_eq!(
            session.stats().unwrap(),
            SessionStats {
                model_loads: 1,
                live_tensors: 0,
                predictions: 8
            }
        );
        // Dropping the final client/handle also shuts down and joins automatically.
    }

    #[test]
    fn handles_keep_owner_alive_and_worker_panics_do_not_hang_callers() {
        let session = Session::coreml(ComputePolicy::CpuOnly).unwrap();
        let tensor = session.upload(host(2.)).unwrap();
        let weak = Arc::downgrade(&session.inner);
        drop(session);
        assert!(weak.upgrade().is_some());
        drop(tensor);
        assert!(weak.upgrade().is_none());
        let session = Session::coreml(ComputePolicy::CpuOnly).unwrap();
        assert!(matches!(
            session.inner.request(Request::Panic),
            Err(Error::Worker(_))
        ));
        assert!(matches!(session.stats(), Err(Error::SessionClosed)));
        session.shutdown().unwrap();
    }

    #[test]
    fn abandoned_replies_release_tensors_and_shutdown_handles_races() {
        let session = Session::coreml(ComputePolicy::CpuOnly).unwrap();
        let (reply, abandoned) = mpsc::sync_channel(1);
        drop(abandoned);
        session
            .inner
            .sender
            .send(Envelope {
                request: Request::Upload(host(1.)),
                reply: Some(reply),
            })
            .unwrap();
        assert_eq!(session.stats().unwrap().live_tensors, 0);
        let barrier = Arc::new(std::sync::Barrier::new(65));
        let threads = (0..64)
            .map(|_| {
                let client = session.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    assert!(matches!(client.stats(), Ok(_) | Err(Error::SessionClosed)));
                })
            })
            .collect::<Vec<_>>();
        barrier.wait();
        session.shutdown().unwrap();
        for thread in threads {
            thread.join().unwrap();
        }
    }
}
