//! Device-independent ordering for replacement/seek epochs.
//! Active-epoch route recovery and retained PCM remain in Player::update.

pub(crate) trait Endpoint {
    type Error;
    fn take_route_change(&mut self) -> Result<bool, Self::Error>;
    fn reset_endpoint(&mut self) -> Result<(), Self::Error>;
    fn reopen_endpoint(&mut self) -> Result<(), Self::Error>;
}

pub(crate) fn prepare<E: Endpoint>(endpoint: &mut E) -> Result<(), E::Error> {
    // Consume notices preceding this boundary. A change arriving while opening
    // the replacement remains pending for normal active-epoch route recovery.
    if endpoint.take_route_change()? || endpoint.reset_endpoint().is_err() {
        // A removed endpoint must not have to accept Stop/Reset before recovery.
        endpoint.reopen_endpoint()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct MockEndpoint {
        notices: u32,
        removed: bool,
        fail_open: bool,
        notify_on_open: bool,
        calls: Vec<&'static str>,
    }
    impl Endpoint for MockEndpoint {
        type Error = &'static str;

        fn take_route_change(&mut self) -> Result<bool, Self::Error> {
            self.calls.push("notices");
            let notices = std::mem::take(&mut self.notices);
            if notices > 32 {
                return Err("overflow");
            }
            Ok(notices != 0)
        }

        fn reset_endpoint(&mut self) -> Result<(), Self::Error> {
            self.calls.push("reset");
            if self.removed {
                Err("removed")
            } else {
                Ok(())
            }
        }

        fn reopen_endpoint(&mut self) -> Result<(), Self::Error> {
            self.calls.push("open");
            if self.notify_on_open {
                self.notices += 1;
            }
            if self.fail_open {
                Err("open failed")
            } else {
                self.removed = false;
                Ok(())
            }
        }
    }

    #[test]
    fn idle_route_change_is_consumed_before_next_speech_with_or_without_device_removal() {
        for removed in [false, true] {
            let mut endpoint = MockEndpoint {
                notices: 2,
                removed,
                ..Default::default()
            };
            prepare(&mut endpoint).unwrap();
            assert_eq!(endpoint.calls, ["notices", "open"]);
            // The next enqueue's route check cannot pause the new speech
            // because of an event queued while the previous speech was idle.
            assert!(!endpoint.take_route_change().unwrap());
            assert!(!endpoint.removed);
        }
    }

    #[test]
    fn ordinary_flush_resets_and_missing_notification_still_allows_recovery() {
        let mut endpoint = MockEndpoint::default();
        prepare(&mut endpoint).unwrap();
        assert_eq!(endpoint.calls, ["notices", "reset"]);
        endpoint.calls.clear();
        endpoint.removed = true;
        prepare(&mut endpoint).unwrap();
        assert_eq!(endpoint.calls, ["notices", "reset", "open"]);
    }

    #[test]
    fn change_during_reopen_remains_pending_for_active_epoch_pause() {
        let mut endpoint = MockEndpoint {
            notices: 1,
            notify_on_open: true,
            ..Default::default()
        };
        prepare(&mut endpoint).unwrap();
        assert!(endpoint.take_route_change().unwrap());
    }

    #[test]
    fn recovery_failure_and_notification_overflow_are_reported() {
        let mut endpoint = MockEndpoint {
            notices: 1,
            fail_open: true,
            ..Default::default()
        };
        assert_eq!(prepare(&mut endpoint), Err("open failed"));
        endpoint.notices = 33;
        endpoint.calls.clear();
        assert_eq!(prepare(&mut endpoint), Err("overflow"));
        assert_eq!(endpoint.calls, ["notices"]);
    }
}
