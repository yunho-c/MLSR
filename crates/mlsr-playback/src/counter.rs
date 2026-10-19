//! Bounded notification counters for native output callbacks.
use std::sync::atomic::{AtomicU32, Ordering};

pub(crate) fn increment_bounded(counter: &AtomicU32, limit: u32) {
    // Equivalent to fetch_update(AcqRel, Acquire, ...), which is deprecated.
    // Its replacement try_update requires Rust 1.95; our minimum is 1.88.
    // Retry against concurrent drains as well as writers, including at the cap.
    let mut previous = counter.load(Ordering::Acquire);
    loop {
        let next = previous.saturating_add(1).min(limit);
        match counter.compare_exchange_weak(previous, next, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => return,
            Err(actual) => previous = actual,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;

    #[test]
    fn overflow_remains_observable_and_a_drained_counter_can_restart() {
        let counter = AtomicU32::new(0);
        std::thread::scope(|scope| {
            for _ in 0..4 {
                let counter = &counter;
                scope.spawn(move || {
                    for _ in 0..100 {
                        increment_bounded(counter, 33);
                    }
                });
            }
        });
        assert_eq!(counter.swap(0, Ordering::AcqRel), 33);
        increment_bounded(&counter, 33);
        assert_eq!(counter.load(Ordering::Acquire), 1);
        counter.store(u32::MAX, Ordering::Release);
        increment_bounded(&counter, u32::MAX);
        assert_eq!(counter.load(Ordering::Acquire), u32::MAX);
    }

    #[test]
    fn concurrent_writers_and_drains_account_for_every_notice_below_the_cap() {
        let counter = AtomicU32::new(0);
        let finished = AtomicU32::new(0);
        let barrier = Barrier::new(5);
        std::thread::scope(|scope| {
            for _ in 0..4 {
                let (counter, finished, barrier) = (&counter, &finished, &barrier);
                scope.spawn(move || {
                    barrier.wait();
                    for _ in 0..10000 {
                        increment_bounded(counter, u32::MAX);
                    }
                    finished.fetch_add(1, Ordering::Release);
                });
            }
            barrier.wait();
            let mut observed = 0;
            while finished.load(Ordering::Acquire) != 4 {
                observed += counter.swap(0, Ordering::AcqRel);
                std::thread::yield_now();
            }
            observed += counter.swap(0, Ordering::AcqRel);
            assert_eq!(observed, 40000);
        });
    }
}
