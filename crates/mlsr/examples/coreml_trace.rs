//! Real public-job workload for Instruments. No random synthetic network inputs.
//! Timing here is instrumented; use benchmark_synthesis.py for ordinary latency.
use mlsr::{
    config::{EngineOptions, ExecutionVariant, JobOptions, SynthesisOptions},
    engine::{Engine, SynthesisRequest},
    job::{Event, Outcome, Poll},
    runtime::ComputePolicy,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs::OpenOptions,
    time::{Duration, Instant},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 7 {
        return Err(
            "usage: coreml_trace MANIFEST BUNDLE REPORT baseline|fused8 cpu|ane REPETITIONS SIGNPOST_LIBRARY".into(),
        );
    }
    let variant = match args[3].as_str() {
        "baseline" => ExecutionVariant::Baseline,
        "fused8" => ExecutionVariant::Fused8,
        _ => return Err("unsupported variant".into()),
    };
    let policy = match args[4].as_str() {
        "cpu" => ComputePolicy::CpuOnly,
        "ane" => ComputePolicy::CpuAndNeuralEngine,
        _ => return Err("unsupported compute policy".into()),
    };
    let repetitions: usize = args[5].parse()?;
    if !(1..=100).contains(&repetitions) {
        return Err("repetitions must be 1..=100".into());
    }
    let markers = markers::Markers::load(&args[6])?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[2])?;
    let started = Instant::now();
    let engine = Engine::load(
        &args[0],
        &args[1],
        EngineOptions {
            compute_policy: policy,
            execution_variant: variant,
            ..EngineOptions::default()
        },
    )?;
    let load_ms = started.elapsed().as_secs_f64() * 1000.;
    markers::wait_for_trace()?;
    let mut results = Vec::new();
    for iteration in 0..=repetitions {
        markers.mark(iteration as u64 + 1, true);
        let began = Instant::now();
        let job = engine.start(
            SynthesisRequest {
                text: "Hello from MLSR.",
                language: "en",
                voice: "M1",
                options: SynthesisOptions::new(8, 1.05, 1729)?,
            },
            JobOptions::default(),
        )?;
        let mut frames = 0;
        let mut hasher = Sha256::new();
        loop {
            match job.poll(Duration::from_secs(60)) {
                Poll::Event(event) => match event.event {
                    Event::Predicted(_) => (),
                    Event::Audio(chunk) => {
                        if chunk.start_frame != frames {
                            return Err("noncontiguous audio".into());
                        }
                        for sample in chunk.audio.samples() {
                            if !sample.is_finite() {
                                return Err("nonfinite audio".into());
                            }
                            hasher.update(sample.to_le_bytes());
                        }
                        frames += chunk.audio.frames() as u64;
                    }
                    Event::Terminal(report) => {
                        if !matches!(report.outcome, Outcome::Completed)
                            || frames == 0
                            || report.frames != frames
                        {
                            return Err(format!("unexpected terminal: {report:?}").into());
                        }
                        break;
                    }
                },
                other => return Err(format!("missing terminal: {other:?}").into()),
            }
        }
        results.push(json!({"iteration":iteration,"warmup":iteration==0,"frames":frames,
            "pcm_sha256":format!("{:x}",hasher.finalize()),"instrumented_ms":began.elapsed().as_secs_f64()*1000.}));
        markers.mark(iteration as u64 + 1, false);
    }
    let stats = engine.runtime_stats()?;
    engine.shutdown()?;
    // Allow Instruments to drain late hardware records after the final prediction.
    std::thread::sleep(Duration::from_secs(2));
    serde_yaml_ng::to_writer(
        &mut file,
        &json!({"passed":true,"process_id":std::process::id(),
        "variant":args[3],"compute_policy":args[4],"load_ms":load_ms,
        "stats":{"model_loads":stats.model_loads,"predictions":stats.predictions,"live_tensors":stats.live_tensors},
        "requests":results,"scope":"instrumented real speech; not an unprofiled latency benchmark"}),
    )?;
    Ok(())
}

// Only this development example loads the tiny public-SDK signpost helper. The
// production runtime remains entirely cidre-based and does not link this helper.
#[cfg(target_os = "macos")]
mod markers {
    use std::{
        ffi::{CString, c_char, c_int, c_void},
        io::Write,
        time::{Duration, Instant},
    };
    unsafe extern "C" {
        fn dlopen(path: *const c_char, mode: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
        fn dlclose(handle: *mut c_void) -> c_int;
        fn notify_register_check(name: *const c_char, token: *mut c_int) -> u32;
        fn notify_check(token: c_int, changed: *mut c_int) -> u32;
        fn notify_cancel(token: c_int) -> u32;
    }
    pub fn wait_for_trace() -> Result<(), Box<dyn std::error::Error>> {
        let Ok(path) = std::env::var("MLSR_TRACE_READY_PATH") else {
            return Ok(());
        };
        let name = CString::new(std::env::var("MLSR_TRACE_NOTIFICATION")?)?;
        let mut token = 0;
        // SAFETY: Terminated notification name and a valid token output slot.
        if unsafe { notify_register_check(name.as_ptr(), &mut token) } != 0 {
            return Err("could not arm the Instruments start notification".into());
        }
        struct Registration(c_int);
        impl Drop for Registration {
            fn drop(&mut self) {
                // SAFETY: This registration owns the live notification token.
                unsafe { notify_cancel(self.0) };
            }
        }
        let registration = Registration(token);
        let check = || -> Result<bool, Box<dyn std::error::Error>> {
            let mut changed = 0;
            // SAFETY: Live registration and valid output slot. Calls are serial.
            if unsafe { notify_check(registration.0, &mut changed) } != 0 {
                return Err("failed to read Instruments start notification".into());
            }
            Ok(changed != 0)
        };
        check()?; // The first check reports an initial state change; drain it.
        let mut ready = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        writeln!(ready, "{}", std::process::id())?;
        drop(ready);
        let deadline = Instant::now() + Duration::from_secs(120);
        while !check()? {
            if Instant::now() >= deadline {
                return Err("Instruments did not signal recording start".into());
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        Ok(())
    }
    pub struct Markers {
        handle: *mut c_void,
        mark: unsafe extern "C" fn(u64, u8),
    }
    impl Markers {
        pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
            let path = CString::new(path)?;
            // SAFETY: Valid terminated path; Darwin RTLD_NOW | RTLD_LOCAL. The
            // handle is retained until Drop after all marker calls finish.
            let handle = unsafe { dlopen(path.as_ptr(), 2 | 4) };
            if handle.is_null() {
                return Err("could not load the supplied Instruments marker library".into());
            }
            // SAFETY: Live handle and static terminated symbol name.
            let symbol = unsafe { dlsym(handle, c"mlsr_trace_mark".as_ptr()) };
            if symbol.is_null() {
                // SAFETY: No function pointer was retained from this live handle.
                unsafe { dlclose(handle) };
                return Err("marker library does not export mlsr_trace_mark".into());
            }
            Ok(Self {
                handle,
                // SAFETY: The caller supplies the helper compiled by
                // trace_synthesis.py from trace_signposts.m with this exact ABI.
                mark: unsafe {
                    std::mem::transmute::<*mut c_void, unsafe extern "C" fn(u64, u8)>(symbol)
                },
            })
        }
        pub fn mark(&self, id: u64, begin: bool) {
            // SAFETY: The verified helper ABI accepts these integer values and
            // calls are serialized on this example's main thread.
            unsafe { (self.mark)(id, u8::from(begin)) }
        }
    }
    impl Drop for Markers {
        fn drop(&mut self) {
            // SAFETY: The loader handle is owned here and no marker call remains.
            unsafe { dlclose(self.handle) };
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod markers {
    pub fn wait_for_trace() -> Result<(), Box<dyn std::error::Error>> {
        Err("Instruments qualification requires macOS".into())
    }
    pub struct Markers;
    impl Markers {
        pub fn load(_: &str) -> Result<Self, Box<dyn std::error::Error>> {
            Err("Instruments qualification requires macOS".into())
        }
        pub fn mark(&self, _: u64, _: bool) {}
    }
}
