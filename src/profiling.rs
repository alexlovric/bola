use std::{
    collections::HashMap,
    sync::{
        Mutex, OnceLock,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

struct ProfileData {
    calls: AtomicUsize,
    duration_nanos: AtomicU64,
}

// A global, thread-safe, lazily-initialized HashMap to store all profiling data.
static PROFILER: OnceLock<Mutex<HashMap<&'static str, ProfileData>>> = OnceLock::new();

// Helper function to access the global profiler.
fn profiler() -> &'static Mutex<HashMap<&'static str, ProfileData>> {
    PROFILER.get_or_init(|| Mutex::new(HashMap::new()))
}

/// A timer that records its duration upon being dropped.
/// It also increments the call count for the function being timed.
pub struct ScopedTimer {
    start: Instant,
    name: &'static str,
}

impl ScopedTimer {
    pub fn new(name: &'static str) -> Self {
        let profiler = profiler();
        let mut guard = profiler.lock().unwrap();
        let entry = guard.entry(name).or_insert_with(|| ProfileData {
            calls: AtomicUsize::new(0),
            duration_nanos: AtomicU64::new(0),
        });
        entry.calls.fetch_add(1, Ordering::SeqCst);

        ScopedTimer {
            start: Instant::now(),
            name,
        }
    }
}

impl Drop for ScopedTimer {
    fn drop(&mut self) {
        let duration = self.start.elapsed().as_nanos() as u64;
        let profiler = profiler();
        let guard = profiler.lock().unwrap();
        // The entry is guaranteed to exist from the `new` call.
        if let Some(data) = guard.get(self.name) {
            data.duration_nanos.fetch_add(duration, Ordering::SeqCst);
        }
    }
}

/// Resets all profiling data.
pub fn reset_counters() {
    let profiler = profiler();
    let mut guard = profiler.lock().unwrap();
    guard.clear();
}

/// Prints a formatted report of all functions that were called.
/// The percentages are calculated relative to the total time of the specified top-level function.
/// Prints a formatted report of all functions that were called.
/// The function with the longest total duration is automatically used as the 100% baseline.
pub fn print_profile(iterations: f64) {
    println!("\n--- Function Call Profiling (Average over {} iterations) ---", iterations);
    let profiler = profiler();
    let guard = profiler.lock().unwrap();

    // Find the maximum duration to use as the baseline for percentages.
    let total_time_nanos = guard
        .values()
        .map(|data| data.duration_nanos.load(Ordering::SeqCst))
        .max()
        .unwrap_or(0);

    if total_time_nanos == 0 {
        println!("No profiling data recorded.");
        return;
    }

    let mut entries: Vec<_> = guard.iter().collect();
    // Sort by percentage of total time, descending, to see hotspots first.
    entries.sort_by_key(|(_, data)| std::cmp::Reverse(data.duration_nanos.load(Ordering::SeqCst)));

    println!(
        "{:<10} | {:<12} | {:<15} | {:<15} | {:<10}",
        "Function", "Calls", "Avg Time/Iter", "Avg Time/Call", "Percentage"
    );
    println!("{:-<78}", ""); // Separator line

    for (name, data) in entries {
        let calls = data.calls.load(Ordering::SeqCst);
        if calls > 0 {
            let total_nanos = data.duration_nanos.load(Ordering::SeqCst);
            let avg_nanos_per_iter = (total_nanos as f64 / iterations) as u64;
            let avg_nanos_per_call = total_nanos / calls as u64;
            let percentage = total_nanos as f64 / total_time_nanos as f64 * 100.0;

            println!(
                "{:<10} | {:<12} | {:<15.2?} | {:<15.2?} | {:>9.2?}%",
                name,
                calls,
                Duration::from_nanos(avg_nanos_per_iter),
                Duration::from_nanos(avg_nanos_per_call),
                percentage
            );
        }
    }
}

// The benchmark function remains unchanged as it serves a different purpose.
pub fn benchmark<F, G>(name: &str, mut my_impl: F, mut lapack_impl: G, repetitions: usize)
where
    F: FnMut() -> f64,
    G: FnMut() -> f64,
{
    println!("--- Comparing {} ---", name);

    let mut elapsed_time1 = 0.0;
    for _ in 0..repetitions {
        elapsed_time1 += my_impl();
    }
    let my_avg_time = elapsed_time1 / repetitions as f64;
    println!("My {} time: {:.2?}ms", name, my_avg_time * 1000.0);

    let mut elapsed_time2 = 0.0;
    for _ in 0..repetitions {
        elapsed_time2 += lapack_impl();
    }
    let lapack_avg_time = elapsed_time2 / repetitions as f64;
    println!("CBLAS {} time: {:.2?}ms", name, lapack_avg_time * 1000.0);

    println!("Ratio: {:?}\n", my_avg_time / lapack_avg_time);
}
