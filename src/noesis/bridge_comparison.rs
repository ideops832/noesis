//! Comparative evaluation of the four bridge strategies.

use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::hdc::encoder::Encoder;
use crate::hdc::real::RealHV;
use crate::lnn::ncp::NCPConfig;
use crate::noesis::bridge::BridgeStrategy;
use crate::noesis::semantic_field::{SemanticField, SemanticFieldConfig};

/// Result of one test for one configuration.
#[derive(Debug)]
pub struct BridgeTestResult {
    pub strategy: String,
    pub dim: usize,
    pub ncp_preset: String,
    pub distance_correlation: f32,
    pub compositionality_top1: f32,
    pub compositionality_top3: f32,
    pub compositionality_top5: f32,
    pub convergence_steps: Option<usize>,
    pub performance_us: f32,
}

/// Runs the full bridge comparison and returns results.
pub fn run_comparison() -> Vec<BridgeTestResult> {
    let strategies = [("RandomProjection", BridgeStrategy::RandomProjection),
        ("DualTrack", BridgeStrategy::DualTrack),
        ("InputPreserving", BridgeStrategy::InputPreserving)];
    // DirectHD and SparseHD are too expensive at D=4096
    // We test them only at D=1024
    let strategies_1024_only = [("DirectHD", BridgeStrategy::DirectHD),
        ("SparseHD", BridgeStrategy::SparseHD)];

    let presets = vec![
        ("tiny", NCPConfig::tiny()),
        ("standard", NCPConfig::standard()),
    ];

    let dims = vec![1024usize, 4096];
    let mut results = Vec::new();

    for &dim in &dims {
        let active_strategies: Vec<_> = if dim <= 1024 {
            strategies.iter().chain(strategies_1024_only.iter()).collect()
        } else {
            strategies.iter().collect()
        };

        for (strat_name, strategy) in &active_strategies {
            for (preset_name, ncp_config) in &presets {
                // Skip invalid combos
                if matches!(strategy, BridgeStrategy::DirectHD | BridgeStrategy::SparseHD)
                    && *preset_name == "standard"
                {
                    continue; // DirectHD/SparseHD use custom NCP configs
                }

                let config = SemanticFieldConfig::new(
                    dim,
                    (*strategy).clone(),
                    ncp_config.clone(),
                );

                let result = run_single_comparison(strat_name, &config, dim, preset_name);
                results.push(result);
            }
        }
    }

    results
}

fn run_single_comparison(
    strat_name: &str,
    config: &SemanticFieldConfig,
    dim: usize,
    preset_name: &str,
) -> BridgeTestResult {
    let n_trials = 3;

    let mut total_dist_corr = 0.0f32;
    let mut total_top1 = 0.0f32;
    let mut total_top3 = 0.0f32;
    let mut total_top5 = 0.0f32;
    let mut total_conv_steps = 0usize;
    let mut conv_count = 0usize;
    let mut total_perf_us = 0.0f32;

    for trial in 0..n_trials {
        let seed = 42 + trial as u64;

        // Test 1: Distance preservation (round-trip without LNN dynamics)
        total_dist_corr += test_distance_preservation(config, seed, dim);

        // Test 2: Compositionality
        let (t1, t3, t5) = test_compositionality(config, seed, dim);
        total_top1 += t1;
        total_top3 += t3;
        total_top5 += t5;

        // Test 4: Convergence
        if let Some(steps) = test_convergence(config, seed, dim) {
            total_conv_steps += steps;
            conv_count += 1;
        }

        // Test 5: Performance
        total_perf_us += test_performance(config, seed, dim);
    }

    BridgeTestResult {
        strategy: strat_name.to_string(),
        dim,
        ncp_preset: preset_name.to_string(),
        distance_correlation: total_dist_corr / n_trials as f32,
        compositionality_top1: total_top1 / n_trials as f32,
        compositionality_top3: total_top3 / n_trials as f32,
        compositionality_top5: total_top5 / n_trials as f32,
        convergence_steps: if conv_count > 0 {
            Some(total_conv_steps / conv_count)
        } else {
            None
        },
        performance_us: total_perf_us / n_trials as f32,
    }
}

fn test_distance_preservation(config: &SemanticFieldConfig, seed: u64, dim: usize) -> f32 {
    let mut sf = SemanticField::new(config.clone(), seed);
    let mut rng = StdRng::seed_from_u64(seed + 1000);

    let n_pairs = 50;
    let mut original_sims = Vec::new();
    let mut roundtrip_sims = Vec::new();

    for _ in 0..n_pairs {
        let a = RealHV::random(dim, &mut rng);
        let b = RealHV::random(dim, &mut rng);
        let orig_sim = RealHV::cosine_similarity(&a, &b);

        // Round trip: encode → step → decode (1 step)
        sf.reset();
        sf.step(&a, 0.1);
        let state_a = sf.state().clone();

        sf.reset();
        sf.step(&b, 0.1);
        let state_b = sf.state().clone();

        let rt_sim = RealHV::cosine_similarity(&state_a, &state_b);

        original_sims.push(orig_sim);
        roundtrip_sims.push(rt_sim);
    }

    // Pearson correlation
    pearson_correlation(&original_sims, &roundtrip_sims)
}

fn test_compositionality(config: &SemanticFieldConfig, seed: u64, dim: usize) -> (f32, f32, f32) {
    let _rng = StdRng::seed_from_u64(seed + 2000);
    let mut encoder = Encoder::new(dim, seed + 3000);

    let roles = ["role0", "role1", "role2", "role3", "role4"];
    let values = ["val_a", "val_b", "val_c", "val_d"];

    // Register all
    for r in &roles {
        encoder.register(r);
    }
    for v in &values {
        encoder.register(v);
    }

    let mut top1_correct = 0;
    let mut top3_correct = 0;
    let mut top5_correct = 0;
    let mut total = 0;

    for (ri, role) in roles.iter().enumerate().take(3) {
        for value in &values {
            let record_bin = encoder.encode_record_named(&[(role, value)]);
            let record_real = record_bin.to_real();

            let mut sf = SemanticField::new(config.clone(), seed + ri as u64);

            // Inject record for several steps
            for _ in 0..5 {
                sf.step(&record_real, 0.1);
            }

            // Query: unbind with role
            let role_bin = encoder.encode_atom(role).unwrap();
            let role_real = role_bin.to_real();
            let query_result = sf.query(&role_real);

            // Find nearest in vocabulary
            let decoded = encoder.vocab().nearest_k(&query_result.to_binary(), 5);

            total += 1;
            if !decoded.is_empty() && decoded[0].0 == *value {
                top1_correct += 1;
            }
            if decoded.iter().take(3).any(|(l, _)| l == value) {
                top3_correct += 1;
            }
            if decoded.iter().take(5).any(|(l, _)| l == value) {
                top5_correct += 1;
            }
        }
    }

    if total == 0 {
        return (0.0, 0.0, 0.0);
    }
    (
        top1_correct as f32 / total as f32,
        top3_correct as f32 / total as f32,
        top5_correct as f32 / total as f32,
    )
}

fn test_convergence(config: &SemanticFieldConfig, seed: u64, dim: usize) -> Option<usize> {
    let mut sf = SemanticField::new(config.clone(), seed);
    let mut rng = StdRng::seed_from_u64(seed + 4000);
    let input = RealHV::random(dim, &mut rng);

    for step in 0..100 {
        let prev = sf.state().clone();
        sf.step(&input, 0.1);
        let diff: f32 = sf
            .state()
            .data
            .iter()
            .zip(&prev.data)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / dim as f32;

        if diff < 0.001 {
            return Some(step);
        }
    }
    None
}

fn test_performance(config: &SemanticFieldConfig, seed: u64, dim: usize) -> f32 {
    let mut sf = SemanticField::new(config.clone(), seed);
    let mut rng = StdRng::seed_from_u64(seed + 5000);
    let input = RealHV::random(dim, &mut rng);

    // Warm up
    for _ in 0..10 {
        sf.step(&input, 0.1);
    }

    let n_iters = 100;
    let start = Instant::now();
    for _ in 0..n_iters {
        sf.step(&input, 0.1);
    }
    let elapsed = start.elapsed();
    elapsed.as_micros() as f32 / n_iters as f32
}

fn pearson_correlation(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len() as f32;
    let mean_x: f32 = x.iter().sum::<f32>() / n;
    let mean_y: f32 = y.iter().sum::<f32>() / n;

    let mut cov = 0.0f32;
    let mut var_x = 0.0f32;
    let mut var_y = 0.0f32;

    for (xi, yi) in x.iter().zip(y) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-10 {
        0.0
    } else {
        cov / denom
    }
}

/// Prints and saves the comparison results.
pub fn print_results(results: &[BridgeTestResult]) {
    println!("\n{:=<100}", "");
    println!("NOESIS Bridge Comparison Results");
    println!("{:=<100}", "");
    println!(
        "{:<20} {:>5} {:>10} {:>8} {:>8} {:>8} {:>10} {:>10}",
        "Strategy", "D", "NCP", "Top1%", "Top3%", "Top5%", "Conv", "μs/step"
    );
    println!("{:-<100}", "");

    for r in results {
        let conv = match r.convergence_steps {
            Some(s) => format!("{}", s),
            None => "N/A".to_string(),
        };
        println!(
            "{:<20} {:>5} {:>10} {:>7.1}% {:>7.1}% {:>7.1}% {:>10} {:>10.1}",
            r.strategy,
            r.dim,
            r.ncp_preset,
            r.compositionality_top1 * 100.0,
            r.compositionality_top3 * 100.0,
            r.compositionality_top5 * 100.0,
            conv,
            r.performance_us,
        );
    }
    println!("{:=<100}", "");
}

/// Saves results as CSV.
pub fn save_csv(results: &[BridgeTestResult], path: &str) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    writeln!(
        f,
        "strategy,dim,ncp_preset,dist_corr,top1,top3,top5,conv_steps,us_per_step"
    )?;
    for r in results {
        writeln!(
            f,
            "{},{},{},{:.4},{:.4},{:.4},{:.4},{},{:.1}",
            r.strategy,
            r.dim,
            r.ncp_preset,
            r.distance_correlation,
            r.compositionality_top1,
            r.compositionality_top3,
            r.compositionality_top5,
            r.convergence_steps
                .map(|s| s.to_string())
                .unwrap_or_else(|| "N/A".to_string()),
            r.performance_us,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_comparison() {
        let results = run_comparison();
        print_results(&results);

        // Save CSV
        let _ = save_csv(&results, "results/bridge_comparison.csv");

        // At least one result should exist
        assert!(!results.is_empty());

        // Print summary for analysis
        for r in &results {
            println!(
                "{} D={} NCP={}: Top1={:.1}% Top3={:.1}% Top5={:.1}%",
                r.strategy,
                r.dim,
                r.ncp_preset,
                r.compositionality_top1 * 100.0,
                r.compositionality_top3 * 100.0,
                r.compositionality_top5 * 100.0,
            );
        }
    }
}
