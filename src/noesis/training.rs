//! NCP weight training via Evolutionary Strategy (ES).
//!
//! Optimizes NCP weights to maximize compositionality preservation:
//! after injecting bind(ROLE, VALUE) into the SemanticField for N steps,
//! querying with ROLE should still recover VALUE.
//!
//! Uses OpenAI-style ES: gradient estimated from random perturbations,
//! parallelized with rayon.

use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;

use crate::hdc::encoder::Encoder;
use crate::hdc::real::RealHV;
use crate::lnn::ncp::NCPConfig;
use crate::noesis::bridge::BridgeStrategy;
use crate::noesis::semantic_field::{SemanticField, SemanticFieldConfig};

/// Configuration for ES training.
#[derive(Clone, Debug)]
pub struct TrainingConfig {
    /// HD dimensionality.
    pub hd_dim: usize,
    /// Number of ES generations.
    pub generations: usize,
    /// Population size (number of perturbations per generation).
    pub population_size: usize,
    /// Noise standard deviation for perturbations.
    pub sigma: f32,
    /// Learning rate.
    pub lr: f32,
    /// Number of role-value pairs in the fitness evaluation.
    pub n_eval_pairs: usize,
    /// Number of steps to run the field after injecting a record.
    pub n_steps: usize,
    /// Time step.
    pub dt: f32,
    /// Seed for reproducibility.
    pub seed: u64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        TrainingConfig {
            hd_dim: 1024,
            generations: 100,
            population_size: 50,
            sigma: 0.02,
            lr: 0.03,
            n_eval_pairs: 12,
            n_steps: 5,
            dt: 0.1,
            seed: 42,
        }
    }
}

/// Result of training.
#[derive(Clone, Debug)]
pub struct TrainingResult {
    /// Best fitness per generation.
    pub fitness_history: Vec<f32>,
    /// Best parameters found.
    pub best_params: Vec<f32>,
    /// Final fitness.
    pub best_fitness: f32,
    /// Training time in seconds.
    pub elapsed_secs: f32,
}

/// Evaluates fitness as mean cosine similarity between unbinding result
/// and the correct value — a continuous signal that gives smooth gradients.
fn evaluate_fitness(
    params: &[f32],
    config: &TrainingConfig,
    eval_seed: u64,
) -> f32 {
    let sf_config = SemanticFieldConfig::new(
        config.hd_dim,
        BridgeStrategy::InputPreserving,
        NCPConfig::tiny(),
    );

    let mut encoder = Encoder::new(config.hd_dim, eval_seed + 1000);

    let roles: Vec<String> = (0..4).map(|i| format!("role_{i}")).collect();
    let values: Vec<String> = (0..4).map(|i| format!("val_{i}")).collect();
    for r in &roles {
        encoder.register(r);
    }
    for v in &values {
        encoder.register(v);
    }

    let mut total_sim = 0.0f32;
    let mut total = 0;

    for (ri, role) in roles.iter().enumerate().take(3) {
        for value in &values {
            let mut sf = SemanticField::new(sf_config.clone(), eval_seed + ri as u64);
            sf.set_ncp_params(params);

            let record_bin = encoder.encode_record_named(&[(role, value)]);
            let record_real = record_bin.to_real();

            for _ in 0..config.n_steps {
                sf.step(&record_real, config.dt);
            }

            // Query: unbind with role
            let role_bin = encoder.encode_atom(role).unwrap();
            let role_real = role_bin.to_real();
            let query_result = sf.query(&role_real);

            // Continuous fitness: similarity to correct value
            let value_bin = encoder.encode_atom(value).unwrap();
            let value_real = value_bin.to_real();
            let sim = RealHV::cosine_similarity(&query_result, &value_real);
            total_sim += sim;
            total += 1;

            if total >= config.n_eval_pairs {
                return total_sim / total as f32;
            }
        }
    }

    if total == 0 { 0.0 } else { total_sim / total as f32 }
}

/// Also compute top-3 and top-5 accuracy for reporting.
fn evaluate_fitness_detailed(
    params: &[f32],
    config: &TrainingConfig,
    eval_seed: u64,
) -> (f32, f32, f32) {
    let sf_config = SemanticFieldConfig::new(
        config.hd_dim,
        BridgeStrategy::InputPreserving,
        NCPConfig::tiny(),
    );

    let mut encoder = Encoder::new(config.hd_dim, eval_seed + 1000);

    let roles: Vec<String> = (0..4).map(|i| format!("role_{i}")).collect();
    let values: Vec<String> = (0..4).map(|i| format!("val_{i}")).collect();
    for r in &roles {
        encoder.register(r);
    }
    for v in &values {
        encoder.register(v);
    }

    let mut top1 = 0;
    let mut top3 = 0;
    let mut top5 = 0;
    let mut total = 0;

    for (ri, role) in roles.iter().enumerate().take(3) {
        for value in &values {
            let mut sf = SemanticField::new(sf_config.clone(), eval_seed + ri as u64);
            sf.set_ncp_params(params);

            let record_bin = encoder.encode_record_named(&[(role, value)]);
            let record_real = record_bin.to_real();

            for _ in 0..config.n_steps {
                sf.step(&record_real, config.dt);
            }

            let role_bin = encoder.encode_atom(role).unwrap();
            let role_real = role_bin.to_real();
            let query_result = sf.query(&role_real);
            let decoded = encoder.vocab().nearest_k(&query_result.to_binary(), 5);

            total += 1;
            if !decoded.is_empty() && decoded[0].0 == *value {
                top1 += 1;
            }
            if decoded.iter().take(3).any(|(l, _)| l == value) {
                top3 += 1;
            }
            if decoded.iter().take(5).any(|(l, _)| l == value) {
                top5 += 1;
            }
        }
    }

    let t = total as f32;
    (top1 as f32 / t, top3 as f32 / t, top5 as f32 / t)
}

/// Runs the ES training loop.
pub fn train(config: &TrainingConfig) -> TrainingResult {
    let start = Instant::now();

    // Initialize base parameters from a fresh SemanticField
    let sf_config = SemanticFieldConfig::new(
        config.hd_dim,
        BridgeStrategy::InputPreserving,
        NCPConfig::tiny(),
    );
    let sf = SemanticField::new(sf_config, config.seed);
    let mut theta = sf.get_ncp_params();
    let n_params = theta.len();

    println!("  NCP parameters: {}", n_params);
    println!("  Population size: {}", config.population_size);
    println!("  Sigma: {}", config.sigma);
    println!("  Learning rate: {}", config.lr);

    // Evaluate initial fitness
    let initial_fitness = evaluate_fitness(&theta, config, config.seed);
    println!("  Initial fitness (top-1): {:.1}%", initial_fitness * 100.0);

    let mut fitness_history = vec![initial_fitness];
    let mut best_fitness = initial_fitness;
    let mut best_params = theta.clone();

    let normal = Normal::new(0.0f32, 1.0f32).unwrap();

    for gen in 0..config.generations {
        // Generate perturbation noise for each population member
        let mut epsilons: Vec<Vec<f32>> = Vec::with_capacity(config.population_size);
        let mut rng = StdRng::seed_from_u64(config.seed + gen as u64 * 1000);
        for _ in 0..config.population_size {
            let eps: Vec<f32> = (0..n_params).map(|_| normal.sample(&mut rng)).collect();
            epsilons.push(eps);
        }

        // Evaluate fitness for each perturbation (parallelized)
        let fitnesses: Vec<f32> = epsilons
            .par_iter()
            .enumerate()
            .map(|(i, eps)| {
                // Positive perturbation
                let mut params_pos = theta.clone();
                for (p, e) in params_pos.iter_mut().zip(eps) {
                    *p += config.sigma * e;
                }
                let f_pos = evaluate_fitness(&params_pos, config, config.seed + i as u64 * 7);

                // Negative perturbation (antithetic)
                let mut params_neg = theta.clone();
                for (p, e) in params_neg.iter_mut().zip(eps) {
                    *p -= config.sigma * e;
                }
                let f_neg = evaluate_fitness(&params_neg, config, config.seed + i as u64 * 7);

                f_pos - f_neg
            })
            .collect();

        // Update parameters: theta += lr / (2 * pop * sigma) * sum(fitness_diff * epsilon)
        let scale = config.lr / (2.0 * config.population_size as f32 * config.sigma);
        for j in 0..n_params {
            let mut grad = 0.0f32;
            for (i, eps) in epsilons.iter().enumerate() {
                grad += fitnesses[i] * eps[j];
            }
            theta[j] += scale * grad;
        }

        // Evaluate current fitness
        let current_fitness = evaluate_fitness(&theta, config, config.seed);
        fitness_history.push(current_fitness);

        if current_fitness > best_fitness {
            best_fitness = current_fitness;
            best_params = theta.clone();
        }

        if (gen + 1) % 10 == 0 || gen == 0 {
            println!(
                "  Gen {:3}: fitness = {:.1}% (best = {:.1}%)",
                gen + 1,
                current_fitness * 100.0,
                best_fitness * 100.0,
            );
        }
    }

    let elapsed = start.elapsed().as_secs_f32();

    // Final detailed evaluation
    let (t1, t3, t5) = evaluate_fitness_detailed(&best_params, config, config.seed);
    println!("\n  Final evaluation with best params:");
    println!("    Top-1: {:.1}%", t1 * 100.0);
    println!("    Top-3: {:.1}%", t3 * 100.0);
    println!("    Top-5: {:.1}%", t5 * 100.0);
    println!("    Training time: {:.1}s", elapsed);

    TrainingResult {
        fitness_history,
        best_params,
        best_fitness,
        elapsed_secs: elapsed,
    }
}

/// Save training result to CSV.
pub fn save_training_csv(result: &TrainingResult, path: &str) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "generation,fitness")?;
    for (gen, &fitness) in result.fitness_history.iter().enumerate() {
        writeln!(f, "{},{:.6}", gen, fitness)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_short() {
        println!("\n=== NCP Training (ES) ===\n");

        let config = TrainingConfig {
            hd_dim: 1024,
            generations: 60,
            population_size: 40,
            sigma: 0.05,
            lr: 0.1,
            n_eval_pairs: 12,
            n_steps: 5,
            dt: 0.1,
            seed: 42,
        };

        let result = train(&config);

        // Save CSV
        let _ = save_training_csv(&result, "results/training_fitness.csv");

        // The fitness should improve or at least not crash
        assert!(
            result.best_fitness >= 0.0,
            "Fitness should be non-negative"
        );

        // Report final detailed accuracy with best params
        let (t1, t3, t5) = evaluate_fitness_detailed(&result.best_params, &config, config.seed);
        println!("\n  Best params evaluation:");
        println!("    Top-1: {:.1}%", t1 * 100.0);
        println!("    Top-3: {:.1}%", t3 * 100.0);
        println!("    Top-5: {:.1}%", t5 * 100.0);

        // Save best params for use in experiments
        let params_json = serde_json::to_string(&result.best_params).unwrap();
        std::fs::write("results/best_ncp_params.json", params_json).unwrap();
        println!("  Saved best params to results/best_ncp_params.json");

        println!("\n=== Training Complete ===\n");
    }
}
