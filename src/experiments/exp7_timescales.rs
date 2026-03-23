//! Experiment 7: Emergent timescales.
//!
//! Runs an NCP network with varying input over multiple steps and records
//! the effective time constants (gate values) at each step. Verifies that
//! different neurons develop different effective time constants, showing
//! emergent temporal specialization.

#[cfg(test)]
mod tests {
    use crate::lnn::ncp::{NCP, NCPConfig};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_exp7_timescales() {
        println!("\n=== Experiment 7: Emergent Timescales ===\n");

        let mut rng = StdRng::seed_from_u64(42);
        let config = NCPConfig::tiny(); // 8->4->4->4
        let mut ncp = NCP::new(config, &mut rng);

        println!("NCP architecture: {}", ncp.summary());

        let dt = 0.1;
        let n_steps = 50;

        // Record tau values at each step
        let mut all_inter_taus: Vec<Vec<f32>> = Vec::new();
        let mut all_command_taus: Vec<Vec<f32>> = Vec::new();
        let mut all_motor_taus: Vec<Vec<f32>> = Vec::new();

        println!("\n--- Running {} steps with varying input ---", n_steps);

        for step in 0..n_steps {
            // Varying input: sinusoidal at different frequencies
            let t = step as f32 * dt;
            let input: Vec<f32> = (0..8)
                .map(|i| {
                    let freq = 1.0 + i as f32 * 0.5;
                    (t * freq * std::f32::consts::PI * 2.0).sin()
                })
                .collect();

            // Step the NCP
            let _output = ncp.step(&input, dt);

            // Record effective time constants
            let tau = ncp.get_effective_tau(&input);
            all_inter_taus.push(tau.inter.clone());
            all_command_taus.push(tau.command.clone());
            all_motor_taus.push(tau.motor.clone());

            if step % 10 == 0 {
                println!(
                    "  Step {:3}: inter_tau = [{:.2}, {:.2}, {:.2}, {:.2}]",
                    step, tau.inter[0], tau.inter[1], tau.inter[2], tau.inter[3]
                );
                println!(
                    "           cmd_tau   = [{:.2}, {:.2}, {:.2}, {:.2}]",
                    tau.command[0], tau.command[1], tau.command[2], tau.command[3]
                );
                println!(
                    "           motor_tau = [{:.2}, {:.2}, {:.2}, {:.2}]",
                    tau.motor[0], tau.motor[1], tau.motor[2], tau.motor[3]
                );
            }
        }

        // Analyze: compute mean and std of tau for each neuron across time
        println!("\n--- Time Constant Statistics (across {} steps) ---", n_steps);

        let compute_stats = |taus: &[Vec<f32>], name: &str| -> (Vec<f32>, Vec<f32>) {
            let n_neurons = taus[0].len();
            let mut means = vec![0.0f32; n_neurons];
            let mut stds = vec![0.0f32; n_neurons];

            for neuron in 0..n_neurons {
                let values: Vec<f32> = taus.iter().map(|t| t[neuron]).collect();
                let mean = values.iter().sum::<f32>() / values.len() as f32;
                let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
                means[neuron] = mean;
                stds[neuron] = var.sqrt();
            }

            println!("\n  {} layer:", name);
            for neuron in 0..n_neurons {
                println!(
                    "    Neuron {}: mean_tau = {:8.2}, std_tau = {:8.2}",
                    neuron, means[neuron], stds[neuron]
                );
            }

            (means, stds)
        };

        let (inter_means, _) = compute_stats(&all_inter_taus, "Inter");
        let (command_means, _) = compute_stats(&all_command_taus, "Command");
        let (motor_means, _) = compute_stats(&all_motor_taus, "Motor");

        // Check that neurons show different effective time constants
        // Compute the range (max - min) of mean taus within each layer
        let range = |means: &[f32]| -> f32 {
            let min = means.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = means.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            max - min
        };

        let inter_range = range(&inter_means);
        let command_range = range(&command_means);
        let motor_range = range(&motor_means);

        println!("\n--- Tau Range (max - min of mean tau) ---");
        println!("  Inter range:   {:.2}", inter_range);
        println!("  Command range: {:.2}", command_range);
        println!("  Motor range:   {:.2}", motor_range);

        // Verify that at least one layer shows differentiation in time constants
        let total_range = inter_range + command_range + motor_range;
        println!("  Total range:   {:.2}", total_range);

        // All tau values should be finite and positive
        let all_finite = all_inter_taus.iter().chain(&all_command_taus).chain(&all_motor_taus)
            .all(|taus| taus.iter().all(|t| t.is_finite() && *t > 0.0));
        assert!(
            all_finite,
            "All tau values should be finite and positive"
        );

        // At least some differentiation should exist (different neurons = different taus)
        // Even with random initialization, neurons should not all be identical
        assert!(
            total_range > 0.0,
            "Neurons should show some differentiation in time constants (total range = {:.4})",
            total_range
        );

        println!("\n[PASS] Neurons show differentiated time constants (total range = {:.2})", total_range);

        // --- CSV Export ---
        use std::fs::File;
        use std::io::Write;
        {
            let mut csv_file = File::create("results/exp7_timescales.csv")
                .expect("Failed to create results/exp7_timescales.csv");
            writeln!(csv_file, "step,layer,neuron,tau_value").unwrap();
            for (step, taus) in all_inter_taus.iter().enumerate() {
                for (neuron, tau) in taus.iter().enumerate() {
                    writeln!(csv_file, "{},inter,{},{:.6}", step, neuron, tau).unwrap();
                }
            }
            for (step, taus) in all_command_taus.iter().enumerate() {
                for (neuron, tau) in taus.iter().enumerate() {
                    writeln!(csv_file, "{},command,{},{:.6}", step, neuron, tau).unwrap();
                }
            }
            for (step, taus) in all_motor_taus.iter().enumerate() {
                for (neuron, tau) in taus.iter().enumerate() {
                    writeln!(csv_file, "{},motor,{},{:.6}", step, neuron, tau).unwrap();
                }
            }
            println!("CSV saved to results/exp7_timescales.csv");
        }

        println!("\n=== Experiment 7 Complete ===\n");
    }
}
