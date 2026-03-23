//! Experiment 6: Concepts as attractors.
//!
//! Uses the AttractorAnalyzer with inputs from three semantic categories
//! (animals, vehicles, food) and verifies that trajectories are valid.
//! Explores whether different semantic inputs lead to different attractor
//! basins in the semantic field.

#[cfg(test)]
mod tests {
    use crate::hdc::hypervector::HyperVector;
    use crate::hdc::real::RealHV;
    use crate::language::composer::Composer;
    use crate::lnn::ncp::NCPConfig;
    use crate::noesis::attractor::AttractorAnalyzer;
    use crate::noesis::bridge::BridgeStrategy;
    use crate::noesis::semantic_field::SemanticFieldConfig;

    const D: usize = 1024;

    #[test]
    fn test_exp6_attractors() {
        println!("\n=== Experiment 6: Concepts as Attractors ===\n");

        let mut composer = Composer::new(D, 42);

        // Create category inputs by encoding representative sentences
        let animal_sentences = vec![
            "Il gatto dorme sul divano",
            "Il cane corre nel parco",
            "Il felino mangia il pesce",
        ];
        let vehicle_sentences = vec![
            "Il treno parte dalla stazione",
            "La macchina viaggia veloce",
            "L'autobus porta i passeggeri",
        ];
        let food_sentences = vec![
            "La pasta cuoce nella pentola",
            "La pizza esce dal forno",
            "Il pane fresco profuma",
        ];

        // Encode and bundle each category into a representative vector
        let encode_category = |composer: &mut Composer, sentences: &[&str]| -> RealHV {
            let hvs: Vec<RealHV> = sentences
                .iter()
                .filter_map(|s| composer.encode_sentence(s))
                .collect();
            let refs: Vec<&RealHV> = hvs.iter().collect();
            RealHV::bundle_normalized(&refs)
        };

        let animal_hv = encode_category(&mut composer, &animal_sentences);
        let vehicle_hv = encode_category(&mut composer, &vehicle_sentences);
        let food_hv = encode_category(&mut composer, &food_sentences);

        println!("Category vectors created:");
        println!("  Animal norm:  {:.4}", animal_hv.norm());
        println!("  Vehicle norm: {:.4}", vehicle_hv.norm());
        println!("  Food norm:    {:.4}", food_hv.norm());

        // Inter-category similarities (should be relatively low)
        let sim_av = RealHV::cosine_similarity(&animal_hv, &vehicle_hv);
        let sim_af = RealHV::cosine_similarity(&animal_hv, &food_hv);
        let sim_vf = RealHV::cosine_similarity(&vehicle_hv, &food_hv);
        println!("\nInter-category similarities:");
        println!("  Animal-Vehicle: {:.4}", sim_av);
        println!("  Animal-Food:    {:.4}", sim_af);
        println!("  Vehicle-Food:   {:.4}", sim_vf);

        // Create AttractorAnalyzer
        let config = SemanticFieldConfig::new(D, BridgeStrategy::DualTrack, NCPConfig::tiny());
        let analyzer = AttractorAnalyzer::new(config, 42);

        // Analyze trajectories from each category input
        let categories = vec![
            ("Animal", &animal_hv),
            ("Vehicle", &vehicle_hv),
            ("Food", &food_hv),
        ];

        let mut trajectory_results = Vec::new();

        for (name, hv) in &categories {
            println!("\n--- Trajectory: {} ---", name);
            let traj = analyzer.trajectory_analysis(hv, 10);

            println!("  States recorded: {}", traj.states.len());
            println!("  Converged: {}", traj.converged);
            if let Some(step) = traj.convergence_step {
                println!("  Convergence at step: {}", step);
            }

            // Report early and late convergence values
            let n = traj.convergence.len();
            if n >= 5 {
                let early_avg: f32 = traj.convergence[..5].iter().sum::<f32>() / 5.0;
                let late_avg: f32 = traj.convergence[n - 5..].iter().sum::<f32>() / 5.0;
                println!("  Early avg convergence: {:.4}", early_avg);
                println!("  Late avg convergence:  {:.4}", late_avg);
            }

            // Verify trajectory is valid
            assert!(
                traj.states.iter().all(|s| s.data.iter().all(|x| x.is_finite())),
                "{} trajectory should contain only finite values",
                name
            );
            assert!(
                !traj.states.is_empty(),
                "{} trajectory should have states",
                name
            );

            trajectory_results.push((name.to_string(), traj));
        }

        // Compare final states across categories
        println!("\n--- Final State Comparisons ---");
        for i in 0..trajectory_results.len() {
            for j in (i + 1)..trajectory_results.len() {
                let final_i = trajectory_results[i].1.states.last().unwrap();
                let final_j = trajectory_results[j].1.states.last().unwrap();
                let sim = RealHV::cosine_similarity(final_i, final_j);
                println!(
                    "  {} vs {}: {:.4}",
                    trajectory_results[i].0, trajectory_results[j].0, sim
                );
            }
        }

        // Find fixed points with a small number of probes
        println!("\n--- Fixed Point Search (5 probes) ---");
        let fps = analyzer.find_fixed_points(5);
        println!("  Fixed points found: {}", fps.len());
        for (i, fp) in fps.iter().enumerate() {
            println!("  FP {}: stability = {:.4}, norm = {:.4}", i, fp.stability, fp.state.norm());
        }

        println!("\n[PASS] All trajectories valid, {} fixed points found", fps.len());

        // --- CSV Export ---
        use std::fs::File;
        use std::io::Write;
        {
            // Trajectory convergence CSV
            let mut csv_file = File::create("results/exp6_attractors.csv")
                .expect("Failed to create results/exp6_attractors.csv");
            writeln!(csv_file, "category,step,convergence_similarity").unwrap();
            for (name, traj) in &trajectory_results {
                for (step, conv) in traj.convergence.iter().enumerate() {
                    writeln!(csv_file, "{},{},{:.6}", name, step, conv).unwrap();
                }
            }
            println!("CSV saved to results/exp6_attractors.csv");

            // Fixed points CSV
            let mut fp_file = File::create("results/exp6_fixed_points.csv")
                .expect("Failed to create results/exp6_fixed_points.csv");
            writeln!(fp_file, "fp_index,stability,norm").unwrap();
            for (i, fp) in fps.iter().enumerate() {
                writeln!(fp_file, "{},{:.6},{:.6}", i, fp.stability, fp.state.norm()).unwrap();
            }
            println!("CSV saved to results/exp6_fixed_points.csv");
        }

        println!("\n=== Experiment 6 Complete ===\n");
    }
}
