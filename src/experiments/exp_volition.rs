//! Experiment: Volition Phase 1 — γ-modulated deliberative field.
//!
//! Scenario:
//!   - A tactical goal HV encodes "classify concept in correct domain"
//!     (encoded via a small seed vocabulary).
//!   - The system is fed 20 inputs:
//!       * first 10 are aligned with the goal (domain-matching concepts)
//!       * last  10 are strong local attractors but semantically irrelevant
//!   - The same run is repeated for γ ∈ {0.0, 0.2, 0.4, 0.6, 0.8}.
//!
//! For each (γ, step) we log:
//!   - sim(combined_state, goal)
//!   - coherence_score
//!   - field_resistance
//!   - goal_status
//!
//! Success metrics:
//!   STRONG   — with γ=0.6: sim(field,goal) > 0.5 after the 10 attractors,
//!              with γ=0.0: same value < 0.3, decay curves are distinguishable.
//!   MODERATE — measurable gap between low/high γ + 7/10 correct choices
//!              on the coherence-selection sub-test.
//!
//! CSV outputs go into `results/volition/`.

#[cfg(test)]
mod tests {
    use std::fs::{create_dir_all, File};
    use std::io::Write;

    use crate::hdc::real::RealHV;
    use crate::lnn::ncp::NCPConfig;
    use crate::noesis::bridge::BridgeStrategy;
    use crate::noesis::multiscale::{MultiScaleConfig, MultiScaleField};
    use crate::noesis::volition::{GoalLevel, VolitionLayer};

    const D: usize = 1024;
    const RESULTS_DIR: &str = "results/volition";

    fn ensure_results_dir() {
        let _ = create_dir_all(RESULTS_DIR);
    }

    fn make_field() -> MultiScaleField {
        let mut cfg = MultiScaleConfig::default_for_dim(D);
        cfg.strategy = BridgeStrategy::InputPreserving;
        cfg.ncp_config = NCPConfig::tiny();
        MultiScaleField::new(cfg, 7)
    }

    /// Build a deterministic set of "aligned" and "attractor" hypervectors.
    ///
    /// The trick: the goal HV is an independent random bipolar vector. The
    /// aligned inputs are the goal + small orthogonal noise. The attractors
    /// are completely independent random vectors. This gives us a measurable
    /// contrast that the composer-based approach cannot: natural HDC bundles
    /// of English words saturate at sim > 0.95 regardless of content.
    fn build_goal_and_inputs(goal_seed: u64) -> (RealHV, Vec<RealHV>, Vec<RealHV>) {
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let mut rng = StdRng::seed_from_u64(goal_seed);
        let goal = RealHV::random(D, &mut rng);

        // Aligned inputs: goal + independent small noise. cos(input, goal) ≈ 0.9
        let mut aligned = Vec::new();
        for _ in 0..10 {
            let noise = RealHV::random_normal(D, &mut rng);
            // Scale noise so the aligned input is still dominated by the goal.
            let mix = RealHV::add(&goal, &noise.scale(0.15));
            aligned.push(mix.normalized());
        }

        // Attractors: fully independent random vectors (cos ≈ 0).
        let mut attractors = Vec::new();
        for _ in 0..10 {
            attractors.push(RealHV::random(D, &mut rng));
        }

        (goal, aligned, attractors)
    }

    /// Run the 20-input scenario for a single γ.
    /// Returns rows: (step, label, sim_field_goal, coherence_score, resistance, phase, status)
    fn run_scenario(
        gamma: f32,
        goal_hv: &RealHV,
        aligned: &[RealHV],
        attractors: &[RealHV],
    ) -> Vec<(usize, String, f32, f32, f32, &'static str, String)> {
        let mut field = make_field();
        let mut layer = VolitionLayer::new(D);
        let _goal_id = layer.set_goal_raw(
            goal_hv.clone(),
            GoalLevel::Tactical,
            gamma,
            0.8,
        );

        let mut rows = Vec::new();
        let mut step = 0usize;

        // Measure on fast_state: this is the scale that reflects the current
        // input after γ-biasing. combined_state is dominated by the slow
        // field (weight 0.4) which anchors forever to the very first input
        // and thus masks γ effects.
        for (i, input) in aligned.iter().enumerate() {
            let out = layer.process(input, &mut field);
            let sim_to_goal = RealHV::cosine_similarity(field.fast_state(), goal_hv);
            rows.push((
                step, format!("aligned_{}", i),
                sim_to_goal, out.coherence.score, out.field_resistance,
                "aligned", format!("{:?}", out.goal_status),
            ));
            step += 1;
        }

        for (i, input) in attractors.iter().enumerate() {
            let out = layer.process(input, &mut field);
            let sim_to_goal = RealHV::cosine_similarity(field.fast_state(), goal_hv);
            rows.push((
                step, format!("attractor_{}", i),
                sim_to_goal, out.coherence.score, out.field_resistance,
                "attractor", format!("{:?}", out.goal_status),
            ));
            step += 1;
        }

        rows
    }

    /// Candidate-selection sub-test: 10 trials, each with 3 candidates.
    /// One candidate is the goal + small noise (correct); the other two are
    /// independent random HVs. We expect the system to pick the correct one.
    fn run_selection_test(goal_hv: &RealHV) -> (usize, usize) {
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let mut layer = VolitionLayer::new(D);
        layer.set_goal_raw(goal_hv.clone(), GoalLevel::Tactical, 0.6, 0.6);

        let mut rng = StdRng::seed_from_u64(0xBEEF_F00D);
        let mut correct = 0;
        let mut total = 0;

        for trial in 0..10 {
            // Place the correct candidate at a rotating position so the test
            // is not biased by index.
            let correct_idx = trial % 3;
            let mut candidates: Vec<RealHV> = (0..3)
                .map(|_| RealHV::random(D, &mut rng))
                .collect();
            // Replace candidate[correct_idx] with a noisy goal.
            let noise = RealHV::random_normal(D, &mut rng);
            let good = RealHV::add(&goal_hv.clone(), &noise.scale(0.1)).normalized();
            candidates[correct_idx] = good;

            let ranked = layer.propose_and_evaluate(&candidates);
            total += 1;
            if !ranked.is_empty() && ranked[0].0 == correct_idx {
                correct += 1;
            }
        }
        (correct, total)
    }

    #[test]
    fn test_volition_gamma_sweep() {
        println!("\n=== Experiment: Volition γ sweep ===\n");
        ensure_results_dir();

        // Goal & inputs use random HVs to avoid HDC bundle saturation.
        let (goal_hv, aligned, attractors) = build_goal_and_inputs(0xA7E1);

        // Sanity print: similarity of inputs to goal.
        let a_sim = RealHV::cosine_similarity(&aligned[0], &goal_hv);
        let t_sim = RealHV::cosine_similarity(&attractors[0], &goal_hv);
        println!("aligned[0]   sim_to_goal = {:.3}", a_sim);
        println!("attractor[0] sim_to_goal = {:.3}", t_sim);

        // ---------- γ sweep ----------
        let gammas = [0.0f32, 0.2, 0.4, 0.6, 0.8];
        let sweep_path = format!("{}/gamma_sweep.csv", RESULTS_DIR);
        let mut sweep = File::create(&sweep_path).unwrap();
        writeln!(
            sweep,
            "gamma,step,phase,input,sim_field_goal,coherence_score,field_resistance,goal_status"
        ).unwrap();

        let mut summary_rows: Vec<(f32, f32, f32, f32)> = Vec::new();
        //     (gamma, last_sim_aligned, last_sim_attractor, avg_resistance_attractor)

        for &g in &gammas {
            let rows = run_scenario(g, &goal_hv, &aligned, &attractors);
            let mut last_aligned_sim = 0.0;
            let mut last_attractor_sim = 0.0;
            let mut sum_res_attr = 0.0;
            let mut n_attr = 0.0f32;
            for (step, input, sim, score, resistance, phase, status) in &rows {
                writeln!(
                    sweep,
                    "{:.2},{},{},{},{:.4},{:.4},{:.4},{}",
                    g, step, phase, input.replace(',', ";"),
                    sim, score, resistance, status
                ).unwrap();
                if phase == &"aligned" { last_aligned_sim = *sim; }
                if phase == &"attractor" {
                    last_attractor_sim = *sim;
                    sum_res_attr += resistance;
                    n_attr += 1.0;
                }
            }
            let avg_res_attr = if n_attr > 0.0 { sum_res_attr / n_attr } else { 0.0 };
            summary_rows.push((g, last_aligned_sim, last_attractor_sim, avg_res_attr));
            println!(
                "γ={:.2}  aligned_end={:.3}  attractor_end={:.3}  avg_resistance_attr={:.3}",
                g, last_aligned_sim, last_attractor_sim, avg_res_attr
            );
        }

        // ---------- Candidate-selection sub-test ----------
        let (correct, total) = run_selection_test(&goal_hv);
        let selection_path = format!("{}/coherence_test.csv", RESULTS_DIR);
        let mut sel = File::create(&selection_path).unwrap();
        writeln!(sel, "correct,total,accuracy").unwrap();
        writeln!(
            sel, "{},{},{:.3}",
            correct, total, correct as f32 / total as f32
        ).unwrap();
        println!("\nSelection test: {}/{} correct", correct, total);

        // ---------- Verdict ----------
        //   STRONG    : γ=0.6 attractor_end > 0.5 AND γ=0.0 attractor_end < 0.3
        //               AND selection correct >= 7
        //   MODERATE  : γ_high attractor_end - γ_low attractor_end > 0.1
        //               AND selection correct >= 7
        //   WEAK      : otherwise
        let g06 = summary_rows.iter().find(|r| (r.0 - 0.6).abs() < 1e-4).copied().unwrap_or((0.6, 0.0, 0.0, 0.0));
        let g00 = summary_rows.iter().find(|r| (r.0 - 0.0).abs() < 1e-4).copied().unwrap_or((0.0, 0.0, 0.0, 0.0));
        let g08 = summary_rows.iter().find(|r| (r.0 - 0.8).abs() < 1e-4).copied().unwrap_or((0.8, 0.0, 0.0, 0.0));
        let gap = g08.2 - g00.2;

        let strong =
            g06.2 > 0.5 && g00.2 < 0.3 && correct >= 7;
        let moderate =
            (gap > 0.1 || (g06.2 - g00.2) > 0.1) && correct >= 7;

        let verdict = if strong { "STRONG" }
            else if moderate { "MODERATE" }
            else { "WEAK" };

        let summary_path = format!("{}/summary.md", RESULTS_DIR);
        let mut sm = File::create(&summary_path).unwrap();
        writeln!(sm, "# Volition Phase 1 — Benchmark Summary").unwrap();
        writeln!(sm).unwrap();
        writeln!(sm, "## γ sweep").unwrap();
        writeln!(sm).unwrap();
        writeln!(sm, "| γ | last aligned sim | last attractor sim | avg resistance (attractor) |").unwrap();
        writeln!(sm, "|---|------------------|--------------------|----------------------------|").unwrap();
        for (g, last_a, last_attr, avg_res) in &summary_rows {
            writeln!(
                sm,
                "| {:.2} | {:.3} | {:.3} | {:.3} |",
                g, last_a, last_attr, avg_res
            ).unwrap();
        }
        writeln!(sm).unwrap();
        writeln!(sm, "## Candidate-selection").unwrap();
        writeln!(sm, "- correct: {}/{}", correct, total).unwrap();
        writeln!(sm, "- accuracy: {:.2}", correct as f32 / total as f32).unwrap();
        writeln!(sm).unwrap();
        writeln!(sm, "## Verdict: **{}**", verdict).unwrap();
        writeln!(sm).unwrap();
        writeln!(sm, "Criteria:").unwrap();
        writeln!(sm, "- STRONG: γ=0.6 attractor_end > 0.5 AND γ=0.0 attractor_end < 0.3 AND correct ≥ 7").unwrap();
        writeln!(sm, "- MODERATE: gap between low/high γ > 0.1 AND correct ≥ 7").unwrap();
        writeln!(sm, "- WEAK: otherwise").unwrap();
        writeln!(sm).unwrap();
        writeln!(sm, "Measured:").unwrap();
        writeln!(sm, "- γ=0.0 attractor_end = {:.3}", g00.2).unwrap();
        writeln!(sm, "- γ=0.6 attractor_end = {:.3}", g06.2).unwrap();
        writeln!(sm, "- γ=0.8 attractor_end = {:.3}", g08.2).unwrap();
        writeln!(sm, "- gap (γ=0.8 − γ=0.0) = {:.3}", gap).unwrap();

        println!("\nVerdict: {}", verdict);
        println!("Outputs written under {}", RESULTS_DIR);

        // The experiment always succeeds as a `cargo test` assertion — the
        // verdict (STRONG/MODERATE/WEAK) is the scientific outcome, not a
        // pass/fail. We still require that files were written.
        assert!(std::path::Path::new(&sweep_path).exists());
        assert!(std::path::Path::new(&selection_path).exists());
        assert!(std::path::Path::new(&summary_path).exists());
    }
}
