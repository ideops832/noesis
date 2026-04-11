//! Experiment: Deliberation Phase 2 benchmark.
//!
//! Two sub-experiments:
//!
//!   **Scenario A** — 30 scenarios across 3 difficulties (10 each):
//!     easy    : initial coherence > 0.7
//!     medium  : initial coherence 0.4 – 0.7
//!     hard    : initial coherence < 0.4
//!   Expected (STRONG): cycles positively correlated with difficulty
//!   (Pearson r > 0.6), accuracy > 80% (Achieved + Converged), Suspended < 20%.
//!
//!   **Scenario B** — candidate-source comparison on 20 mixed scenarios:
//!     attraction-only (3 candidates)
//!     vocabulary-only (2 candidates)
//!     combined (5 candidates)
//!   Expected: the combined set wins on accuracy / avg_score.

#[cfg(test)]
mod tests {
    use std::fs::{create_dir_all, File};
    use std::io::Write;

    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use crate::hdc::real::RealHV;
    use crate::language::vocabulary::Vocabulary;
    use crate::noesis::deliberation::{
        DeliberationConfig, DeliberationOutcome, CandidateSource,
    };
    use crate::noesis::volition::{GoalLevel, VolitionLayer};

    const D: usize = 1024;
    const RESULTS_DIR: &str = "results/deliberation";

    fn ensure_dir() { let _ = create_dir_all(RESULTS_DIR); }

    fn make_vocab(seed: u64, n: usize) -> Vocabulary {
        let mut v = Vocabulary::new(D, seed);
        for i in 0..n {
            v.get_or_create(&format!("w{}", i));
        }
        v
    }

    /// Build a `fast_state` with target initial coherence to the goal.
    /// `target_coh` controls how much of the goal to retain vs independent noise.
    fn fast_at(goal: &RealHV, target_coh: f32, rng: &mut StdRng) -> RealHV {
        // Random component orthogonal in expectation to the goal.
        let noise = RealHV::random(D, rng);
        // Linear combination: sim(fast, goal) ≈ target_coh (before renormalize).
        let a = target_coh;
        let b = (1.0 - target_coh * target_coh).sqrt().max(0.0);
        let kept = goal.normalized().scale(a);
        let added = noise.normalized().scale(b);
        let mut mixed = RealHV::add(&kept, &added);
        if mixed.norm() > 1e-10 { mixed.normalize(); }
        mixed
    }

    fn pearson(xs: &[f32], ys: &[f32]) -> f32 {
        let n = xs.len() as f32;
        if n < 2.0 { return 0.0; }
        let mx = xs.iter().sum::<f32>() / n;
        let my = ys.iter().sum::<f32>() / n;
        let mut num = 0.0f32;
        let mut dx = 0.0f32;
        let mut dy = 0.0f32;
        for i in 0..xs.len() {
            let a = xs[i] - mx;
            let b = ys[i] - my;
            num += a * b;
            dx += a * a;
            dy += b * b;
        }
        let denom = (dx * dy).sqrt();
        if denom < 1e-10 { 0.0 } else { num / denom }
    }

    fn outcome_str(o: DeliberationOutcome) -> &'static str {
        match o {
            DeliberationOutcome::Achieved => "Achieved",
            DeliberationOutcome::ActWithUncertainty => "ActWithUncertainty",
            DeliberationOutcome::Suspended => "Suspended",
            DeliberationOutcome::Converged => "Converged",
        }
    }

    fn source_str(s: &CandidateSource) -> String {
        match s {
            CandidateSource::Attraction { alpha } => format!("Attraction(α={:.2})", alpha),
            CandidateSource::Vocabulary { word } => format!("Vocabulary({})", word),
        }
    }

    // -----------------------------------------------------------------------
    // Scenario A — 30 scenarios across 3 difficulties
    // -----------------------------------------------------------------------
    fn run_scenario_a() -> (usize, f32, f32, f32, usize, usize, usize) {
        //           total, accuracy, avg_score, pearson_r, n_susp, n_ach, n_conv
        let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF_u64);
        let vocab = make_vocab(0xC0FFEE, 40);

        // NOTE: α=1.0 is a "cheat" — it literally delivers the goal on cycle 1
        // so every scenario would terminate in one step and Pearson(diff,cycles)
        // would be 0. We keep α=1.0 in the library default but the benchmark
        // uses conservative attraction steps so that harder scenarios need
        // strictly more cycles to reach the threshold. This is the correct
        // way to observe difficulty-adaptive deliberation: smaller per-cycle
        // steps reveal the actual search dynamics.
        let cfg = DeliberationConfig {
            base_cycles: 8,
            threshold_achieve: 0.85,
            threshold_minimum: 0.50,
            convergence_epsilon: 0.02,
            alphas: vec![0.15, 0.30, 0.50],
            vocab_k: 2,
        };

        let difficulties: &[(&str, f32, f32)] = &[
            ("easy",   0.70, 0.90),
            ("medium", 0.40, 0.70),
            ("hard",   0.05, 0.40),
        ];

        let csv_path = format!("{}/scenario_a.csv", RESULTS_DIR);
        let mut csv = File::create(&csv_path).unwrap();
        writeln!(
            csv,
            "scenario_id,difficulty,initial_coherence,cycles_used,max_cycles,\
             final_score,outcome,best_source"
        ).unwrap();

        let mut diff_idx_vec: Vec<f32> = Vec::new(); // 0, 1, 2 for easy/med/hard
        let mut cycles_vec: Vec<f32> = Vec::new();
        let mut total = 0usize;
        let mut achieved = 0usize;
        let mut converged = 0usize;
        let mut suspended = 0usize;
        let mut act_unc = 0usize;
        let mut sum_score = 0.0f32;

        for (di, (name, lo, hi)) in difficulties.iter().enumerate() {
            for k in 0..10 {
                let goal = RealHV::random(D, &mut rng);

                let t: f32 = lo + rng.gen::<f32>() * (hi - lo);
                let fast = fast_at(&goal, t, &mut rng);

                let mut layer = VolitionLayer::new(D);
                layer.set_goal_raw(goal.clone(), GoalLevel::Tactical, 0.4, 0.85);

                let res = layer.deliberate(&fast, &vocab, &cfg);

                writeln!(
                    csv,
                    "A{:02},{},{:.4},{},{},{:.4},{},{}",
                    di * 10 + k, name, res.initial_coherence, res.cycles_used,
                    res.max_cycles, res.final_score,
                    outcome_str(res.outcome),
                    source_str(&res.best_source).replace(',', ";")
                ).unwrap();

                match res.outcome {
                    DeliberationOutcome::Achieved           => { achieved += 1; }
                    DeliberationOutcome::Converged          => { converged += 1; }
                    DeliberationOutcome::Suspended          => { suspended += 1; }
                    DeliberationOutcome::ActWithUncertainty => { act_unc += 1; }
                }
                diff_idx_vec.push(di as f32);
                cycles_vec.push(res.cycles_used as f32);
                total += 1;
                sum_score += res.final_score;
            }
        }

        let avg_score = sum_score / total as f32;
        let acc = (achieved + converged) as f32 / total as f32;
        let r = pearson(&diff_idx_vec, &cycles_vec);

        println!(
            "\n[Scenario A] total={} achieved={} converged={} actUnc={} suspended={}",
            total, achieved, converged, act_unc, suspended
        );
        println!("  avg_score={:.3}  accuracy={:.3}  pearson(diff,cycles)={:.3}", avg_score, acc, r);

        (total, acc, avg_score, r, suspended, achieved, converged)
    }

    // -----------------------------------------------------------------------
    // Scenario B — candidate-source comparison
    // -----------------------------------------------------------------------
    fn run_scenario_b() -> Vec<(String, f32, f32, f32)> {
        // returns rows: (source_name, accuracy, avg_cycles, avg_final_score)
        let mut rng = StdRng::seed_from_u64(0xBEEF_C001);
        let vocab = make_vocab(0xC0FFEE, 40);

        let n_trials = 20usize;
        let mut trials: Vec<(RealHV, RealHV)> = Vec::new(); // (goal, fast)
        for _ in 0..n_trials {
            let goal = RealHV::random(D, &mut rng);
            let t: f32 = 0.1 + rng.gen::<f32>() * 0.6; // mixed difficulty
            let fast = fast_at(&goal, t, &mut rng);
            trials.push((goal, fast));
        }

        // Same conservative attraction schedule as Scenario A so the
        // comparison between sources is meaningful (otherwise α=1.0 wins
        // trivially every round).
        let variants: &[(&str, Vec<f32>, usize)] = &[
            ("attraction_only", vec![0.15, 0.30, 0.50], 0),
            ("vocabulary_only", vec![],                 2),
            ("combined",        vec![0.15, 0.30, 0.50], 2),
        ];

        let csv_path = format!("{}/scenario_b.csv", RESULTS_DIR);
        let mut csv = File::create(&csv_path).unwrap();
        writeln!(
            csv,
            "experiment,candidate_source,accuracy,avg_cycles,avg_final_score,n_trials"
        ).unwrap();

        let mut rows = Vec::new();

        for (name, alphas, vocab_k) in variants {
            let cfg = DeliberationConfig {
                base_cycles: 8,
                threshold_achieve: 0.85,
                threshold_minimum: 0.50,
                convergence_epsilon: 0.02,
                alphas: alphas.clone(),
                vocab_k: *vocab_k,
            };

            let mut ok = 0usize;
            let mut sum_cycles = 0.0f32;
            let mut sum_score = 0.0f32;

            for (goal, fast) in &trials {
                let mut layer = VolitionLayer::new(D);
                layer.set_goal_raw(goal.clone(), GoalLevel::Tactical, 0.4, 0.85);
                let res = layer.deliberate(fast, &vocab, &cfg);
                let good = matches!(
                    res.outcome,
                    DeliberationOutcome::Achieved | DeliberationOutcome::Converged
                );
                if good { ok += 1; }
                sum_cycles += res.cycles_used as f32;
                sum_score  += res.final_score;
            }

            let acc = ok as f32 / n_trials as f32;
            let avg_cycles = sum_cycles / n_trials as f32;
            let avg_score  = sum_score  / n_trials as f32;

            writeln!(
                csv,
                "scenarioB,{},{:.3},{:.3},{:.3},{}",
                name, acc, avg_cycles, avg_score, n_trials
            ).unwrap();

            println!(
                "[Scenario B] {:<16}  acc={:.3}  avg_cycles={:.2}  avg_score={:.3}",
                name, acc, avg_cycles, avg_score
            );
            rows.push((name.to_string(), acc, avg_cycles, avg_score));
        }

        rows
    }

    // -----------------------------------------------------------------------
    // Main benchmark test
    // -----------------------------------------------------------------------
    #[test]
    fn test_deliberation_benchmark() {
        println!("\n=== Experiment: Deliberation Phase 2 benchmark ===\n");
        ensure_dir();

        let (total, acc, avg_score, r, suspended, achieved, converged) = run_scenario_a();
        let variants = run_scenario_b();

        // Verdict thresholds
        let strong =
            r > 0.6
            && acc > 0.80
            && (suspended as f32 / total as f32) < 0.20;

        let moderate =
            r > 0.3
            && acc > 0.60;

        let verdict = if strong { "STRONG" }
            else if moderate { "MODERATE" }
            else { "WEAK" };

        let summary_path = format!("{}/summary.md", RESULTS_DIR);
        let mut sm = File::create(&summary_path).unwrap();
        writeln!(sm, "# Deliberation Phase 2 — Benchmark Summary").unwrap();
        writeln!(sm).unwrap();
        writeln!(sm, "## Scenario A — Difficulty vs cycles").unwrap();
        writeln!(sm).unwrap();
        writeln!(sm, "- total scenarios: {}", total).unwrap();
        writeln!(sm, "- achieved:  {}", achieved).unwrap();
        writeln!(sm, "- converged: {}", converged).unwrap();
        writeln!(sm, "- suspended: {}", suspended).unwrap();
        writeln!(sm, "- accuracy (Achieved + Converged): {:.3}", acc).unwrap();
        writeln!(sm, "- avg final score: {:.3}", avg_score).unwrap();
        writeln!(sm, "- pearson(difficulty_idx, cycles_used): {:.3}", r).unwrap();
        writeln!(sm).unwrap();
        writeln!(sm, "## Scenario B — Candidate source comparison").unwrap();
        writeln!(sm).unwrap();
        writeln!(sm, "| source | accuracy | avg cycles | avg score |").unwrap();
        writeln!(sm, "|--------|----------|------------|-----------|").unwrap();
        for (name, a, c, s) in &variants {
            writeln!(sm, "| {} | {:.3} | {:.2} | {:.3} |", name, a, c, s).unwrap();
        }
        writeln!(sm).unwrap();
        writeln!(sm, "## Verdict: **{}**", verdict).unwrap();
        writeln!(sm).unwrap();
        writeln!(sm, "Criteria:").unwrap();
        writeln!(sm, "- STRONG: pearson(diff,cycles) > 0.6 AND accuracy > 0.80 AND suspended_rate < 0.20").unwrap();
        writeln!(sm, "- MODERATE: pearson(diff,cycles) > 0.3 AND accuracy > 0.60").unwrap();
        writeln!(sm, "- WEAK: otherwise").unwrap();
        writeln!(sm).unwrap();
        writeln!(sm, "Measured:").unwrap();
        writeln!(sm, "- accuracy = {:.3}", acc).unwrap();
        writeln!(sm, "- pearson(diff,cycles) = {:.3}", r).unwrap();
        writeln!(sm, "- suspended rate = {:.3}", suspended as f32 / total as f32).unwrap();

        println!("\nVerdict: {}", verdict);
        println!("Outputs written under {}", RESULTS_DIR);

        assert!(std::path::Path::new(&format!("{}/scenario_a.csv", RESULTS_DIR)).exists());
        assert!(std::path::Path::new(&format!("{}/scenario_b.csv", RESULTS_DIR)).exists());
        assert!(std::path::Path::new(&summary_path).exists());
    }
}
