//! Experiment: Metacognition Phase 3 benchmark.
//!
//! Three sub-scenarios exercise the meta-field + self-model stack:
//!
//! - **A**: 50 steps alternating easy/hard. Expect negative correlation
//!   between perceived_difficulty and alpha_max, positive between
//!   difficulty and gamma_adjustment.
//!
//! - **B**: 100 steps across three domains (easy / hard / mixed). After
//!   ~60 steps the self-model should distinguish strong vs weak domains
//!   and its prediction error should shrink over time.
//!
//! - **C**: 10 orthogonal extreme scenarios. With a preloaded history of
//!   high cycle usage, `MetaSignal::Overloaded` should fire in ≥7/10.
//!
//! All runs go to `results/metacognition/`.

#[cfg(test)]
mod tests {
    use std::fs::{create_dir_all, File};
    use std::io::Write;

    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use crate::hdc::real::RealHV;
    use crate::language::vocabulary::Vocabulary;
    use crate::noesis::deliberation::{DeliberationConfig, DeliberationOutcome};
    use crate::noesis::meta_field::{MetaField, MetaSignal};
    use crate::noesis::self_model::SelfModel;
    use crate::noesis::volition::{GoalLevel, VolitionLayer};

    const D: usize = 1024;
    const RESULTS_DIR: &str = "results/metacognition";

    fn ensure_dir() { let _ = create_dir_all(RESULTS_DIR); }

    fn make_vocab(seed: u64, n: usize) -> Vocabulary {
        let mut v = Vocabulary::new(D, seed);
        for i in 0..n { v.get_or_create(&format!("w{}", i)); }
        v
    }

    /// Build `fast_state` with a target coherence to the goal.
    fn fast_at(goal: &RealHV, target: f32, rng: &mut StdRng) -> RealHV {
        let noise = RealHV::random(D, rng);
        let a = target.clamp(0.0, 1.0);
        let b = (1.0 - a * a).sqrt().max(0.0);
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
        let (mut num, mut dx, mut dy) = (0.0f32, 0.0f32, 0.0f32);
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

    fn signal_str(s: MetaSignal) -> &'static str {
        match s {
            MetaSignal::Normal => "Normal",
            MetaSignal::Uncertain => "Uncertain",
            MetaSignal::Overloaded => "Overloaded",
        }
    }

    fn outcome_str(o: DeliberationOutcome) -> &'static str {
        match o {
            DeliberationOutcome::Achieved => "Achieved",
            DeliberationOutcome::Converged => "Converged",
            DeliberationOutcome::ActWithUncertainty => "ActWithUncertainty",
            DeliberationOutcome::Suspended => "Suspended",
        }
    }

    // -----------------------------------------------------------------------
    // Scenario A — parameter adaptation under alternating difficulty
    // -----------------------------------------------------------------------
    fn run_scenario_a() -> (f32, f32, f32) {
        // returns (pearson_alpha, pearson_gamma, alpha_gap)
        let mut rng = StdRng::seed_from_u64(0xA001);
        let vocab = make_vocab(0xC0FFEE, 30);
        let goal = RealHV::random(D, &mut rng);
        let slow = goal.clone(); // slow_state anchored to goal

        let mut layer = VolitionLayer::new(D);
        layer.set_goal_raw(goal.clone(), GoalLevel::Tactical, 0.4, 0.85);

        let cfg = DeliberationConfig {
            base_cycles: 8,
            threshold_achieve: 0.85,
            threshold_minimum: 0.50,
            convergence_epsilon: 0.02,
            alphas: vec![0.15, 0.30, 0.50],
            vocab_k: 2,
        };

        let mut mf = MetaField::new(cfg.base_cycles, 10);
        let mut sm = SelfModel::new(cfg.base_cycles);

        let path = format!("{}/scenario_a.csv", RESULTS_DIR);
        let mut csv = File::create(&path).unwrap();
        writeln!(
            csv,
            "step,phase,initial_coherence,perceived_difficulty,alpha_max,\
             gamma_adjustment,meta_signal,cycles_used,outcome,\
             internal_coherence,field_stability,cycle_trend"
        ).unwrap();

        let mut diffs: Vec<f32> = Vec::new();
        let mut alpha_maxs: Vec<f32> = Vec::new();
        let mut gamma_adjs: Vec<f32> = Vec::new();
        let mut easy_alpha_maxs: Vec<f32> = Vec::new();
        let mut hard_alpha_maxs: Vec<f32> = Vec::new();

        for step in 0..50 {
            let phase = if step % 2 == 0 { "easy" } else { "hard" };
            let target = if phase == "easy" {
                0.80 + rng.gen::<f32>() * 0.10
            } else {
                0.05 + rng.gen::<f32>() * 0.15
            };
            let fast = fast_at(&goal, target, &mut rng);

            let (res, obs, reg) =
                layer.deliberate_with_meta(&fast, &slow, &vocab, &cfg, &mut mf, &mut sm);

            let alpha_max = reg.alpha_max();
            writeln!(
                csv,
                "{},{},{:.4},{:.4},{:.4},{:.4},{},{},{},{:.4},{:.4},{:.4}",
                step, phase, res.initial_coherence, obs.perceived_difficulty,
                alpha_max, reg.gamma_adjustment, signal_str(reg.meta_signal),
                res.cycles_used, outcome_str(res.outcome),
                obs.internal_coherence, obs.field_stability, obs.cycle_trend
            ).unwrap();

            diffs.push(obs.perceived_difficulty);
            alpha_maxs.push(alpha_max);
            gamma_adjs.push(reg.gamma_adjustment);
            if phase == "easy" {
                easy_alpha_maxs.push(alpha_max);
            } else {
                hard_alpha_maxs.push(alpha_max);
            }
        }

        let r_alpha = pearson(&diffs, &alpha_maxs);
        let r_gamma = pearson(&diffs, &gamma_adjs);
        let gap = {
            let easy_avg = easy_alpha_maxs.iter().sum::<f32>() / easy_alpha_maxs.len() as f32;
            let hard_avg = hard_alpha_maxs.iter().sum::<f32>() / hard_alpha_maxs.len() as f32;
            easy_avg - hard_avg
        };

        println!("\n[Scenario A] pearson(diff, alpha_max)={:.3}", r_alpha);
        println!("[Scenario A] pearson(diff, gamma_adj)={:.3}", r_gamma);
        println!("[Scenario A] alpha_max gap (easy - hard)={:.3}", gap);

        (r_alpha, r_gamma, gap)
    }

    // -----------------------------------------------------------------------
    // Scenario B — self-model maturation across 3 domains
    // -----------------------------------------------------------------------
    fn run_scenario_b() -> (bool, f32, f32) {
        // returns (predicts_correctly_after_60, err_first30, err_last30)
        let mut rng = StdRng::seed_from_u64(0xB002);
        let vocab = make_vocab(0xBBB0, 30);
        let slow = RealHV::random(D, &mut rng); // neutral slow anchor

        let centroid_a = RealHV::random(D, &mut rng); // easy
        let centroid_b = RealHV::random(D, &mut rng); // hard
        let centroid_c = RealHV::random(D, &mut rng); // mixed

        let cfg = DeliberationConfig {
            base_cycles: 8,
            threshold_achieve: 0.85,
            threshold_minimum: 0.50,
            convergence_epsilon: 0.02,
            alphas: vec![0.15, 0.30, 0.50],
            vocab_k: 2,
        };

        let path = format!("{}/scenario_b.csv", RESULTS_DIR);
        let mut csv = File::create(&path).unwrap();
        writeln!(
            csv,
            "step,domain,predicted_difficulty,actual_difficulty,prediction_error,\
             suspension_rate_rolling,strong_count,weak_count"
        ).unwrap();

        let mut mf = MetaField::new(cfg.base_cycles, 10);
        let mut sm = SelfModel::new(cfg.base_cycles);
        let mut layer = VolitionLayer::new(D);
        layer.set_goal_raw(centroid_a.clone(), GoalLevel::Tactical, 0.4, 0.85);
        // Goal gets swapped each step to the current domain centroid so
        // that the fast_state's similarity to the goal varies naturally.

        let mut prediction_errors: Vec<f32> = Vec::new();
        let mut err_first30 = 0.0f32;
        let mut err_last30 = 0.0f32;
        let mut count_first30 = 0.0f32;
        let mut count_last30 = 0.0f32;

        // Helper to build a fast state inside a domain family with given coh.
        let mut step_index = 0usize;
        let run_steps = |range_name: &str,
                         n: usize,
                         centroid: &RealHV,
                         coh_range: (f32, f32),
                         rng: &mut StdRng,
                         layer: &mut VolitionLayer,
                         mf: &mut MetaField,
                         sm: &mut SelfModel,
                         csv: &mut File,
                         prediction_errors: &mut Vec<f32>,
                         err_first30: &mut f32,
                         err_last30: &mut f32,
                         count_first30: &mut f32,
                         count_last30: &mut f32,
                         step_index: &mut usize| {
            for _ in 0..n {
                // Build a new fast sample in the family.
                let coh = coh_range.0 + rng.gen::<f32>() * (coh_range.1 - coh_range.0);
                // Re-point the active goal to this centroid for this step.
                layer.active_goal_id = Some("g0".into());
                // (layer.goals.get_mut can't happen — just reuse raw set)
                // For simplicity we create a fresh layer-wide goal id each
                // time: set_goal_raw pushes a new goal under a new id,
                // increases memory but is bounded by 100 total steps.
                let gid = layer.goals.add_goal_raw(
                    GoalLevel::Tactical, centroid.clone(), None, 0.4, 0.85,
                );
                layer.active_goal_id = Some(gid);

                let fast = fast_at(centroid, coh, rng);

                // Prediction BEFORE the update.
                let pred = sm.predict_difficulty(centroid);

                let (res, obs, _reg) = layer.deliberate_with_meta(
                    &fast, &slow, &vocab, &cfg, mf, sm
                );
                let actual = obs.perceived_difficulty;
                let err = (pred - actual).abs();

                writeln!(
                    csv,
                    "{},{},{:.4},{:.4},{:.4},{:.4},{},{}",
                    *step_index, range_name, pred, actual, err,
                    sm.suspension_rate,
                    sm.strong_domains.len(), sm.weak_domains.len()
                ).unwrap();

                prediction_errors.push(err);
                if *step_index < 30 {
                    *err_first30 += err;
                    *count_first30 += 1.0;
                }
                if *step_index >= 70 {
                    *err_last30 += err;
                    *count_last30 += 1.0;
                }
                // Silence unused in some paths.
                let _ = res;
                *step_index += 1;
            }
        };

        // 30 easy (Domain A)
        run_steps(
            "A", 30, &centroid_a, (0.75, 0.92), &mut rng,
            &mut layer, &mut mf, &mut sm, &mut csv,
            &mut prediction_errors, &mut err_first30, &mut err_last30,
            &mut count_first30, &mut count_last30, &mut step_index,
        );
        // 30 hard (Domain B)
        run_steps(
            "B", 30, &centroid_b, (0.05, 0.22), &mut rng,
            &mut layer, &mut mf, &mut sm, &mut csv,
            &mut prediction_errors, &mut err_first30, &mut err_last30,
            &mut count_first30, &mut count_last30, &mut step_index,
        );
        // 40 mixed (Domain C)
        run_steps(
            "C", 40, &centroid_c, (0.10, 0.85), &mut rng,
            &mut layer, &mut mf, &mut sm, &mut csv,
            &mut prediction_errors, &mut err_first30, &mut err_last30,
            &mut count_first30, &mut count_last30, &mut step_index,
        );

        let pred_a_after_60 = sm.predict_difficulty(&centroid_a);
        let pred_b_after_60 = sm.predict_difficulty(&centroid_b);
        let ok = pred_a_after_60 < pred_b_after_60;

        let e1 = if count_first30 > 0.0 { err_first30 / count_first30 } else { 0.0 };
        let e2 = if count_last30  > 0.0 { err_last30  / count_last30  } else { 0.0 };

        println!(
            "\n[Scenario B] predict(A)={:.3}  predict(B)={:.3}  ok={}",
            pred_a_after_60, pred_b_after_60, ok
        );
        println!(
            "[Scenario B] err_first30={:.3}  err_last30={:.3}",
            e1, e2
        );

        (ok, e1, e2)
    }

    // -----------------------------------------------------------------------
    // Scenario C — overloaded signal under extreme difficulty
    // -----------------------------------------------------------------------
    fn run_scenario_c() -> usize {
        // returns number of Overloaded / 10
        let mut rng = StdRng::seed_from_u64(0xC003);
        let vocab = make_vocab(0xCCC0, 30);

        let path = format!("{}/scenario_c.csv", RESULTS_DIR);
        let mut csv = File::create(&path).unwrap();
        writeln!(
            csv,
            "scenario_id,perceived_difficulty,cycle_trend,meta_signal,outcome"
        ).unwrap();

        let mut overloaded = 0usize;

        for id in 0..10 {
            let goal = RealHV::random(D, &mut rng);
            // Slow orthogonal to goal → low internal coherence from the start.
            let slow = RealHV::random(D, &mut rng);
            let fast = RealHV::random(D, &mut rng); // orthogonal to both

            let cfg = DeliberationConfig {
                base_cycles: 4, // tight budget, cycle_trend will explode
                threshold_achieve: 0.99, // unreachable
                threshold_minimum: 0.90, // very high
                convergence_epsilon: 0.001,
                alphas: vec![0.05, 0.10, 0.15],
                vocab_k: 1,
            };

            let mut mf = MetaField::new(cfg.base_cycles, 10);
            // Preload cycle history with high values — simulate a prior run
            // of hard problems, so cycle_trend > 2.0 from the first observe.
            mf.preload_cycles([10, 10, 10, 10, 10]);

            let mut sm = SelfModel::new(cfg.base_cycles);
            let mut layer = VolitionLayer::new(D);
            layer.set_goal_raw(goal.clone(), GoalLevel::Operational, 0.2, 0.99);

            // Seed the meta-field with a different fast frame so that the
            // stability term on the real observation is not capped at 1.0.
            let seed_fast = RealHV::random(D, &mut rng);
            let _ = mf.observe(&seed_fast, &slow, 10);

            let (res, obs, reg) =
                layer.deliberate_with_meta(&fast, &slow, &vocab, &cfg, &mut mf, &mut sm);

            let is_overloaded = matches!(reg.meta_signal, MetaSignal::Overloaded);
            if is_overloaded { overloaded += 1; }

            writeln!(
                csv,
                "{},{:.4},{:.4},{},{}",
                id, obs.perceived_difficulty, obs.cycle_trend,
                signal_str(reg.meta_signal), outcome_str(res.outcome)
            ).unwrap();
        }

        println!("\n[Scenario C] Overloaded = {}/10", overloaded);
        overloaded
    }

    // -----------------------------------------------------------------------
    // Main benchmark test
    // -----------------------------------------------------------------------
    #[test]
    fn test_metacognition_benchmark() {
        println!("\n=== Experiment: Metacognition Phase 3 benchmark ===\n");
        ensure_dir();

        let (r_alpha, r_gamma, gap) = run_scenario_a();
        let (pred_ok, err1, err2) = run_scenario_b();
        let overloaded = run_scenario_c();

        let strong =
            r_alpha < -0.5
            && r_gamma > 0.5
            && pred_ok
            && overloaded >= 7;

        let moderate =
            r_alpha < -0.3
            && pred_ok
            && overloaded >= 5;

        let verdict = if strong { "STRONG" }
            else if moderate { "MODERATE" }
            else { "WEAK" };

        let summary_path = format!("{}/summary.md", RESULTS_DIR);
        let mut sm = File::create(&summary_path).unwrap();
        writeln!(sm, "# Metacognition Phase 3 — Benchmark Summary").unwrap();
        writeln!(sm).unwrap();
        writeln!(sm, "## Scenario A — Parameter adaptation").unwrap();
        writeln!(sm, "- pearson(difficulty, alpha_max): {:.3}", r_alpha).unwrap();
        writeln!(sm, "- pearson(difficulty, gamma_adjustment): {:.3}", r_gamma).unwrap();
        writeln!(sm, "- alpha_max gap (easy - hard): {:.3}", gap).unwrap();
        writeln!(sm).unwrap();
        writeln!(sm, "## Scenario B — Self-model maturation").unwrap();
        writeln!(sm, "- predict(A) < predict(B) after 100 steps: {}", pred_ok).unwrap();
        writeln!(sm, "- prediction error first 30 steps: {:.3}", err1).unwrap();
        writeln!(sm, "- prediction error last 30 steps:  {:.3}", err2).unwrap();
        writeln!(sm).unwrap();
        writeln!(sm, "## Scenario C — Overloaded signal").unwrap();
        writeln!(sm, "- Overloaded count: {}/10", overloaded).unwrap();
        writeln!(sm).unwrap();
        writeln!(sm, "## Verdict: **{}**", verdict).unwrap();
        writeln!(sm).unwrap();
        writeln!(sm, "Criteria:").unwrap();
        writeln!(sm, "- STRONG: r_alpha < −0.5 AND r_gamma > 0.5 AND predict(A)<predict(B) AND overloaded ≥ 7/10").unwrap();
        writeln!(sm, "- MODERATE: r_alpha < −0.3 AND predict(A)<predict(B) AND overloaded ≥ 5/10").unwrap();
        writeln!(sm, "- WEAK: otherwise").unwrap();

        println!("\nVerdict: {}", verdict);
        println!("Outputs in {}", RESULTS_DIR);

        assert!(std::path::Path::new(&format!("{}/scenario_a.csv", RESULTS_DIR)).exists());
        assert!(std::path::Path::new(&format!("{}/scenario_b.csv", RESULTS_DIR)).exists());
        assert!(std::path::Path::new(&format!("{}/scenario_c.csv", RESULTS_DIR)).exists());
        assert!(std::path::Path::new(&summary_path).exists());
    }
}
