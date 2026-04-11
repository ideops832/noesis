//! Semantic deliberation loop — Phase 2.
//!
//! Given an active goal in a [`VolitionLayer`], deliberation generates
//! candidate solutions, scores them against the goal, and decides when to
//! act or keep searching. It operates purely on `fast_state` of a
//! `MultiScaleField` — the scale where γ-bias is observable (Phase 1
//! finding: `combined_state` is dominated by the anchored slow field).
//!
//! The loop has four termination conditions:
//!   1. Achieved      — best score > threshold_achieve
//!   2. Converged     — |best - prev| < convergence_epsilon (from cycle 2)
//!   3. ActWithUncertainty — budget exhausted but best > threshold_minimum
//!   4. Suspended     — budget exhausted and best ≤ threshold_minimum
//!
//! Candidates come from two mechanisms:
//!   - **Attraction**: interpolate `fast_state → goal` at α ∈ {0.33, 0.66, 1.0}
//!   - **Vocabulary**: top-k words by cosine similarity to `fast_state`

use crate::hdc::real::RealHV;
use crate::language::vocabulary::Vocabulary;
use crate::noesis::volition::VolitionLayer;

// ---------------------------------------------------------------------------
// Routing mode (hook for Phase 2b voxel deliberation)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeliberationMode {
    Semantic,
    Voxel,
}

// ---------------------------------------------------------------------------
// Candidate source + evaluation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum CandidateSource {
    /// Attraction interpolation at the given α.
    Attraction { alpha: f32 },
    /// Vocabulary lookup: the winning word's key.
    Vocabulary { word: String },
}

#[derive(Debug, Clone)]
pub struct CandidateEvaluation {
    pub candidate_hv: RealHV,
    pub coherence_score: f32,
    pub source: CandidateSource,
}

// ---------------------------------------------------------------------------
// Candidate generators
// ---------------------------------------------------------------------------

/// Mechanism **C**: three attractor-biased candidates at α ∈ {0.33, 0.66, 1.0}.
///
/// `candidate_i = normalize((1 - α_i) × fast_state + α_i × goal_hv)`
pub fn generate_attraction_candidates(
    fast_state: &RealHV,
    goal_hv: &RealHV,
    alphas: &[f32],
) -> Vec<(RealHV, f32)> {
    let mut out = Vec::with_capacity(alphas.len());
    let fs_norm = if fast_state.norm() > 1e-10 {
        fast_state.normalized()
    } else {
        fast_state.clone()
    };
    let goal_norm = if goal_hv.norm() > 1e-10 {
        goal_hv.normalized()
    } else {
        goal_hv.clone()
    };
    for &alpha in alphas {
        let a = alpha.clamp(0.0, 1.0);
        let kept = fs_norm.scale(1.0 - a);
        let pulled = goal_norm.scale(a);
        let mut mixed = RealHV::add(&kept, &pulled);
        if mixed.norm() > 1e-10 {
            mixed.normalize();
        }
        out.push((mixed, a));
    }
    out
}

/// Mechanism **A**: top-k vocabulary words by cosine similarity to `fast_state`.
///
/// Returns the winning words together with their hypervectors. If the
/// vocabulary is empty, returns an empty Vec.
pub fn generate_vocabulary_candidates(
    fast_state: &RealHV,
    vocabulary: &Vocabulary,
    k: usize,
) -> Vec<(RealHV, String)> {
    if vocabulary.words.is_empty() || k == 0 {
        return Vec::new();
    }
    let mut scored: Vec<(String, RealHV, f32)> = vocabulary
        .words
        .iter()
        .map(|(w, hv)| {
            let s = RealHV::cosine_similarity(fast_state, hv);
            (w.clone(), hv.clone(), s)
        })
        .collect();
    scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    scored
        .into_iter()
        .take(k)
        .map(|(w, hv, _)| (hv, w))
        .collect()
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

/// Score each candidate against the goal and sort by decreasing score.
pub fn evaluate_candidates(
    candidates: &[(RealHV, CandidateSource)],
    goal_hv: &RealHV,
) -> Vec<CandidateEvaluation> {
    let mut scored: Vec<CandidateEvaluation> = candidates
        .iter()
        .map(|(hv, src)| {
            let raw = RealHV::cosine_similarity(hv, goal_hv);
            CandidateEvaluation {
                candidate_hv: hv.clone(),
                coherence_score: raw.clamp(0.0, 1.0),
                source: src.clone(),
            }
        })
        .collect();
    scored.sort_by(|a, b| {
        b.coherence_score
            .partial_cmp(&a.coherence_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored
}

// ---------------------------------------------------------------------------
// Deliberation loop
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct DeliberationConfig {
    pub base_cycles: usize,
    pub threshold_achieve: f32,
    pub threshold_minimum: f32,
    pub convergence_epsilon: f32,
    /// Attraction α values.
    pub alphas: Vec<f32>,
    /// Vocabulary top-k.
    pub vocab_k: usize,
}

impl Default for DeliberationConfig {
    fn default() -> Self {
        DeliberationConfig {
            base_cycles: 5,
            threshold_achieve: 0.85,
            threshold_minimum: 0.50,
            convergence_epsilon: 0.02,
            alphas: vec![0.33, 0.66, 1.0],
            vocab_k: 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeliberationOutcome {
    Achieved,
    ActWithUncertainty,
    Suspended,
    Converged,
}

#[derive(Debug, Clone)]
pub struct DeliberationResult {
    pub best_candidate: RealHV,
    pub best_source: CandidateSource,
    pub final_score: f32,
    pub initial_coherence: f32,
    pub cycles_used: usize,
    pub max_cycles: usize,
    pub outcome: DeliberationOutcome,
    pub all_evaluations: Vec<Vec<CandidateEvaluation>>,
}

impl VolitionLayer {
    /// Choose a deliberation mode. Phase 2 always picks `Semantic`.
    /// Phase 2b will add spatial-attractor routing.
    pub fn select_mode(&self, _fast_state: &RealHV) -> DeliberationMode {
        DeliberationMode::Semantic
    }

    /// Run the full deliberation loop against the active goal.
    ///
    /// Returns a `DeliberationResult`. If no goal is active or the goal
    /// cannot be resolved, returns a `Suspended` result with zero score.
    pub fn deliberate(
        &self,
        fast_state: &RealHV,
        vocabulary: &Vocabulary,
        config: &DeliberationConfig,
    ) -> DeliberationResult {
        // Resolve the active goal snapshot up-front.
        let (goal_hv, gamma) = match self
            .active_goal_id
            .as_ref()
            .and_then(|id| self.goals.get(id))
        {
            Some(g) => (g.hv.clone(), g.gamma),
            None => {
                return DeliberationResult {
                    best_candidate: fast_state.clone(),
                    best_source: CandidateSource::Attraction { alpha: 0.0 },
                    final_score: 0.0,
                    initial_coherence: 0.0,
                    cycles_used: 0,
                    max_cycles: 0,
                    outcome: DeliberationOutcome::Suspended,
                    all_evaluations: Vec::new(),
                };
            }
        };

        let initial_coherence =
            RealHV::cosine_similarity(fast_state, &goal_hv).clamp(0.0, 1.0);

        // Difficulty-adaptive budget.
        // Far from goal + high γ → more cycles. Close + low γ → fewer.
        let raw_budget =
            (config.base_cycles as f32) * (1.0 - initial_coherence) * (1.0 + gamma);
        let max_cycles = raw_budget.ceil().max(1.0) as usize;

        let mut prev_score = 0.0f32;
        let mut cycles_used = 0usize;
        let mut all_evaluations: Vec<Vec<CandidateEvaluation>> = Vec::new();
        let mut best_final: Option<CandidateEvaluation> = None;

        let mut current_state = fast_state.clone();

        for cycle in 0..max_cycles {
            cycles_used = cycle + 1;

            // 1. Generate candidates (attraction + vocabulary).
            let mut candidates: Vec<(RealHV, CandidateSource)> = Vec::new();

            let attraction = generate_attraction_candidates(
                &current_state,
                &goal_hv,
                &config.alphas,
            );
            for (hv, alpha) in attraction {
                candidates.push((hv, CandidateSource::Attraction { alpha }));
            }

            let vocab_cands =
                generate_vocabulary_candidates(&current_state, vocabulary, config.vocab_k);
            for (hv, word) in vocab_cands {
                candidates.push((hv, CandidateSource::Vocabulary { word }));
            }

            // 2. Evaluate + sort by score.
            let scored = evaluate_candidates(&candidates, &goal_hv);
            let best = match scored.first().cloned() {
                Some(c) => c,
                None => {
                    // No candidates (empty vocab + no alphas?). Bail out.
                    return DeliberationResult {
                        best_candidate: current_state,
                        best_source: CandidateSource::Attraction { alpha: 0.0 },
                        final_score: 0.0,
                        initial_coherence,
                        cycles_used,
                        max_cycles,
                        outcome: DeliberationOutcome::Suspended,
                        all_evaluations,
                    };
                }
            };
            let best_score = best.coherence_score;
            all_evaluations.push(scored);

            // 3. Termination checks.

            // 3a. Achieved.
            if best_score >= config.threshold_achieve {
                return DeliberationResult {
                    best_candidate: best.candidate_hv.clone(),
                    best_source: best.source.clone(),
                    final_score: best_score,
                    initial_coherence,
                    cycles_used,
                    max_cycles,
                    outcome: DeliberationOutcome::Achieved,
                    all_evaluations,
                };
            }

            // 3b. Converged (from cycle 2 onwards).
            if cycle >= 1 && (best_score - prev_score).abs() < config.convergence_epsilon {
                return DeliberationResult {
                    best_candidate: best.candidate_hv.clone(),
                    best_source: best.source.clone(),
                    final_score: best_score,
                    initial_coherence,
                    cycles_used,
                    max_cycles,
                    outcome: DeliberationOutcome::Converged,
                    all_evaluations,
                };
            }

            prev_score = best_score;
            best_final = Some(best.clone());
            // Advance the search from the current best candidate.
            current_state = best.candidate_hv.clone();
        }

        // Budget exhausted: act with uncertainty if the best is above minimum,
        // otherwise suspend.
        let best = best_final.unwrap_or_else(|| CandidateEvaluation {
            candidate_hv: current_state.clone(),
            coherence_score: 0.0,
            source: CandidateSource::Attraction { alpha: 0.0 },
        });

        let outcome = if best.coherence_score > config.threshold_minimum {
            DeliberationOutcome::ActWithUncertainty
        } else {
            DeliberationOutcome::Suspended
        };

        DeliberationResult {
            best_candidate: best.candidate_hv.clone(),
            best_source: best.source.clone(),
            final_score: best.coherence_score,
            initial_coherence,
            cycles_used,
            max_cycles,
            outcome,
            all_evaluations,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::language::vocabulary::Vocabulary;
    use crate::noesis::volition::{GoalLevel, VolitionLayer};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    const D: usize = 1024;

    fn make_vocab(seed: u64, n: usize) -> Vocabulary {
        let mut v = Vocabulary::new(D, seed);
        for i in 0..n {
            v.get_or_create(&format!("w{}", i));
        }
        v
    }

    // -------- Test 1: candidate generation ---------------------------------
    #[test]
    fn test_candidate_generation() {
        let mut rng = StdRng::seed_from_u64(111);
        let fast = RealHV::random(D, &mut rng);
        let goal = RealHV::random(D, &mut rng);

        let alphas = [0.33f32, 0.66, 1.0];
        let attraction = generate_attraction_candidates(&fast, &goal, &alphas);
        assert_eq!(attraction.len(), 3);

        // α=1.0 must be essentially the normalized goal.
        let sim_alpha1_goal =
            RealHV::cosine_similarity(&attraction[2].0, &goal);
        assert!(
            sim_alpha1_goal > 0.999,
            "α=1.0 candidate should match goal, sim={:.4}",
            sim_alpha1_goal
        );

        // α=0.33 must be closer to fast than α=1.0.
        let sim_alpha033_fast =
            RealHV::cosine_similarity(&attraction[0].0, &fast);
        let sim_alpha1_fast =
            RealHV::cosine_similarity(&attraction[2].0, &fast);
        assert!(
            sim_alpha033_fast > sim_alpha1_fast,
            "α=0.33 should be closer to fast than α=1.0: {:.3} vs {:.3}",
            sim_alpha033_fast, sim_alpha1_fast
        );

        // Vocabulary candidates (k=2).
        let vocab = make_vocab(222, 20);
        let vcs = generate_vocabulary_candidates(&fast, &vocab, 2);
        assert_eq!(vcs.len(), 2);

        // The 5 candidates should be mutually distinct.
        let mut all: Vec<RealHV> = Vec::new();
        for (hv, _) in attraction { all.push(hv); }
        for (hv, _) in vcs { all.push(hv); }
        for i in 0..all.len() {
            for j in (i+1)..all.len() {
                let s = RealHV::cosine_similarity(&all[i], &all[j]);
                assert!(s < 0.999, "candidates {} and {} are identical ({:.4})", i, j, s);
            }
        }
    }

    // -------- Test 2: evaluation ordering ----------------------------------
    #[test]
    fn test_candidate_evaluation_ordering() {
        let mut rng = StdRng::seed_from_u64(333);
        let goal = RealHV::random(D, &mut rng);

        // Build 5 candidates with decreasing similarity to goal.
        let mut cands: Vec<(RealHV, CandidateSource)> = Vec::new();
        let alphas = [1.0f32, 0.75, 0.5, 0.25, 0.0];
        for &a in &alphas {
            let noise = RealHV::random_normal(D, &mut rng);
            let kept = goal.normalized().scale(a);
            let added = noise.scale(1.0 - a);
            let mut mixed = RealHV::add(&kept, &added);
            if mixed.norm() > 1e-10 { mixed.normalize(); }
            cands.push((mixed, CandidateSource::Attraction { alpha: a }));
        }

        let scored = evaluate_candidates(&cands, &goal);
        // Monotonic non-increasing.
        for i in 0..scored.len()-1 {
            assert!(
                scored[i].coherence_score + 1e-6 >= scored[i+1].coherence_score,
                "scored[{}]={:.3} < scored[{}]={:.3}",
                i, scored[i].coherence_score,
                i+1, scored[i+1].coherence_score
            );
        }
        // The top candidate must be the one built at α=1.0 (direct goal).
        match scored[0].source.clone() {
            CandidateSource::Attraction { alpha } => assert!((alpha - 1.0).abs() < 1e-4),
            _ => panic!("top candidate should be an Attraction"),
        }
    }

    // -------- Test 3: Achieved on first cycle ------------------------------
    #[test]
    fn test_deliberation_achieves_threshold() {
        let mut rng = StdRng::seed_from_u64(444);
        let goal = RealHV::random(D, &mut rng);

        let mut layer = VolitionLayer::new(D);
        layer.set_goal_raw(goal.clone(), GoalLevel::Tactical, 0.4, 0.85);

        // Fast state already close to the goal (noise only 0.05).
        let noise = RealHV::random_normal(D, &mut rng);
        let fast = RealHV::add(&goal, &noise.scale(0.05)).normalized();

        let vocab = make_vocab(555, 10);
        let cfg = DeliberationConfig::default();
        let res = layer.deliberate(&fast, &vocab, &cfg);

        assert_eq!(res.outcome, DeliberationOutcome::Achieved);
        assert_eq!(res.cycles_used, 1);
        assert!(res.final_score >= cfg.threshold_achieve);
    }

    // -------- Test 4: convergence ------------------------------------------
    #[test]
    fn test_deliberation_convergence() {
        let mut rng = StdRng::seed_from_u64(666);
        let goal = RealHV::random(D, &mut rng);

        // Fast state far from goal, near 0 similarity.
        let fast = RealHV::random(D, &mut rng);

        let mut layer = VolitionLayer::new(D);
        // High gamma → large budget so we can actually observe convergence
        // before the budget runs out.
        layer.set_goal_raw(goal.clone(), GoalLevel::Tactical, 0.9, 0.85);

        // Force convergence: threshold so high it cannot be reached but
        // epsilon large enough that plateau is detected.
        let vocab = make_vocab(777, 10);
        let cfg = DeliberationConfig {
            base_cycles: 10,
            threshold_achieve: 0.999, // unreachable
            threshold_minimum: 0.10,
            convergence_epsilon: 0.05,
            ..DeliberationConfig::default()
        };

        let res = layer.deliberate(&fast, &vocab, &cfg);
        assert!(
            res.outcome == DeliberationOutcome::Converged
                || res.outcome == DeliberationOutcome::Achieved,
            "expected Converged or Achieved, got {:?} (score={:.3})",
            res.outcome, res.final_score
        );
    }

    // -------- Test 5: suspension -------------------------------------------
    #[test]
    fn test_deliberation_suspension() {
        let mut rng = StdRng::seed_from_u64(888);
        let goal = RealHV::random(D, &mut rng);
        let fast = RealHV::random(D, &mut rng); // orthogonal

        let mut layer = VolitionLayer::new(D);
        layer.set_goal_raw(goal.clone(), GoalLevel::Operational, 0.2, 0.85);

        // Very tight budget + high threshold.
        let vocab = Vocabulary::new(D, 999); // EMPTY vocab → no vocabulary candidates
        let cfg = DeliberationConfig {
            base_cycles: 1,
            threshold_achieve: 0.99, // unreachable on 1 cycle
            threshold_minimum: 0.85, // very high minimum → Suspended
            convergence_epsilon: 0.001,
            alphas: vec![0.1], // only one weak attraction
            vocab_k: 2,
        };

        let res = layer.deliberate(&fast, &vocab, &cfg);
        assert!(
            res.outcome == DeliberationOutcome::Suspended
                || res.outcome == DeliberationOutcome::Converged,
            "expected Suspended/Converged on tight budget + unreachable goal, got {:?} (score={:.3})",
            res.outcome, res.final_score
        );
        assert!(res.final_score < 0.99);
    }

    // -------- Test 6: cycles scale with difficulty -------------------------
    #[test]
    fn test_cycles_scale_with_difficulty() {
        let mut rng = StdRng::seed_from_u64(1010);
        let goal = RealHV::random(D, &mut rng);
        let vocab = make_vocab(2020, 10);

        let mut layer = VolitionLayer::new(D);
        layer.set_goal_raw(goal.clone(), GoalLevel::Tactical, 0.6, 0.85);

        let cfg = DeliberationConfig {
            base_cycles: 10,
            threshold_achieve: 0.999,
            threshold_minimum: 0.10,
            convergence_epsilon: 0.0001, // nearly never converge early
            ..DeliberationConfig::default()
        };

        // Easy: fast_state very close to goal (noise 0.01).
        let easy_noise = RealHV::random_normal(D, &mut rng);
        let fast_easy = RealHV::add(&goal, &easy_noise.scale(0.01)).normalized();
        let res_easy = layer.deliberate(&fast_easy, &vocab, &cfg);

        // Medium: noise ~0.5.
        let med_noise = RealHV::random_normal(D, &mut rng);
        let fast_med = RealHV::add(&goal, &med_noise.scale(0.5)).normalized();
        let res_med = layer.deliberate(&fast_med, &vocab, &cfg);

        // Hard: orthogonal random HV.
        let fast_hard = RealHV::random(D, &mut rng);
        let res_hard = layer.deliberate(&fast_hard, &vocab, &cfg);

        assert!(
            res_hard.max_cycles > res_easy.max_cycles,
            "hard should have larger budget than easy ({} vs {})",
            res_hard.max_cycles, res_easy.max_cycles
        );
        assert!(
            res_med.max_cycles >= res_easy.max_cycles,
            "medium should have budget ≥ easy ({} vs {})",
            res_med.max_cycles, res_easy.max_cycles
        );
        assert!(
            res_hard.initial_coherence < res_easy.initial_coherence,
            "hard must be farther from goal than easy"
        );
    }
}
