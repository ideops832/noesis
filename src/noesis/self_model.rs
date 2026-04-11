//! Self-model — the system's memory about its own performance.
//!
//! Tracks rolling averages of perceived difficulty and cycles, the recent
//! suspension rate, and two small caches of "strong" (easy) and "weak"
//! (hard) domain hypervectors. From these caches it predicts the expected
//! difficulty of a new domain.
//!
//! Updates use EMA with α=0.1 so the self-model reacts smoothly:
//! ```text
//! avg = 0.9 × avg + 0.1 × new
//! ```
//!
//! Prediction formula (both caches populated):
//! ```text
//! sim_strong = max cosine(domain, d) for d in strong_domains
//! sim_weak   = max cosine(domain, d) for d in weak_domains
//! predict    = clamp(0.5 + 0.4 × (sim_weak − sim_strong), 0.0, 1.0)
//! ```

use std::collections::VecDeque;

use crate::hdc::real::RealHV;
use crate::noesis::deliberation::DeliberationOutcome;
use crate::noesis::meta_field::MetaObservation;

const EMA_ALPHA: f32 = 0.1;
const MAX_DOMAIN_CACHE: usize = 5;
const OUTCOME_WINDOW: usize = 20;
const STRONG_THRESHOLD: f32 = 0.3;
const WEAK_THRESHOLD: f32 = 0.7;

#[derive(Debug, Clone)]
pub struct SelfModel {
    pub avg_difficulty: f32,
    pub avg_cycles: f32,
    pub suspension_rate: f32,
    pub strong_domains: Vec<RealHV>,
    pub weak_domains: Vec<RealHV>,
    pub total_observations: usize,
    recent_outcomes: VecDeque<bool>,
}

impl SelfModel {
    pub fn new(initial_base_cycles: usize) -> Self {
        SelfModel {
            avg_difficulty: 0.5,
            avg_cycles: initial_base_cycles as f32,
            suspension_rate: 0.0,
            strong_domains: Vec::new(),
            weak_domains: Vec::new(),
            total_observations: 0,
            recent_outcomes: VecDeque::with_capacity(OUTCOME_WINDOW),
        }
    }

    /// Incorporate a new observation into the running estimates.
    pub fn update(
        &mut self,
        observation: &MetaObservation,
        outcome: DeliberationOutcome,
        domain_hv: &RealHV,
    ) {
        // EMA averages.
        self.avg_difficulty = (1.0 - EMA_ALPHA) * self.avg_difficulty
            + EMA_ALPHA * observation.perceived_difficulty;
        // Re-derive current cycles from cycle_trend × base_cycles is not
        // reliable here (we don't know base_cycles), so update avg_cycles
        // from cycle_trend directly: cycle_trend is already mean/base so
        // we fold (trend * avg_cycles_init) to keep units consistent.
        // Approximation: treat cycle_trend as a unit and blend.
        let cycles_estimate = observation.cycle_trend * self.avg_cycles.max(1.0);
        self.avg_cycles = (1.0 - EMA_ALPHA) * self.avg_cycles + EMA_ALPHA * cycles_estimate;

        // Suspension rate over a rolling window of OUTCOME_WINDOW.
        let is_suspended = matches!(outcome, DeliberationOutcome::Suspended);
        if self.recent_outcomes.len() >= OUTCOME_WINDOW {
            self.recent_outcomes.pop_front();
        }
        self.recent_outcomes.push_back(is_suspended);
        let n = self.recent_outcomes.len() as f32;
        let sus = self.recent_outcomes.iter().filter(|b| **b).count() as f32;
        self.suspension_rate = if n > 0.0 { sus / n } else { 0.0 };

        // Strong / weak domain caches (FIFO cap).
        if observation.perceived_difficulty < STRONG_THRESHOLD {
            if self.strong_domains.len() >= MAX_DOMAIN_CACHE {
                self.strong_domains.remove(0);
            }
            self.strong_domains.push(domain_hv.clone());
        } else if observation.perceived_difficulty > WEAK_THRESHOLD {
            if self.weak_domains.len() >= MAX_DOMAIN_CACHE {
                self.weak_domains.remove(0);
            }
            self.weak_domains.push(domain_hv.clone());
        }

        self.total_observations += 1;
    }

    /// Predict the difficulty expected for a new domain.
    pub fn predict_difficulty(&self, domain_hv: &RealHV) -> f32 {
        let strong_empty = self.strong_domains.is_empty();
        let weak_empty = self.weak_domains.is_empty();

        if strong_empty && weak_empty {
            // Cold start: fall back to the historical average or 0.5 early on.
            return if self.total_observations < 5 { 0.5 } else { self.avg_difficulty };
        }

        let sim_strong = self
            .strong_domains
            .iter()
            .map(|d| RealHV::cosine_similarity(domain_hv, d).max(0.0))
            .fold(0.0f32, f32::max);
        let sim_weak = self
            .weak_domains
            .iter()
            .map(|d| RealHV::cosine_similarity(domain_hv, d).max(0.0))
            .fold(0.0f32, f32::max);

        if !strong_empty && weak_empty {
            return if sim_strong > 0.5 { 0.3 } else { 0.5 };
        }
        if strong_empty && !weak_empty {
            return if sim_weak > 0.5 { 0.7 } else { 0.5 };
        }

        (0.5 + 0.4 * (sim_weak - sim_strong)).clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    const D: usize = 1024;

    fn mk_obs(diff: f32) -> MetaObservation {
        MetaObservation {
            internal_coherence: 1.0 - diff,
            cycle_trend: 1.0,
            field_stability: 1.0 - diff,
            perceived_difficulty: diff,
        }
    }

    // Test 5 -----------------------------------------------------------------
    #[test]
    fn test_self_model_update() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut sm = SelfModel::new(5);

        // 10 easy updates with a single domain HV
        let easy_domain = RealHV::random(D, &mut rng);
        for _ in 0..10 {
            sm.update(&mk_obs(0.1), DeliberationOutcome::Achieved, &easy_domain);
        }
        assert!(!sm.strong_domains.is_empty(), "easy updates should populate strong");
        // EMA from 0.5 with 10 updates at 0.1:
        //   0.5 → 0.46 → 0.424 → … converges towards 0.1
        assert!(
            sm.avg_difficulty < 0.35,
            "avg_difficulty should move towards 0.1 after 10 easy updates, got {}",
            sm.avg_difficulty
        );

        // 10 hard updates with another domain HV
        let hard_domain = RealHV::random(D, &mut rng);
        for _ in 0..10 {
            sm.update(&mk_obs(0.9), DeliberationOutcome::Suspended, &hard_domain);
        }
        assert!(!sm.weak_domains.is_empty(), "hard updates should populate weak");
        // Suspension rate computed over the last 20 (=full window): 10 easy
        // (not suspended) + 10 hard (suspended) → 0.5
        assert!((sm.suspension_rate - 0.5).abs() < 1e-4);
        // EMA should move up again
        assert!(sm.avg_difficulty > 0.35);

        // Domain caches respect the FIFO cap.
        for _ in 0..10 {
            sm.update(&mk_obs(0.1), DeliberationOutcome::Achieved, &easy_domain);
        }
        assert!(sm.strong_domains.len() <= MAX_DOMAIN_CACHE);
        for _ in 0..10 {
            sm.update(&mk_obs(0.9), DeliberationOutcome::Suspended, &hard_domain);
        }
        assert!(sm.weak_domains.len() <= MAX_DOMAIN_CACHE);
    }

    // Test 6 -----------------------------------------------------------------
    #[test]
    fn test_self_model_prediction() {
        let mut rng = StdRng::seed_from_u64(777);
        let mut sm = SelfModel::new(5);

        let domain_a = RealHV::random(D, &mut rng);
        let domain_b = RealHV::random(D, &mut rng);

        // Populate strong with domain_a family and weak with domain_b family.
        for _ in 0..3 {
            let noise = RealHV::random_normal(D, &mut rng);
            let near_a = RealHV::add(&domain_a, &noise.scale(0.05)).normalized();
            sm.update(&mk_obs(0.1), DeliberationOutcome::Achieved, &near_a);
        }
        for _ in 0..3 {
            let noise = RealHV::random_normal(D, &mut rng);
            let near_b = RealHV::add(&domain_b, &noise.scale(0.05)).normalized();
            sm.update(&mk_obs(0.9), DeliberationOutcome::Suspended, &near_b);
        }

        // Query with fresh samples from each domain.
        let new_near_a =
            RealHV::add(&domain_a, &RealHV::random_normal(D, &mut rng).scale(0.1)).normalized();
        let new_near_b =
            RealHV::add(&domain_b, &RealHV::random_normal(D, &mut rng).scale(0.1)).normalized();

        let pred_a = sm.predict_difficulty(&new_near_a);
        let pred_b = sm.predict_difficulty(&new_near_b);

        assert!(
            pred_a < pred_b,
            "expected predict(A) < predict(B), got {:.3} vs {:.3}", pred_a, pred_b
        );
        assert!(pred_a < 0.5, "predict(A) should be below 0.5, got {}", pred_a);
        assert!(pred_b > 0.5, "predict(B) should be above 0.5, got {}", pred_b);
    }
}
