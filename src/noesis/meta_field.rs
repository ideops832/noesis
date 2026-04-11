//! Meta-field — internal-state observer and parameter regulator.
//!
//! The meta-field does not process external inputs. It observes NOESIS's
//! own internal state (`fast_state`, `slow_state`, cycle history) and
//! emits a [`MetaRegulation`] that the deliberation loop uses to adjust
//! its α and γ parameters dynamically.
//!
//! Three metrics drive perceived difficulty:
//!
//! 1. **internal_coherence** = sim(fast_state, slow_state) — how aligned
//!    the current moment is with the long-term goal.
//! 2. **cycle_trend** = mean(recent cycles) / base_cycles — whether the
//!    system has been systematically burning more budget than expected.
//! 3. **field_stability** = sim(fast_t, fast_{t-1}) — does the field
//!    converge or oscillate?
//!
//! ```text
//! difficulty = (1 − coherence) × (1 + cycle_trend) × (1 − stability)
//! ```
//! clamped to [0, 1].
//!
//! Regulation maps difficulty back to actionable knobs:
//!
//! ```text
//! alpha_max  = 1 − 0.5 × difficulty
//! alphas     = [alpha_max × 0.30, alpha_max × 0.60, alpha_max × 1.0]
//! gamma_adj  = 0.3 × difficulty
//! ```
//!
//! A qualitative [`MetaSignal`] is emitted: `Overloaded` when difficulty
//! > 0.8 **and** cycle_trend > 2.0, `Uncertain` when difficulty > 0.6,
//! `Normal` otherwise.

use std::collections::VecDeque;

use crate::hdc::real::RealHV;

// ---------------------------------------------------------------------------
// Observation / regulation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MetaObservation {
    pub internal_coherence: f32,
    pub cycle_trend: f32,
    pub field_stability: f32,
    pub perceived_difficulty: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetaSignal {
    Normal,
    Uncertain,
    Overloaded,
}

#[derive(Debug, Clone)]
pub struct MetaRegulation {
    pub alphas: Vec<f32>,
    pub gamma_adjustment: f32,
    pub meta_signal: MetaSignal,
}

impl MetaRegulation {
    /// Apply the `gamma_adjustment` to a given `current_gamma`, clamping the
    /// result to the operating range [0.2, 0.95].
    pub fn effective_gamma(&self, current_gamma: f32) -> f32 {
        (current_gamma + self.gamma_adjustment).clamp(0.2, 0.95)
    }

    /// Convenience: the largest α in the regulated set.
    pub fn alpha_max(&self) -> f32 {
        self.alphas.last().copied().unwrap_or(0.0)
    }
}

// ---------------------------------------------------------------------------
// Meta-field
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MetaField {
    history: VecDeque<MetaObservation>,
    history_size: usize,
    prev_fast_state: Option<RealHV>,
    cycle_history: VecDeque<usize>,
    pub base_cycles: usize,
}

impl MetaField {
    pub fn new(base_cycles: usize, history_size: usize) -> Self {
        MetaField {
            history: VecDeque::with_capacity(history_size),
            history_size,
            prev_fast_state: None,
            cycle_history: VecDeque::with_capacity(history_size),
            base_cycles: base_cycles.max(1),
        }
    }

    /// Default with base_cycles=8 (matches the benchmark config) and
    /// history_size=10.
    pub fn default_config() -> Self {
        Self::new(8, 10)
    }

    /// Number of observations currently retained.
    pub fn history_len(&self) -> usize { self.history.len() }

    /// Access the latest observation, if any.
    pub fn last(&self) -> Option<&MetaObservation> { self.history.back() }

    /// Seed cycle history with a known sequence (used by experiments to
    /// simulate a prior session of difficult work).
    pub fn preload_cycles(&mut self, cycles: impl IntoIterator<Item = usize>) {
        for c in cycles {
            if self.cycle_history.len() >= self.history_size {
                self.cycle_history.pop_front();
            }
            self.cycle_history.push_back(c);
        }
    }

    /// Observe the current internal state and produce a `MetaObservation`.
    ///
    /// `cycles_used` is the number of deliberation cycles the last
    /// operation spent — it updates the cycle trend.
    pub fn observe(
        &mut self,
        fast_state: &RealHV,
        slow_state: &RealHV,
        cycles_used: usize,
    ) -> MetaObservation {
        // 1. Internal coherence
        let internal_coherence =
            RealHV::cosine_similarity(fast_state, slow_state).clamp(0.0, 1.0);

        // 2. Field stability (1.0 on first call — no prior frame).
        let field_stability = match &self.prev_fast_state {
            Some(prev) => RealHV::cosine_similarity(fast_state, prev).clamp(0.0, 1.0),
            None => 1.0,
        };

        // 3. Cycle trend — update history then compute mean / base_cycles.
        if self.cycle_history.len() >= self.history_size {
            self.cycle_history.pop_front();
        }
        self.cycle_history.push_back(cycles_used);
        let cycle_trend = if self.cycle_history.is_empty() {
            1.0
        } else {
            let mean = self.cycle_history.iter().sum::<usize>() as f32
                / self.cycle_history.len() as f32;
            mean / self.base_cycles as f32
        };

        // 4. Perceived difficulty
        let raw_difficulty =
            (1.0 - internal_coherence) * (1.0 + cycle_trend) * (1.0 - field_stability);
        let perceived_difficulty = raw_difficulty.clamp(0.0, 1.0);

        let obs = MetaObservation {
            internal_coherence,
            cycle_trend,
            field_stability,
            perceived_difficulty,
        };

        // 5. Advance history & prev_fast_state.
        if self.history.len() >= self.history_size {
            self.history.pop_front();
        }
        self.history.push_back(obs.clone());
        self.prev_fast_state = Some(fast_state.clone());

        obs
    }

    /// Derive α and γ adjustments from an observation.
    pub fn regulate(
        &self,
        observation: &MetaObservation,
        _current_gamma: f32,
    ) -> MetaRegulation {
        let d = observation.perceived_difficulty.clamp(0.0, 1.0);
        let alpha_max = 1.0 - 0.5 * d;
        let alphas = vec![alpha_max * 0.30, alpha_max * 0.60, alpha_max * 1.0];
        let gamma_adjustment = 0.3 * d;

        let meta_signal = if d > 0.8 && observation.cycle_trend > 2.0 {
            MetaSignal::Overloaded
        } else if d > 0.6 {
            MetaSignal::Uncertain
        } else {
            MetaSignal::Normal
        };

        MetaRegulation { alphas, gamma_adjustment, meta_signal }
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

    fn mkfield() -> MetaField { MetaField::new(8, 10) }

    // Test 1 -----------------------------------------------------------------
    #[test]
    fn test_meta_observation() {
        let mut rng = StdRng::seed_from_u64(10);
        let fast = RealHV::random(D, &mut rng);
        // slow = noisy copy of fast → high coherence
        let noise = RealHV::random_normal(D, &mut rng);
        let slow = RealHV::add(&fast, &noise.scale(0.2)).normalized();

        let mut mf = mkfield();
        let obs = mf.observe(&fast, &slow, 2);

        // internal_coherence matches direct cosine computation
        let expected = RealHV::cosine_similarity(&fast, &slow).clamp(0.0, 1.0);
        assert!(
            (obs.internal_coherence - expected).abs() < 1e-6,
            "{} vs {}", obs.internal_coherence, expected
        );
        // First call: field stability defaults to 1.0
        assert!((obs.field_stability - 1.0).abs() < 1e-6);
        // cycle_trend = 2 / 8 = 0.25
        assert!(
            (obs.cycle_trend - 0.25).abs() < 1e-6,
            "cycle_trend={}", obs.cycle_trend
        );
        // perceived_difficulty = (1 - coh) * 1.25 * 0 = 0 on first step
        assert!(obs.perceived_difficulty < 1e-6);

        // Second call: stability now compares to first frame.
        let fast2 = RealHV::random(D, &mut rng); // random → near-orthogonal
        let obs2 = mf.observe(&fast2, &slow, 8);
        assert!(obs2.field_stability < 0.3,
            "stability should be low between random frames, got {}",
            obs2.field_stability);
        // cycle_trend = mean(2, 8) / 8 = 0.625
        assert!(
            (obs2.cycle_trend - 0.625).abs() < 1e-4,
            "cycle_trend={}", obs2.cycle_trend
        );
    }

    // Test 2 -----------------------------------------------------------------
    #[test]
    fn test_alpha_regulation() {
        let mf = mkfield();

        // Difficulty 0 → alpha_max = 1.0
        let obs_easy = MetaObservation {
            internal_coherence: 1.0,
            cycle_trend: 0.0,
            field_stability: 1.0,
            perceived_difficulty: 0.0,
        };
        let reg_easy = mf.regulate(&obs_easy, 0.4);
        assert!((reg_easy.alpha_max() - 1.0).abs() < 1e-6);
        assert!((reg_easy.alphas[0] - 0.30).abs() < 1e-6);
        assert!((reg_easy.alphas[1] - 0.60).abs() < 1e-6);
        assert!((reg_easy.alphas[2] - 1.00).abs() < 1e-6);

        // Difficulty 1.0 → alpha_max = 0.5
        let obs_hard = MetaObservation {
            internal_coherence: 0.0,
            cycle_trend: 2.0,
            field_stability: 0.0,
            perceived_difficulty: 1.0,
        };
        let reg_hard = mf.regulate(&obs_hard, 0.4);
        assert!((reg_hard.alpha_max() - 0.5).abs() < 1e-6);
        assert!((reg_hard.alphas[0] - 0.15).abs() < 1e-6);
        assert!((reg_hard.alphas[1] - 0.30).abs() < 1e-6);
        assert!((reg_hard.alphas[2] - 0.50).abs() < 1e-6);

        // Ordering is preserved and alpha_max decreases with difficulty.
        assert!(reg_easy.alpha_max() > reg_hard.alpha_max());
    }

    // Test 3 -----------------------------------------------------------------
    #[test]
    fn test_gamma_regulation() {
        let mf = mkfield();

        let obs_low = MetaObservation {
            internal_coherence: 0.9,
            cycle_trend: 0.2,
            field_stability: 0.9,
            perceived_difficulty: 0.1,
        };
        let reg_low = mf.regulate(&obs_low, 0.4);
        assert!((reg_low.gamma_adjustment - 0.03).abs() < 1e-6);
        // γ applied + clamp
        assert!((reg_low.effective_gamma(0.4) - 0.43).abs() < 1e-6);

        let obs_high = MetaObservation {
            internal_coherence: 0.1,
            cycle_trend: 2.5,
            field_stability: 0.1,
            perceived_difficulty: 1.0,
        };
        let reg_high = mf.regulate(&obs_high, 0.8);
        assert!((reg_high.gamma_adjustment - 0.3).abs() < 1e-6);
        // 0.8 + 0.3 = 1.1 → clamped to 0.95
        assert!((reg_high.effective_gamma(0.8) - 0.95).abs() < 1e-6);

        // Clamp low side.
        assert!((reg_low.effective_gamma(0.0) - 0.2).abs() < 1e-6);
    }

    // Test 4 -----------------------------------------------------------------
    #[test]
    fn test_meta_signal() {
        let mf = mkfield();

        // Overloaded: difficulty high + trend high
        let overloaded = MetaObservation {
            internal_coherence: 0.05,
            cycle_trend: 2.5,
            field_stability: 0.05,
            perceived_difficulty: 0.9,
        };
        assert_eq!(mf.regulate(&overloaded, 0.4).meta_signal, MetaSignal::Overloaded);

        // Uncertain: difficulty > 0.6 but trend not extreme
        let uncertain = MetaObservation {
            internal_coherence: 0.3,
            cycle_trend: 0.8,
            field_stability: 0.3,
            perceived_difficulty: 0.65,
        };
        assert_eq!(mf.regulate(&uncertain, 0.4).meta_signal, MetaSignal::Uncertain);

        // Normal
        let normal = MetaObservation {
            internal_coherence: 0.7,
            cycle_trend: 0.4,
            field_stability: 0.7,
            perceived_difficulty: 0.3,
        };
        assert_eq!(mf.regulate(&normal, 0.4).meta_signal, MetaSignal::Normal);
    }
}
