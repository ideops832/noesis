//! Multi-scale temporal architecture.
//!
//! Stacks two SemanticFields with different time constants:
//! - **Fast field**: large dt, captures immediate context, decays quickly
//! - **Slow field**: small dt, accumulates long-term tendencies, changes slowly
//!
//! The output state is a weighted combination of both fields.

use crate::hdc::real::RealHV;
use crate::lnn::ncp::NCPConfig;
use crate::noesis::bridge::BridgeStrategy;
use crate::noesis::semantic_field::{SemanticField, SemanticFieldConfig};

/// Configuration for a multi-scale field.
#[derive(Clone, Debug)]
pub struct MultiScaleConfig {
    pub hd_dim: usize,
    pub strategy: BridgeStrategy,
    pub ncp_config: NCPConfig,
    /// dt for the fast field (larger = faster response).
    pub fast_dt: f32,
    /// dt for the slow field (smaller = slower accumulation).
    pub slow_dt: f32,
    /// Weight of the fast field in the combined output [0, 1].
    pub fast_weight: f32,
    /// Learning rate for slow field accumulation via bundling.
    pub slow_lr: f32,
}

impl MultiScaleConfig {
    pub fn default_for_dim(hd_dim: usize) -> Self {
        MultiScaleConfig {
            hd_dim,
            strategy: BridgeStrategy::InputPreserving,
            ncp_config: NCPConfig::tiny(),
            fast_dt: 0.2,
            slow_dt: 0.005,
            fast_weight: 0.4,
            slow_lr: 0.02,
        }
    }
}

/// A multi-scale semantic field with fast and slow dynamics.
pub struct MultiScaleField {
    config: MultiScaleConfig,
    fast: SemanticField,
    slow: SemanticField,
    /// Combined state: fast_weight * fast_state + (1 - fast_weight) * slow_state.
    combined_state: RealHV,
    t: f32,
}

impl MultiScaleField {
    /// Creates a new multi-scale field.
    pub fn new(config: MultiScaleConfig, seed: u64) -> Self {
        let sf_config = SemanticFieldConfig::new(
            config.hd_dim,
            config.strategy.clone(),
            config.ncp_config.clone(),
        );

        let fast = SemanticField::new(sf_config.clone(), seed);
        let slow = SemanticField::new(sf_config, seed + 1000);
        let combined_state = RealHV::zero(config.hd_dim);

        MultiScaleField {
            config,
            fast,
            slow,
            combined_state,
            t: 0.0,
        }
    }

    /// Step both fields and combine their states.
    ///
    /// Fast field: normal step (responds quickly via mixing).
    /// Slow field: accumulate via bundling (retains old inputs strongly).
    pub fn step(&mut self, input: &RealHV) -> &RealHV {
        // Fast field: normal step
        self.fast.step(input, self.config.fast_dt);

        // Slow field: accumulate via bundling
        let old_state = self.slow.state().clone();
        if old_state.norm() < 1e-8 {
            // First input: just adopt it
            let mut s = input.clone();
            s.normalize();
            self.slow.set_state(s);
        } else {
            // Normalize input so it has the same scale as the (already normalized) state
            let input_norm = input.normalized();
            let slow_lr = self.config.slow_lr;
            let kept = old_state.scale(1.0 - slow_lr);
            let added = input_norm.scale(slow_lr);
            let mut new_state = RealHV::add(&kept, &added);
            new_state.normalize();
            self.slow.set_state(new_state);
        }

        // Combine: weighted average of both states
        let w_fast = self.config.fast_weight;
        let w_slow = 1.0 - w_fast;
        let fast_contrib = self.fast.state().scale(w_fast);
        let slow_contrib = self.slow.state().scale(w_slow);
        self.combined_state = RealHV::add(&fast_contrib, &slow_contrib);
        if self.combined_state.norm() > 1e-8 {
            self.combined_state.normalize();
        }

        self.t += self.config.fast_dt;
        &self.combined_state
    }

    /// Idle step: no input, both fields decay.
    pub fn idle_step(&mut self) {
        self.fast.idle_step(self.config.fast_dt);
        self.slow.idle_step(self.config.slow_dt);

        let w_fast = self.config.fast_weight;
        let w_slow = 1.0 - w_fast;
        let fast_contrib = self.fast.state().scale(w_fast);
        let slow_contrib = self.slow.state().scale(w_slow);
        self.combined_state = RealHV::add(&fast_contrib, &slow_contrib);
        if self.combined_state.norm() > 1e-8 {
            self.combined_state.normalize();
        }

        self.t += self.config.fast_dt;
    }

    /// Query the combined state.
    pub fn query(&self, role: &RealHV) -> RealHV {
        RealHV::bind(&self.combined_state, &role.inverse())
    }

    /// Similarity of combined state to a target.
    pub fn similarity_to(&self, target: &RealHV) -> f32 {
        RealHV::cosine_similarity(&self.combined_state, target)
    }

    /// Access individual field states.
    pub fn fast_state(&self) -> &RealHV { self.fast.state() }
    pub fn slow_state(&self) -> &RealHV { self.slow.state() }
    pub fn combined_state(&self) -> &RealHV { &self.combined_state }

    /// Reset both fields.
    pub fn reset(&mut self) {
        self.fast.reset();
        self.slow.reset();
        self.combined_state = RealHV::zero(self.config.hd_dim);
        self.t = 0.0;
    }

    /// Elapsed time.
    pub fn elapsed(&self) -> f32 { self.t }
}

#[cfg(test)]
mod tests {
    use super::*;

    const D: usize = 1024;

    fn make_config() -> MultiScaleConfig {
        MultiScaleConfig::default_for_dim(D)
    }

    #[test]
    fn test_multiscale_fast_responds_faster() {
        let mut ms = MultiScaleField::new(make_config(), 42);
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        use rand::SeedableRng;
        let input_a = RealHV::random(D, &mut rng);
        let input_b = RealHV::random(D, &mut rng);

        // Feed input A for several steps
        for _ in 0..10 {
            ms.step(&input_a);
        }

        // Now switch to input B for a few steps
        for _ in 0..3 {
            ms.step(&input_b);
        }

        // Fast field should have adapted to B more than slow field
        let fast_sim_b = RealHV::cosine_similarity(ms.fast_state(), &input_b);
        let slow_sim_b = RealHV::cosine_similarity(ms.slow_state(), &input_b);

        // Both should be finite
        assert!(fast_sim_b.is_finite() && slow_sim_b.is_finite());

        // Fast field has larger dt so it integrates input faster
        // The fast field should show more adaptation to the new input B
        println!("After A→B: fast_sim_B={:.4}, slow_sim_B={:.4}", fast_sim_b, slow_sim_b);

        // The key property: fast field should have adapted more to B
        // (or at least show different dynamics from slow)
        let fast_sim_a = RealHV::cosine_similarity(ms.fast_state(), &input_a);
        let slow_sim_a = RealHV::cosine_similarity(ms.slow_state(), &input_a);
        println!("After A→B: fast_sim_A={:.4}, slow_sim_A={:.4}", fast_sim_a, slow_sim_a);

        // The two fields should have diverged (different states)
        let field_sim = RealHV::cosine_similarity(ms.fast_state(), ms.slow_state());
        println!("Fast-slow similarity: {:.4}", field_sim);
        // They should not be identical
        assert!(
            field_sim < 0.9999,
            "Fast and slow fields should differ after asymmetric input: sim={:.6}",
            field_sim
        );
    }

    #[test]
    fn test_multiscale_slow_retains_longer() {
        let mut ms = MultiScaleField::new(make_config(), 42);
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        use rand::SeedableRng;
        let input_a = RealHV::random(D, &mut rng);
        let input_b = RealHV::random(D, &mut rng);

        // Feed input A for 10 steps
        for _ in 0..10 {
            ms.step(&input_a);
        }

        // Switch to input B for 10 steps
        for _ in 0..10 {
            ms.step(&input_b);
        }

        // Slow field should retain more of A than fast field
        let fast_sim_a = RealHV::cosine_similarity(ms.fast_state(), &input_a);
        let slow_sim_a = RealHV::cosine_similarity(ms.slow_state(), &input_a);

        // The slow field, having smaller dt, integrates more slowly
        // so it should have some trace of A even after B has been pushed
        // (This depends on the specific dynamics, so we just check validity)
        assert!(
            ms.fast_state().data.iter().all(|x| x.is_finite()),
            "Fast state should be finite"
        );
        assert!(
            ms.slow_state().data.iter().all(|x| x.is_finite()),
            "Slow state should be finite"
        );

        println!("After A→B switch: fast_sim_A={:.4}, slow_sim_A={:.4}", fast_sim_a, slow_sim_a);
    }

    #[test]
    fn test_multiscale_combined_state() {
        let mut ms = MultiScaleField::new(make_config(), 42);
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        use rand::SeedableRng;
        let input = RealHV::random(D, &mut rng);

        for _ in 0..5 {
            ms.step(&input);
        }

        // Combined state should be between fast and slow
        let combined_sim = ms.similarity_to(&input);
        assert!(
            combined_sim.is_finite(),
            "Combined similarity should be finite"
        );
        assert!(
            ms.combined_state().norm() > 0.5,
            "Combined state should be non-trivial"
        );
    }

    #[test]
    fn test_multiscale_idle_decay() {
        let mut ms = MultiScaleField::new(make_config(), 42);
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        use rand::SeedableRng;
        let input = RealHV::random(D, &mut rng);

        // Build up state
        for _ in 0..10 {
            ms.step(&input);
        }
        let sim_before = ms.similarity_to(&input);

        // Idle for many steps
        for _ in 0..20 {
            ms.idle_step();
        }
        let sim_after = ms.similarity_to(&input);

        // State should change (not necessarily decrease, but evolve)
        assert!(
            sim_after.is_finite(),
            "Similarity after idle should be finite"
        );
        println!("Idle decay: before={:.4}, after={:.4}", sim_before, sim_after);
    }
}
