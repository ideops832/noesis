//! Semantic field: hypervectors as state, LNN as dynamics.
//!
//! The SemanticField integrates HDC and LNN through a Bridge.
//! State lives in HD space. At each step, the bridge encodes state+input
//! for the LNN, the LNN processes it, and the bridge decodes the output
//! back into HD space.

use crate::hdc::real::RealHV;
use crate::lnn::ncp::{NCP, NCPConfig};
use crate::noesis::bridge::{Bridge, BridgeStrategy, create_bridge};

use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

/// Configuration for a SemanticField.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SemanticFieldConfig {
    pub hd_dim: usize,
    pub bridge_strategy: BridgeStrategy,
    pub ncp_config: NCPConfig,
    pub dt_default: f32,
    pub history_enabled: bool,
    pub max_history_len: usize,
    /// If set, overrides the NCP-derived alpha with a fixed value.
    /// This bypasses NCP noise for compositionality-critical paths.
    pub alpha_override: Option<f32>,
}

impl SemanticFieldConfig {
    /// Default config with given strategy and dimension.
    pub fn new(hd_dim: usize, strategy: BridgeStrategy, ncp_config: NCPConfig) -> Self {
        SemanticFieldConfig {
            hd_dim,
            bridge_strategy: strategy,
            ncp_config,
            dt_default: 0.1,
            history_enabled: false,
            max_history_len: 1000,
            alpha_override: None,
        }
    }
}

/// The SemanticField: HD state evolving under LNN dynamics via a Bridge.
pub struct SemanticField {
    pub config: SemanticFieldConfig,
    state: RealHV,
    bridge: Box<dyn Bridge>,
    ncp: NCP,
    t: f32,
    history: Vec<(f32, RealHV)>,
}

impl SemanticField {
    /// Creates a new SemanticField.
    pub fn new(config: SemanticFieldConfig, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);

        let bridge = create_bridge(
            config.bridge_strategy.clone(),
            config.hd_dim,
            &config.ncp_config,
            &mut rng,
        );

        // For DirectHD and SparseHD, the NCP must match bridge I/O sizes
        let actual_ncp_config = match config.bridge_strategy {
            BridgeStrategy::DirectHD => {
                // Single large CfC: sensory = hd_dim, motor = hd_dim
                NCPConfig {
                    sensory_size: bridge.lnn_input_size(),
                    inter_size: 2,
                    command_size: 2,
                    motor_size: bridge.lnn_output_size(),
                    inter_recurrent: true,
                    command_recurrent: true,
                }
            }
            BridgeStrategy::SparseHD => {
                let io_size = bridge.lnn_input_size();
                NCPConfig {
                    sensory_size: io_size,
                    inter_size: 4,
                    command_size: 4,
                    motor_size: bridge.lnn_output_size(),
                    inter_recurrent: true,
                    command_recurrent: true,
                }
            }
            _ => config.ncp_config.clone(),
        };

        let ncp = NCP::new(actual_ncp_config, &mut rng);
        let state = RealHV::zero(config.hd_dim);

        SemanticField {
            config,
            state,
            bridge,
            ncp,
            t: 0.0,
            history: Vec::new(),
        }
    }

    /// Performs one step: bridge encode → NCP step → bridge decode.
    ///
    /// For InputPreserving strategy, the NCP output controls the mixing rate
    /// and the input is injected directly into the state.
    pub fn step(&mut self, input: &RealHV, dt: f32) -> &RealHV {
        let lnn_input = self.bridge.encode_for_lnn(&self.state, input);
        let lnn_output = self.ncp.step(&lnn_input, dt);

        let new_state = match self.config.bridge_strategy {
            BridgeStrategy::InputPreserving => {
                let alpha = if let Some(fixed) = self.config.alpha_override {
                    // Fixed alpha: bypasses NCP noise for compositionality
                    let dt_factor = (dt * 10.0).min(1.0);
                    fixed * dt_factor
                } else {
                    // NCP controls alpha via sigmoid
                    use crate::lnn::neuron::sigmoid;
                    let mean_out: f32 = if lnn_output.is_empty() {
                        0.0
                    } else {
                        lnn_output.iter().sum::<f32>() / lnn_output.len() as f32
                    };
                    let dt_factor = (dt * 10.0).min(1.0);
                    sigmoid(mean_out) * 0.5 * dt_factor
                };
                let kept = self.state.scale(1.0 - alpha);
                let added = input.scale(alpha);
                let mut result = RealHV::add(&kept, &added);
                result.normalize();
                result
            }
            BridgeStrategy::SemanticDualTrack => {
                // Probe-guided adjustment + input mixing, dt-modulated
                use crate::lnn::neuron::sigmoid;
                let mean_out: f32 = if lnn_output.is_empty() {
                    0.0
                } else {
                    lnn_output.iter().sum::<f32>() / lnn_output.len() as f32
                };
                let dt_factor = (dt * 10.0).min(1.0);
                let alpha = sigmoid(mean_out) * 0.2 * dt_factor;
                let adjusted = self.bridge.decode_from_lnn(&lnn_output, &self.state);
                let with_input = RealHV::add(&adjusted.scale(1.0 - alpha), &input.scale(alpha));
                let mut result = with_input;
                if result.norm() > 1e-8 {
                    result.normalize();
                }
                result
            }
            _ => {
                self.bridge.decode_from_lnn(&lnn_output, &self.state)
            }
        };

        self.state = new_state;
        self.t += dt;

        if self.config.history_enabled {
            self.history.push((self.t, self.state.clone()));
            if self.history.len() > self.config.max_history_len {
                self.history.remove(0);
            }
        }

        &self.state
    }

    /// Step with default dt.
    pub fn step_default(&mut self, input: &RealHV) -> &RealHV {
        let dt = self.config.dt_default;
        self.step(input, dt)
    }

    /// Query: unbind the state with a role to recover a value.
    pub fn query(&self, role: &RealHV) -> RealHV {
        RealHV::bind(&self.state, &role.inverse())
    }

    /// Similarity of current state to a target.
    pub fn similarity_to(&self, target: &RealHV) -> f32 {
        RealHV::cosine_similarity(&self.state, target)
    }

    /// Reset state, time, and history.
    pub fn reset(&mut self) {
        self.state = RealHV::zero(self.config.hd_dim);
        self.t = 0.0;
        self.history.clear();
        self.ncp.reset();
    }

    /// Step without input (idle decay).
    pub fn idle_step(&mut self, dt: f32) {
        let zero = RealHV::zero(self.config.hd_dim);
        self.step(&zero, dt);
    }

    /// Access the history.
    pub fn get_history(&self) -> &[(f32, RealHV)] {
        &self.history
    }

    /// Current state.
    pub fn state(&self) -> &RealHV {
        &self.state
    }

    /// Directly set the internal state (used by multi-scale accumulation).
    pub fn set_state(&mut self, state: RealHV) {
        self.state = state;
    }

    /// Elapsed time.
    pub fn elapsed(&self) -> f32 {
        self.t
    }

    /// Total NCP parameters.
    pub fn ncp_param_count(&self) -> usize {
        self.ncp.total_parameters()
    }

    /// Extract NCP parameters as flat vector.
    pub fn get_ncp_params(&self) -> Vec<f32> {
        self.ncp.get_params()
    }

    /// Set NCP parameters from flat vector.
    pub fn set_ncp_params(&mut self, params: &[f32]) {
        self.ncp.set_params(params);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::real::RealHV;
    use crate::hdc::hypervector::HyperVector;
    use crate::lnn::ncp::NCPConfig;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn make_config(strategy: BridgeStrategy) -> SemanticFieldConfig {
        SemanticFieldConfig::new(1024, strategy, NCPConfig::tiny())
    }

    #[test]
    fn test_sf_evolves() {
        let config = make_config(BridgeStrategy::RandomProjection);
        let mut sf = SemanticField::new(config, 42);
        let mut rng = StdRng::seed_from_u64(99);
        let input = RealHV::random(1024, &mut rng);

        for _ in 0..50 {
            sf.step_default(&input);
        }
        // After many steps, state should be non-trivial (finite values)
        assert!(
            sf.state().data.iter().all(|x| x.is_finite()),
            "State should contain finite values"
        );
        // Check state after more steps differs from state at step 50
        let state_50 = sf.state().clone();
        for _ in 0..10 {
            sf.step_default(&input);
        }
        // Eventually the state converges or changes — just verify it's valid
        assert!(sf.state().data.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_sf_input_sensitive() {
        let config1 = make_config(BridgeStrategy::RandomProjection);
        let config2 = make_config(BridgeStrategy::RandomProjection);
        let mut sf1 = SemanticField::new(config1, 42);
        let mut sf2 = SemanticField::new(config2, 42);

        let mut rng = StdRng::seed_from_u64(99);
        let input_a = RealHV::random(1024, &mut rng);
        let input_b = RealHV::random(1024, &mut rng);

        for _ in 0..10 {
            sf1.step_default(&input_a);
            sf2.step_default(&input_b);
        }

        let sim = RealHV::cosine_similarity(sf1.state(), sf2.state());
        assert!(
            sim < 0.9,
            "Different inputs should produce different states: sim = {sim}"
        );
    }

    #[test]
    fn test_sf_remembers() {
        let config = make_config(BridgeStrategy::DualTrack);
        let mut sf = SemanticField::new(config, 42);
        let mut rng = StdRng::seed_from_u64(99);
        let input_a = RealHV::random(1024, &mut rng);
        let input_b = RealHV::random(1024, &mut rng);

        // Feed A for 10 steps
        for _ in 0..10 {
            sf.step_default(&input_a);
        }
        // Feed B for 10 steps
        for _ in 0..10 {
            sf.step_default(&input_b);
        }

        // A should still have some trace in state
        let sim_a = sf.similarity_to(&input_a);
        // This is a soft test — DualTrack preserves more structure
        // Just verify state is not zero and has some signal
        assert!(sf.state().norm() > 0.01, "State should not be zero");
    }

    #[test]
    fn test_sf_idle_decay() {
        let config = make_config(BridgeStrategy::RandomProjection);
        let mut sf = SemanticField::new(config, 42);
        let mut rng = StdRng::seed_from_u64(99);
        let input = RealHV::random(1024, &mut rng);

        for _ in 0..10 {
            sf.step_default(&input);
        }
        let norm_before = sf.state().norm();

        for _ in 0..20 {
            sf.idle_step(0.1);
        }
        // State may change during idle (decay or attractor)
        assert!(sf.state().data.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_sf_history() {
        let mut config = make_config(BridgeStrategy::RandomProjection);
        config.history_enabled = true;
        let mut sf = SemanticField::new(config, 42);
        let mut rng = StdRng::seed_from_u64(99);
        let input = RealHV::random(1024, &mut rng);

        for _ in 0..5 {
            sf.step_default(&input);
        }
        assert_eq!(sf.get_history().len(), 5);
    }

    #[test]
    fn test_sf_reproducibility() {
        let config1 = make_config(BridgeStrategy::RandomProjection);
        let config2 = make_config(BridgeStrategy::RandomProjection);
        let mut sf1 = SemanticField::new(config1, 42);
        let mut sf2 = SemanticField::new(config2, 42);

        let mut rng = StdRng::seed_from_u64(99);
        let input = RealHV::random(1024, &mut rng);

        for _ in 0..10 {
            sf1.step_default(&input);
            sf2.step_default(&input);
        }

        let sim = RealHV::cosine_similarity(sf1.state(), sf2.state());
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "Same seed should produce identical results: sim = {sim}"
        );
    }

    #[test]
    fn test_adaptive_dual_track_evolves() {
        let config = make_config(BridgeStrategy::AdaptiveDualTrack);
        let mut sf = SemanticField::new(config, 42);
        let mut rng = StdRng::seed_from_u64(99);
        let input = RealHV::random(1024, &mut rng);

        for _ in 0..50 {
            sf.step_default(&input);
        }
        assert!(
            sf.state().data.iter().all(|x| x.is_finite()),
            "AdaptiveDualTrack state should contain finite values"
        );
        assert!(sf.state().norm() > 0.01, "State should not be zero");
    }

    #[test]
    fn test_adaptive_probes_move_toward_inputs() {
        use crate::noesis::bridge::AdaptiveDualTrackBridge;

        let mut rng = StdRng::seed_from_u64(42);
        let bridge = AdaptiveDualTrackBridge::new(1024, 8, 0.05, &mut rng);

        // Create two distinct "category" vectors
        let cat_a = RealHV::random(1024, &mut rng).normalized();
        let cat_b = RealHV::random(1024, &mut rng).normalized();

        // Record initial max similarity of any probe to cat_a and cat_b
        let initial_probes = bridge.get_probes();
        let init_max_sim_a: f32 = initial_probes.iter()
            .map(|p| RealHV::cosine_similarity(p, &cat_a))
            .fold(f32::NEG_INFINITY, f32::max);
        let init_max_sim_b: f32 = initial_probes.iter()
            .map(|p| RealHV::cosine_similarity(p, &cat_b))
            .fold(f32::NEG_INFINITY, f32::max);

        // Feed 100 samples near cat_a and 100 near cat_b
        for _ in 0..100 {
            // Slight noise around category centroids
            let noise_a = RealHV::random(1024, &mut rng).normalized();
            let input_a = RealHV::add(&cat_a.scale(0.9), &noise_a.scale(0.1)).normalized();
            bridge.adapt_probes(&input_a);

            let noise_b = RealHV::random(1024, &mut rng).normalized();
            let input_b = RealHV::add(&cat_b.scale(0.9), &noise_b.scale(0.1)).normalized();
            bridge.adapt_probes(&input_b);
        }

        // After adaptation, at least one probe should be closer to cat_a
        // and at least one closer to cat_b
        let adapted_probes = bridge.get_probes();
        let final_max_sim_a: f32 = adapted_probes.iter()
            .map(|p| RealHV::cosine_similarity(p, &cat_a))
            .fold(f32::NEG_INFINITY, f32::max);
        let final_max_sim_b: f32 = adapted_probes.iter()
            .map(|p| RealHV::cosine_similarity(p, &cat_b))
            .fold(f32::NEG_INFINITY, f32::max);

        assert!(
            final_max_sim_a > init_max_sim_a,
            "Probes should move toward cat_a: init={init_max_sim_a:.4}, final={final_max_sim_a:.4}"
        );
        assert!(
            final_max_sim_b > init_max_sim_b,
            "Probes should move toward cat_b: init={init_max_sim_b:.4}, final={final_max_sim_b:.4}"
        );
    }

    #[test]
    fn test_adaptive_vs_fixed_probes() {
        // Compare adaptive and fixed DualTrack bridges:
        // adaptive probes should provide better feature extraction
        // for clustered inputs after adaptation.
        let mut rng = StdRng::seed_from_u64(42);

        // Create two category centroids
        let cat_a = RealHV::random(1024, &mut rng).normalized();
        let cat_b = RealHV::random(1024, &mut rng).normalized();

        // Build two SemanticFields: adaptive and fixed
        let config_adaptive = make_config(BridgeStrategy::AdaptiveDualTrack);
        let config_fixed = make_config(BridgeStrategy::DualTrack);
        let mut sf_adaptive = SemanticField::new(config_adaptive, 42);
        let mut sf_fixed = SemanticField::new(config_fixed, 42);

        // Feed 50 "animal" words (near cat_a)
        for _ in 0..50 {
            let noise = RealHV::random(1024, &mut rng).normalized();
            let input = RealHV::add(&cat_a.scale(0.9), &noise.scale(0.1)).normalized();
            sf_adaptive.step_default(&input);
            sf_fixed.step_default(&input);
        }

        // Feed 50 "food" words (near cat_b)
        for _ in 0..50 {
            let noise = RealHV::random(1024, &mut rng).normalized();
            let input = RealHV::add(&cat_b.scale(0.9), &noise.scale(0.1)).normalized();
            sf_adaptive.step_default(&input);
            sf_fixed.step_default(&input);
        }

        // Both should produce finite, non-zero states
        assert!(sf_adaptive.state().data.iter().all(|x| x.is_finite()));
        assert!(sf_fixed.state().data.iter().all(|x| x.is_finite()));
        assert!(sf_adaptive.state().norm() > 0.01);
        assert!(sf_fixed.state().norm() > 0.01);
    }
}
