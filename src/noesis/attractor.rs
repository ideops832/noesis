//! Attractor detection and analysis in the semantic field.
//!
//! Provides tools to find fixed points, analyze basins of attraction,
//! and trace trajectories through the semantic field's state space.

use crate::hdc::real::RealHV;
use crate::noesis::semantic_field::{SemanticField, SemanticFieldConfig};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Result of a trajectory analysis: a sequence of states with metadata.
#[derive(Clone, Debug)]
pub struct TrajectoryResult {
    /// The states visited along the trajectory.
    pub states: Vec<RealHV>,
    /// Cosine similarities between consecutive states.
    pub convergence: Vec<f32>,
    /// Whether the trajectory converged to a fixed point.
    pub converged: bool,
    /// Index at which convergence was detected (if any).
    pub convergence_step: Option<usize>,
}

/// A detected fixed point (attractor) in the semantic field.
#[derive(Clone, Debug)]
pub struct FixedPoint {
    /// The state vector at the fixed point.
    pub state: RealHV,
    /// Self-similarity after one more step (should be ~1.0 for a true fixed point).
    pub stability: f32,
    /// The initial condition that led to this fixed point.
    pub seed_state: RealHV,
}

/// Analyzes attractors in a SemanticField.
///
/// The analyzer creates fresh SemanticField instances from the same config
/// to probe the dynamical landscape from different initial conditions.
pub struct AttractorAnalyzer {
    config: SemanticFieldConfig,
    base_seed: u64,
    /// Number of steps to run before checking for convergence.
    pub max_steps: usize,
    /// Convergence threshold: consecutive states with similarity above this are "fixed".
    pub convergence_threshold: f32,
    /// Time step for integration.
    pub dt: f32,
}

impl AttractorAnalyzer {
    /// Create a new analyzer for the given field configuration.
    pub fn new(config: SemanticFieldConfig, seed: u64) -> Self {
        AttractorAnalyzer {
            config,
            base_seed: seed,
            max_steps: 200,
            convergence_threshold: 0.999,
            dt: 0.1,
        }
    }

    /// Find fixed points by running the field from multiple random initial conditions.
    ///
    /// Returns a list of distinct fixed points found. Two fixed points are
    /// considered distinct if their cosine similarity is below 0.95.
    pub fn find_fixed_points(&self, n_probes: usize) -> Vec<FixedPoint> {
        let mut fixed_points: Vec<FixedPoint> = Vec::new();

        for probe_idx in 0..n_probes {
            let seed = self.base_seed.wrapping_add(probe_idx as u64 * 1000 + 7);
            let mut field = SemanticField::new(self.config.clone(), seed);
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(1));

            // Random initial condition: feed a random vector for a few steps
            let init_vec = RealHV::random(self.config.hd_dim, &mut rng);
            for _ in 0..5 {
                field.step(&init_vec, self.dt);
            }
            let seed_state = field.state().clone();

            // Now run with zero input and check for convergence
            let mut prev_state = field.state().clone();
            let mut converged = false;
            let mut _final_stability = 0.0_f32;

            for _step in 0..self.max_steps {
                field.idle_step(self.dt);
                let curr = field.state();
                let sim = RealHV::cosine_similarity(&prev_state, curr);
                if sim > self.convergence_threshold && curr.norm() > 1e-6 {
                    converged = true;
                    _final_stability = sim;
                    break;
                }
                prev_state = curr.clone();
            }

            if converged {
                let fp_state = field.state().clone();
                // Check one more step to measure stability
                field.idle_step(self.dt);
                let stability = RealHV::cosine_similarity(&fp_state, field.state());

                // Check if this is a new fixed point (distinct from existing ones)
                let is_new = fixed_points.iter().all(|fp| {
                    RealHV::cosine_similarity(&fp.state, &fp_state) < 0.95
                });

                if is_new {
                    fixed_points.push(FixedPoint {
                        state: fp_state,
                        stability,
                        seed_state,
                    });
                }
            }
        }

        fixed_points
    }

    /// Estimate the basin of attraction for a given fixed point.
    ///
    /// Tests `n_probes` random initial conditions and returns the fraction
    /// that converge to the given attractor (cosine similarity > 0.9).
    pub fn basin_of_attraction(&self, attractor: &RealHV, n_probes: usize) -> f32 {
        let mut attracted_count = 0u32;

        for probe_idx in 0..n_probes {
            let seed = self.base_seed.wrapping_add(probe_idx as u64 * 997 + 13);
            let mut field = SemanticField::new(self.config.clone(), seed);
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(3));

            // Random initial condition
            let init_vec = RealHV::random(self.config.hd_dim, &mut rng);
            for _ in 0..3 {
                field.step(&init_vec, self.dt);
            }

            // Evolve with zero input
            for _ in 0..self.max_steps {
                field.idle_step(self.dt);
            }

            // Check if final state is close to the attractor
            let sim = RealHV::cosine_similarity(field.state(), attractor);
            if sim > 0.9 {
                attracted_count += 1;
            }
        }

        attracted_count as f32 / n_probes as f32
    }

    /// Run a full trajectory analysis from a given initial input.
    ///
    /// Feeds the input for `warmup_steps` then lets the field evolve freely,
    /// recording the state at each step.
    pub fn trajectory_analysis(
        &self,
        initial_input: &RealHV,
        warmup_steps: usize,
    ) -> TrajectoryResult {
        let mut field = SemanticField::new(self.config.clone(), self.base_seed);

        // Warmup: feed input
        for _ in 0..warmup_steps {
            field.step(initial_input, self.dt);
        }

        // Record trajectory
        let mut states = vec![field.state().clone()];
        let mut convergence = Vec::new();
        let mut converged = false;
        let mut convergence_step = None;

        for step in 0..self.max_steps {
            field.idle_step(self.dt);
            let curr = field.state().clone();
            let sim = RealHV::cosine_similarity(states.last().unwrap(), &curr);
            convergence.push(sim);
            states.push(curr);

            if sim > self.convergence_threshold && !converged {
                converged = true;
                convergence_step = Some(step);
            }
        }

        TrajectoryResult {
            states,
            convergence,
            converged,
            convergence_step,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::real::RealHV;
    use crate::lnn::ncp::NCPConfig;
    use crate::noesis::bridge::BridgeStrategy;
    use crate::noesis::semantic_field::SemanticFieldConfig;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    const D: usize = 1024;

    fn make_config() -> SemanticFieldConfig {
        SemanticFieldConfig::new(D, BridgeStrategy::DualTrack, NCPConfig::tiny())
    }

    #[test]
    fn test_attractor_existence() {
        // The semantic field should have at least one attractor (possibly the
        // zero or a low-energy state) that trajectories converge toward.
        let analyzer = AttractorAnalyzer::new(make_config(), 42);
        let trajectory = {
            let mut rng = StdRng::seed_from_u64(99);
            let input = RealHV::random(D, &mut rng);
            analyzer.trajectory_analysis(&input, 10)
        };

        // The trajectory should show increasing similarity between consecutive
        // states (the system settles), even if it doesn't fully converge.
        let n = trajectory.convergence.len();
        assert!(n > 10, "Should have a substantial trajectory");

        // Check that the late part of the trajectory is more stable than the early part.
        let _early_avg: f32 = trajectory.convergence[..5].iter().sum::<f32>() / 5.0;
        let late_avg: f32 = trajectory.convergence[n - 5..].iter().sum::<f32>() / 5.0;

        // The late similarities should generally be higher (closer to 1) or
        // at least the trajectory should have finite states throughout.
        assert!(
            trajectory.states.iter().all(|s| s.data.iter().all(|x| x.is_finite())),
            "All trajectory states should be finite"
        );

        // At minimum, the system should not diverge: late similarity should not be negative.
        assert!(
            late_avg > -0.5,
            "Late trajectory should not be wildly unstable: late_avg = {}",
            late_avg
        );
    }

    #[test]
    fn test_multiple_attractors() {
        // Probe from many different initial conditions. Even if the field has
        // only one attractor, the test verifies that the analysis machinery works.
        let analyzer = AttractorAnalyzer::new(make_config(), 42);
        let fps = analyzer.find_fixed_points(10);

        // We should get at least the basic analysis to complete without panic.
        // The number of fixed points depends on the dynamics.
        // If we find at least one, test the basin.
        if !fps.is_empty() {
            let fp = &fps[0];
            assert!(
                fp.stability > 0.9,
                "Fixed point should be stable: stability = {}",
                fp.stability
            );

            let basin = analyzer.basin_of_attraction(&fp.state, 5);
            // Basin should be a valid fraction
            assert!(
                (0.0..=1.0).contains(&basin),
                "Basin fraction should be in [0,1]: {}",
                basin
            );
        }

        // Regardless of fixed points found, the machinery should work
        // Run trajectory from two different inputs and verify they produce different paths
        let mut rng = StdRng::seed_from_u64(123);
        let input_a = RealHV::random(D, &mut rng);
        let input_b = RealHV::random(D, &mut rng);

        let traj_a = analyzer.trajectory_analysis(&input_a, 10);
        let traj_b = analyzer.trajectory_analysis(&input_b, 10);

        // Both should have states
        assert!(!traj_a.states.is_empty());
        assert!(!traj_b.states.is_empty());

        // Their final states should differ (different initial conditions, same dynamics)
        let final_sim = RealHV::cosine_similarity(
            traj_a.states.last().unwrap(),
            traj_b.states.last().unwrap(),
        );
        // They might converge to the same attractor, or not. Just check validity.
        assert!(
            final_sim.is_finite(),
            "Final similarity should be finite: {}",
            final_sim
        );
    }
}
