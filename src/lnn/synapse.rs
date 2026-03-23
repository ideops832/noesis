//! Synaptic connection with its own dynamics.
//!
//! Each synapse has a weight, a time constant τ, and internal state.
//! The dynamics follow: `state += dt * (-state + pre_activation * weight) / tau`

use serde::{Deserialize, Serialize};

/// A synaptic connection with first-order dynamics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Synapse {
    /// Synaptic weight.
    pub weight: f32,
    /// Time constant controlling response speed.
    pub tau: f32,
    /// Internal synaptic state.
    pub state: f32,
}

impl Synapse {
    /// Creates a new synapse.
    pub fn new(weight: f32, tau: f32) -> Self {
        Synapse {
            weight,
            tau,
            state: 0.0,
        }
    }

    /// Transmits a signal through the synapse with first-order dynamics.
    ///
    /// `state += dt * (-state + pre_activation * weight) / tau`
    pub fn transmit(&mut self, pre_activation: f32, dt: f32) -> f32 {
        self.state += dt * (-self.state + pre_activation * self.weight) / self.tau;
        self.state
    }

    /// Resets the synaptic state.
    pub fn reset(&mut self) {
        self.state = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synapse_dynamics() {
        let mut fast = Synapse::new(1.0, 0.1);
        let mut slow = Synapse::new(1.0, 10.0);

        let target = 1.0;
        let dt = 0.01;

        // Run both for 100 steps
        for _ in 0..100 {
            fast.transmit(target, dt);
            slow.transmit(target, dt);
        }

        // Fast synapse (τ=0.1) should reach ~80% faster
        let fast_frac = fast.state / target;
        let slow_frac = slow.state / target;

        assert!(
            fast_frac > slow_frac,
            "Fast τ=0.1 ({fast_frac:.3}) should be closer to target than slow τ=10.0 ({slow_frac:.3})"
        );
        assert!(
            fast_frac > 0.5,
            "Fast synapse should be well past 50%: {fast_frac:.3}"
        );
    }
}
