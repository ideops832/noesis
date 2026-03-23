//! Neural Circuit Policy (NCP) architecture with preset configurations.
//!
//! Four neuron groups inspired by C. elegans connectome:
//! Sensory → Inter → Command → Motor
//!
//! Each CfC neuron has hidden_size=1, producing a single scalar output.
//! The output of a layer is the concatenation of all neuron outputs.

use rand::Rng;
use serde::{Deserialize, Serialize};

use super::neuron::CfCNeuron;

/// Configuration for an NCP architecture.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NCPConfig {
    pub sensory_size: usize,
    pub inter_size: usize,
    pub command_size: usize,
    pub motor_size: usize,
    pub inter_recurrent: bool,
    pub command_recurrent: bool,
}

impl NCPConfig {
    /// Tiny preset: 8→4→4→4 = 20 neurons.
    pub fn tiny() -> Self {
        NCPConfig {
            sensory_size: 8,
            inter_size: 4,
            command_size: 4,
            motor_size: 4,
            inter_recurrent: true,
            command_recurrent: true,
        }
    }

    /// Standard preset: 32→16→16→8 = 72 neurons.
    pub fn standard() -> Self {
        NCPConfig {
            sensory_size: 32,
            inter_size: 16,
            command_size: 16,
            motor_size: 8,
            inter_recurrent: true,
            command_recurrent: true,
        }
    }

    /// Total number of neurons.
    pub fn total_neurons(&self) -> usize {
        self.sensory_size + self.inter_size + self.command_size + self.motor_size
    }
}

/// State of the NCP at a given time.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NCPState {
    pub inter: Vec<f32>,
    pub command: Vec<f32>,
    pub motor: Vec<f32>,
    pub t: f32,
}

/// Effective time constants per neuron group.
#[derive(Clone, Debug)]
pub struct NCPTau {
    pub inter: Vec<f32>,
    pub command: Vec<f32>,
    pub motor: Vec<f32>,
}

/// Neural Circuit Policy with CfC neurons.
#[derive(Clone, Serialize, Deserialize)]
pub struct NCP {
    config: NCPConfig,
    inter_neurons: Vec<CfCNeuron>,
    command_neurons: Vec<CfCNeuron>,
    motor_neurons: Vec<CfCNeuron>,
    state_inter: Vec<f32>,
    state_command: Vec<f32>,
    state_motor: Vec<f32>,
    t: f32,
}

impl NCP {
    /// Creates a new NCP with the given configuration.
    pub fn new(config: NCPConfig, rng: &mut impl Rng) -> Self {
        let inter_input_size = if config.inter_recurrent {
            config.sensory_size + config.inter_size
        } else {
            config.sensory_size
        };

        let command_input_size = if config.command_recurrent {
            config.inter_size + config.command_size
        } else {
            config.inter_size
        };

        let motor_input_size = config.command_size;

        let inter_neurons: Vec<CfCNeuron> = (0..config.inter_size)
            .map(|_| CfCNeuron::new(inter_input_size, 1, rng))
            .collect();

        let command_neurons: Vec<CfCNeuron> = (0..config.command_size)
            .map(|_| CfCNeuron::new(command_input_size, 1, rng))
            .collect();

        let motor_neurons: Vec<CfCNeuron> = (0..config.motor_size)
            .map(|_| CfCNeuron::new(motor_input_size, 1, rng))
            .collect();

        NCP {
            state_inter: vec![0.0; config.inter_size],
            state_command: vec![0.0; config.command_size],
            state_motor: vec![0.0; config.motor_size],
            config,
            inter_neurons,
            command_neurons,
            motor_neurons,
            t: 0.0,
        }
    }

    /// Performs one forward step through the NCP.
    ///
    /// Input must have length == config.sensory_size.
    /// Returns the motor output.
    pub fn step(&mut self, input: &[f32], dt: f32) -> Vec<f32> {
        assert_eq!(
            input.len(),
            self.config.sensory_size,
            "Input size mismatch: expected {}, got {}",
            self.config.sensory_size,
            input.len()
        );

        // Inter layer
        for i in 0..self.config.inter_size {
            let neuron_input = if self.config.inter_recurrent {
                let mut v = Vec::with_capacity(self.config.sensory_size + self.config.inter_size);
                v.extend_from_slice(input);
                v.extend_from_slice(&self.state_inter);
                v
            } else {
                input.to_vec()
            };
            let out = self.inter_neurons[i].step(&neuron_input, dt);
            self.state_inter[i] = out[0];
        }

        // Command layer
        for i in 0..self.config.command_size {
            let neuron_input = if self.config.command_recurrent {
                let mut v =
                    Vec::with_capacity(self.config.inter_size + self.config.command_size);
                v.extend_from_slice(&self.state_inter);
                v.extend_from_slice(&self.state_command);
                v
            } else {
                self.state_inter.clone()
            };
            let out = self.command_neurons[i].step(&neuron_input, dt);
            self.state_command[i] = out[0];
        }

        // Motor layer (never recurrent)
        for i in 0..self.config.motor_size {
            let out = self.motor_neurons[i].step(&self.state_command, dt);
            self.state_motor[i] = out[0];
        }

        self.t += dt;
        self.state_motor.clone()
    }

    /// Resets all neuron states to zero.
    pub fn reset(&mut self) {
        self.state_inter.fill(0.0);
        self.state_command.fill(0.0);
        self.state_motor.fill(0.0);
        for n in &mut self.inter_neurons {
            n.reset();
        }
        for n in &mut self.command_neurons {
            n.reset();
        }
        for n in &mut self.motor_neurons {
            n.reset();
        }
        self.t = 0.0;
    }

    /// Returns the current NCP state.
    pub fn get_state(&self) -> NCPState {
        NCPState {
            inter: self.state_inter.clone(),
            command: self.state_command.clone(),
            motor: self.state_motor.clone(),
            t: self.t,
        }
    }

    /// Returns effective time constants (gate values) for each neuron.
    pub fn get_effective_tau(&self, input: &[f32]) -> NCPTau {
        let inter_tau: Vec<f32> = self
            .inter_neurons
            .iter()
            .map(|n| {
                let neuron_input = if self.config.inter_recurrent {
                    let mut v = Vec::with_capacity(self.config.sensory_size + self.config.inter_size);
                    v.extend_from_slice(input);
                    v.extend_from_slice(&self.state_inter);
                    v
                } else {
                    input.to_vec()
                };
                let gates = n.get_gate_values(&neuron_input);
                // Effective tau ≈ 1/gate (higher gate = faster = lower tau)
                if gates[0] > 1e-6 { 1.0 / gates[0] } else { 1e6 }
            })
            .collect();

        let command_tau: Vec<f32> = self
            .command_neurons
            .iter()
            .map(|n| {
                let neuron_input = if self.config.command_recurrent {
                    let mut v = Vec::with_capacity(self.config.inter_size + self.config.command_size);
                    v.extend_from_slice(&self.state_inter);
                    v.extend_from_slice(&self.state_command);
                    v
                } else {
                    self.state_inter.clone()
                };
                let gates = n.get_gate_values(&neuron_input);
                if gates[0] > 1e-6 { 1.0 / gates[0] } else { 1e6 }
            })
            .collect();

        let motor_tau: Vec<f32> = self
            .motor_neurons
            .iter()
            .map(|n| {
                let gates = n.get_gate_values(&self.state_command);
                if gates[0] > 1e-6 { 1.0 / gates[0] } else { 1e6 }
            })
            .collect();

        NCPTau {
            inter: inter_tau,
            command: command_tau,
            motor: motor_tau,
        }
    }

    /// Input size (sensory).
    pub fn input_size(&self) -> usize {
        self.config.sensory_size
    }

    /// Output size (motor).
    pub fn output_size(&self) -> usize {
        self.config.motor_size
    }

    /// Total number of trainable parameters.
    pub fn total_parameters(&self) -> usize {
        let inter: usize = self.inter_neurons.iter().map(|n| n.total_parameters()).sum();
        let cmd: usize = self.command_neurons.iter().map(|n| n.total_parameters()).sum();
        let motor: usize = self.motor_neurons.iter().map(|n| n.total_parameters()).sum();
        inter + cmd + motor
    }

    /// Extracts all trainable parameters as a flat vector.
    pub fn get_params(&self) -> Vec<f32> {
        let mut params = Vec::with_capacity(self.total_parameters());
        for n in &self.inter_neurons {
            params.extend(n.get_params());
        }
        for n in &self.command_neurons {
            params.extend(n.get_params());
        }
        for n in &self.motor_neurons {
            params.extend(n.get_params());
        }
        params
    }

    /// Sets all trainable parameters from a flat vector.
    pub fn set_params(&mut self, params: &[f32]) {
        assert_eq!(params.len(), self.total_parameters(), "Parameter count mismatch");
        let mut offset = 0;
        for n in &mut self.inter_neurons {
            let np = n.total_parameters();
            n.set_params(&params[offset..offset + np]);
            offset += np;
        }
        for n in &mut self.command_neurons {
            let np = n.total_parameters();
            n.set_params(&params[offset..offset + np]);
            offset += np;
        }
        for n in &mut self.motor_neurons {
            let np = n.total_parameters();
            n.set_params(&params[offset..offset + np]);
            offset += np;
        }
    }

    /// Human-readable summary of the architecture.
    pub fn summary(&self) -> String {
        let rec = match (self.config.inter_recurrent, self.config.command_recurrent) {
            (true, true) => "inter+command",
            (true, false) => "inter",
            (false, true) => "command",
            (false, false) => "none",
        };
        let preset = if self.config.sensory_size == 8
            && self.config.inter_size == 4
            && self.config.command_size == 4
            && self.config.motor_size == 4
        {
            "tiny"
        } else if self.config.sensory_size == 32
            && self.config.inter_size == 16
            && self.config.command_size == 16
            && self.config.motor_size == 8
        {
            "standard"
        } else {
            "custom"
        };
        format!(
            "NCP [{}] {}→{}→{}→{} ({} neurons, {} parameters, recurrence: {})",
            preset,
            self.config.sensory_size,
            self.config.inter_size,
            self.config.command_size,
            self.config.motor_size,
            self.config.total_neurons(),
            self.total_parameters(),
            rec
        )
    }

    /// Access to config.
    pub fn config(&self) -> &NCPConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_ncp_forward_tiny() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut ncp = NCP::new(NCPConfig::tiny(), &mut rng);
        let input = vec![0.5; 8];
        let output = ncp.step(&input, 0.1);
        assert_eq!(output.len(), 4, "Tiny output should be 4");
        assert!(output.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_ncp_forward_standard() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut ncp = NCP::new(NCPConfig::standard(), &mut rng);
        let input = vec![0.5; 32];
        let output = ncp.step(&input, 0.1);
        assert_eq!(output.len(), 8, "Standard output should be 8");
        assert!(output.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_ncp_state_persistence() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut ncp = NCP::new(NCPConfig::tiny(), &mut rng);
        let input = vec![0.5; 8];
        let out1 = ncp.step(&input, 0.1);
        let out2 = ncp.step(&input, 0.1);
        // State evolves, so outputs should differ
        assert_ne!(out1, out2, "Two consecutive steps should produce different outputs");
    }

    #[test]
    fn test_ncp_reset() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut ncp = NCP::new(NCPConfig::tiny(), &mut rng);
        let input = vec![0.5; 8];
        let out1 = ncp.step(&input, 0.1);
        ncp.step(&input, 0.1); // second step
        ncp.reset();
        let out3 = ncp.step(&input, 0.1);
        // After reset, first step should match original first step
        let diff: f32 = out1.iter().zip(&out3).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff < 1e-6,
            "After reset, output differs from first step: diff = {diff}"
        );
    }

    #[test]
    fn test_ncp_recurrence_effect() {
        let mut rng1 = StdRng::seed_from_u64(42);
        let mut ncp_rec = NCP::new(NCPConfig::tiny(), &mut rng1);

        let mut rng2 = StdRng::seed_from_u64(42);
        let no_rec_config = NCPConfig {
            inter_recurrent: false,
            command_recurrent: false,
            ..NCPConfig::tiny()
        };
        let mut ncp_no_rec = NCP::new(no_rec_config, &mut rng2);

        let input = vec![0.5; 8];
        let mut out_rec = vec![];
        let mut out_no_rec = vec![];

        for _ in 0..10 {
            out_rec = ncp_rec.step(&input, 0.1);
            out_no_rec = ncp_no_rec.step(&input, 0.1);
        }

        // After several steps, recurrent vs non-recurrent should differ
        let diff: f32 = out_rec
            .iter()
            .zip(&out_no_rec)
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-6,
            "Recurrent vs non-recurrent should differ after 10 steps: diff = {diff}"
        );
    }

    #[test]
    fn test_ncp_minimal() {
        let config = NCPConfig {
            sensory_size: 2,
            inter_size: 1,
            command_size: 1,
            motor_size: 1,
            inter_recurrent: false,
            command_recurrent: false,
        };
        let mut rng = StdRng::seed_from_u64(42);
        let mut ncp = NCP::new(config, &mut rng);
        let output = ncp.step(&[1.0, -1.0], 0.1);
        assert_eq!(output.len(), 1);
        assert!(output[0].is_finite());
    }

    #[test]
    fn test_ncp_serialize() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut ncp = NCP::new(NCPConfig::tiny(), &mut rng);
        let input = vec![0.5; 8];
        ncp.step(&input, 0.1); // advance state

        let json = serde_json::to_string(&ncp).expect("serialize");
        let mut ncp2: NCP = serde_json::from_str(&json).expect("deserialize");

        // Both should produce similar results from the same serialized state
        let out1 = ncp.step(&input, 0.1);
        let out2 = ncp2.step(&input, 0.1);

        let diff: f32 = out1
            .iter()
            .zip(&out2)
            .map(|(a, b)| (a - b).abs())
            .sum();
        // JSON float serialization may lose some precision
        assert!(
            diff < 0.01,
            "Deserialized NCP diverges: diff = {diff}"
        );
    }

    #[test]
    fn test_ncp_summary() {
        let mut rng = StdRng::seed_from_u64(42);
        let ncp_tiny = NCP::new(NCPConfig::tiny(), &mut rng);
        let summary = ncp_tiny.summary();
        println!("{summary}");
        assert!(summary.contains("tiny"));
        assert!(summary.contains("8→4→4→4"));
        assert!(summary.contains("20 neurons"));

        let ncp_std = NCP::new(NCPConfig::standard(), &mut rng);
        let summary = ncp_std.summary();
        println!("{summary}");
        assert!(summary.contains("standard"));
        assert!(summary.contains("32→16→16→8"));
        assert!(summary.contains("72 neurons"));
    }
}
