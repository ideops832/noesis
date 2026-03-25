//! Bridge trait and implementations connecting HDC space to LNN dynamics.
//!
//! Strategies:
//! - A: RandomProjection (JL-style)
//! - B: DirectHD (LNN operates in full HD space)
//! - C: DualTrack (LNN controls HDC operations via probes)
//! - D: SparseHD (each neuron sees a sparse subset of dimensions)
//! - E: InputPreserving (LNN controls mixing rate only)
//! - F: SemanticDualTrack (semantic probes + input-preserving mixing)
//! - R6: AdaptiveDualTrack (LVQ-style probe learning)

use std::cell::RefCell;

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::hdc::real::RealHV;
use crate::lnn::neuron::matmul_flat;

/// Strategy enum for bridge selection.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum BridgeStrategy {
    RandomProjection,
    DirectHD,
    DualTrack,
    SparseHD,
    /// Input-preserving bridge: the NCP controls mixing rate,
    /// but the update direction is always the input itself.
    InputPreserving,
    /// Semantic DualTrack: uses semantic probes (not random) and
    /// combines input-preserving mixing with probe-guided adjustment.
    SemanticDualTrack,
    /// Adaptive DualTrack: like DualTrack but probes adapt via LVQ-style
    /// winner-take-all learning during encode. Probes move toward inputs
    /// they are closest to, becoming category centroids over time.
    AdaptiveDualTrack,
}

/// Bridge configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BridgeConfig {
    pub strategy: BridgeStrategy,
    pub hd_dim: usize,
}

/// Trait for bridge implementations between HDC and LNN spaces.
pub trait Bridge: Send + Sync {
    /// Encodes HD state and input into a vector for the LNN.
    fn encode_for_lnn(&self, state: &RealHV, input: &RealHV) -> Vec<f32>;

    /// Decodes LNN output into an HD state update.
    fn decode_from_lnn(&self, lnn_output: &[f32], current_state: &RealHV) -> RealHV;

    /// Human-readable name of this strategy.
    fn name(&self) -> &str;

    /// Size of the input vector produced for the LNN.
    fn lnn_input_size(&self) -> usize;

    /// Size of the output vector expected from the LNN.
    fn lnn_output_size(&self) -> usize;

    /// HD dimensionality handled by this bridge.
    fn hd_dim(&self) -> usize;
}

// =============================================================================
// Strategy A: Random Projection (Johnson-Lindenstrauss)
// =============================================================================

/// Bridge using random projection matrices (JL-style).
pub struct RandomProjectionBridge {
    /// Projection matrix: lnn_input_size × hd_dim (row-major).
    projection_in: Vec<f32>,
    /// Back-projection matrix: hd_dim × lnn_output_size (row-major).
    projection_out: Vec<f32>,
    hd_dim: usize,
    lnn_input_size: usize,
    lnn_output_size: usize,
}

impl RandomProjectionBridge {
    pub fn new(
        hd_dim: usize,
        lnn_input_size: usize,
        lnn_output_size: usize,
        rng: &mut impl Rng,
    ) -> Self {
        let scale = 1.0 / (hd_dim as f32).sqrt();
        let projection_in: Vec<f32> = (0..lnn_input_size * hd_dim)
            .map(|_| rng.gen_range(-scale..scale))
            .collect();
        let projection_out: Vec<f32> = (0..hd_dim * lnn_output_size)
            .map(|_| rng.gen_range(-scale..scale))
            .collect();

        RandomProjectionBridge {
            projection_in,
            projection_out,
            hd_dim,
            lnn_input_size,
            lnn_output_size,
        }
    }
}

impl Bridge for RandomProjectionBridge {
    fn encode_for_lnn(&self, state: &RealHV, input: &RealHV) -> Vec<f32> {
        let combined = RealHV::bundle_normalized(&[state, input]);
        matmul_flat(&self.projection_in, &combined.data, self.lnn_input_size, self.hd_dim)
    }

    fn decode_from_lnn(&self, lnn_output: &[f32], _current_state: &RealHV) -> RealHV {
        let data = matmul_flat(&self.projection_out, lnn_output, self.hd_dim, self.lnn_output_size);
        RealHV { data, dim: self.hd_dim }
    }

    fn name(&self) -> &str { "RandomProjection" }
    fn lnn_input_size(&self) -> usize { self.lnn_input_size }
    fn lnn_output_size(&self) -> usize { self.lnn_output_size }
    fn hd_dim(&self) -> usize { self.hd_dim }
}

// =============================================================================
// Strategy B: Direct HD (LNN operates in full HD space)
// =============================================================================

/// Merge strategy for combining state and input in direct mode.
#[derive(Clone, Debug)]
pub enum MergeStrategy {
    Bundle,
}

/// Bridge where the LNN operates directly in HD space.
pub struct DirectHDBridge {
    hd_dim: usize,
}

impl DirectHDBridge {
    pub fn new(hd_dim: usize) -> Self {
        DirectHDBridge { hd_dim }
    }
}

impl Bridge for DirectHDBridge {
    fn encode_for_lnn(&self, state: &RealHV, input: &RealHV) -> Vec<f32> {
        RealHV::bundle_normalized(&[state, input]).data
    }

    fn decode_from_lnn(&self, lnn_output: &[f32], _current_state: &RealHV) -> RealHV {
        RealHV::from_slice(lnn_output)
    }

    fn name(&self) -> &str { "DirectHD" }
    fn lnn_input_size(&self) -> usize { self.hd_dim }
    fn lnn_output_size(&self) -> usize { self.hd_dim }
    fn hd_dim(&self) -> usize { self.hd_dim }
}

// =============================================================================
// Strategy C: Dual Track (LNN controls HDC operations via probes)
// =============================================================================

/// Bridge where the LNN operates on similarity features extracted via probes,
/// and produces parameters for HDC operations on the state.
pub struct DualTrackBridge {
    probes: Vec<RealHV>,
    n_probes: usize,
    hd_dim: usize,
}

impl DualTrackBridge {
    pub fn new(hd_dim: usize, n_probes: usize, rng: &mut impl Rng) -> Self {
        let probes: Vec<RealHV> = (0..n_probes)
            .map(|_| RealHV::random(hd_dim, rng).normalized())
            .collect();
        DualTrackBridge {
            probes,
            n_probes,
            hd_dim,
        }
    }
}

impl Bridge for DualTrackBridge {
    fn encode_for_lnn(&self, state: &RealHV, input: &RealHV) -> Vec<f32> {
        let combined = RealHV::bundle_normalized(&[state, input]);
        self.probes
            .iter()
            .map(|probe| RealHV::cosine_similarity(&combined, probe))
            .collect()
    }

    fn decode_from_lnn(&self, lnn_output: &[f32], current_state: &RealHV) -> RealHV {
        // Last value is alpha (mixing parameter)
        let n_weights = self.n_probes - 1;
        let raw_alpha = if n_weights < lnn_output.len() {
            lnn_output[n_weights]
        } else {
            0.1
        };
        let alpha = crate::lnn::neuron::sigmoid(raw_alpha) * 0.5; // Keep alpha in [0, 0.5]

        // Weighted sum of probes
        let mut delta = RealHV::zero(self.hd_dim);
        let weight_count = n_weights.min(lnn_output.len());
        for i in 0..weight_count {
            let scaled = self.probes[i].scale(lnn_output[i]);
            delta = RealHV::add(&delta, &scaled);
        }
        delta.normalize();

        // Mix: new_state = (1-alpha)*current + alpha*delta
        let kept = current_state.scale(1.0 - alpha);
        let added = delta.scale(alpha);
        let mut result = RealHV::add(&kept, &added);
        result.normalize();
        result
    }

    fn name(&self) -> &str { "DualTrack" }
    fn lnn_input_size(&self) -> usize { self.n_probes }
    fn lnn_output_size(&self) -> usize { self.n_probes }
    fn hd_dim(&self) -> usize { self.hd_dim }
}

impl DualTrackBridge {
    /// Number of trainable probe parameters.
    pub fn probe_param_count(&self) -> usize {
        self.n_probes * self.hd_dim
    }

    /// Extract probe data as flat vector.
    pub fn get_probe_params(&self) -> Vec<f32> {
        let mut params = Vec::with_capacity(self.probe_param_count());
        for probe in &self.probes {
            params.extend_from_slice(&probe.data);
        }
        params
    }

    /// Set probe data from flat vector.
    pub fn set_probe_params(&mut self, params: &[f32]) {
        assert_eq!(params.len(), self.probe_param_count());
        for (i, probe) in self.probes.iter_mut().enumerate() {
            let offset = i * self.hd_dim;
            probe.data.copy_from_slice(&params[offset..offset + self.hd_dim]);
            probe.normalize();
        }
    }
}

// =============================================================================
// Strategy D: Sparse HD (each neuron sees a subset of dimensions)
// =============================================================================

/// Bridge where each virtual neuron sees a sparse subset of HD dimensions.
pub struct SparseHDBridge {
    /// For each neuron: the indices of dimensions it sees.
    neuron_masks: Vec<Vec<usize>>,
    hd_dim: usize,
    n_neurons: usize,
    dims_per_neuron: usize,
}

impl SparseHDBridge {
    pub fn new(
        hd_dim: usize,
        n_neurons: usize,
        overlap_factor: usize,
        rng: &mut impl Rng,
    ) -> Self {
        let dims_per_neuron = (hd_dim * overlap_factor) / n_neurons;
        let dims_per_neuron = dims_per_neuron.max(1);

        let mut neuron_masks: Vec<Vec<usize>> = (0..n_neurons)
            .map(|_| {
                let mut indices: Vec<usize> = (0..dims_per_neuron)
                    .map(|_| rng.gen_range(0..hd_dim))
                    .collect();
                indices.sort_unstable();
                indices.dedup();
                // Ensure minimum coverage
                while indices.len() < dims_per_neuron {
                    let idx = rng.gen_range(0..hd_dim);
                    if !indices.contains(&idx) {
                        indices.push(idx);
                        indices.sort_unstable();
                    }
                }
                indices
            })
            .collect();

        // Ensure full coverage: every dimension is seen by at least one neuron
        let mut covered = vec![false; hd_dim];
        for mask in &neuron_masks {
            for &idx in mask {
                covered[idx] = true;
            }
        }
        for (dim_idx, &is_covered) in covered.iter().enumerate() {
            if !is_covered {
                // Add to a random neuron
                let n = rng.gen_range(0..n_neurons);
                neuron_masks[n].push(dim_idx);
            }
        }

        SparseHDBridge {
            neuron_masks,
            hd_dim,
            n_neurons,
            dims_per_neuron,
        }
    }
}

impl Bridge for SparseHDBridge {
    fn encode_for_lnn(&self, state: &RealHV, input: &RealHV) -> Vec<f32> {
        let combined = RealHV::bundle_normalized(&[state, input]);
        let mut result = Vec::with_capacity(self.n_neurons * self.dims_per_neuron);
        for mask in &self.neuron_masks {
            for &idx in mask.iter().take(self.dims_per_neuron) {
                result.push(combined.data[idx]);
            }
            // Pad if mask is shorter than dims_per_neuron
            for _ in mask.len()..self.dims_per_neuron {
                result.push(0.0);
            }
        }
        result
    }

    fn decode_from_lnn(&self, lnn_output: &[f32], _current_state: &RealHV) -> RealHV {
        let mut accum = vec![0.0f32; self.hd_dim];
        let mut counts = vec![0u32; self.hd_dim];

        for (neuron_idx, mask) in self.neuron_masks.iter().enumerate() {
            let offset = neuron_idx * self.dims_per_neuron;
            for (local_idx, &dim_idx) in mask.iter().take(self.dims_per_neuron).enumerate() {
                if offset + local_idx < lnn_output.len() {
                    accum[dim_idx] += lnn_output[offset + local_idx];
                    counts[dim_idx] += 1;
                }
            }
        }

        let data: Vec<f32> = accum
            .iter()
            .zip(&counts)
            .map(|(&sum, &count)| if count > 0 { sum / count as f32 } else { 0.0 })
            .collect();

        RealHV { data, dim: self.hd_dim }
    }

    fn name(&self) -> &str { "SparseHD" }
    fn lnn_input_size(&self) -> usize { self.n_neurons * self.dims_per_neuron }
    fn lnn_output_size(&self) -> usize { self.n_neurons * self.dims_per_neuron }
    fn hd_dim(&self) -> usize { self.hd_dim }
}

// =============================================================================
// Strategy E: Input-Preserving (LNN controls mixing rate only)
// =============================================================================

/// Bridge that preserves the input directly. The NCP only controls the
/// mixing rate alpha between old state and new input. This maximally
/// preserves HDC algebraic structure because the state update is:
///
///   new_state = normalize((1 - alpha) * old_state + alpha * input)
///
/// where alpha = sigmoid(mean(ncp_output)) * alpha_max
///
/// The NCP receives similarity features (like DualTrack) and its output
/// is interpreted purely as a scalar mixing signal.
pub struct InputPreservingBridge {
    probes: Vec<RealHV>,
    n_probes: usize,
    hd_dim: usize,
    alpha_max: f32,
}

impl InputPreservingBridge {
    pub fn new(hd_dim: usize, n_probes: usize, alpha_max: f32, rng: &mut impl Rng) -> Self {
        let probes: Vec<RealHV> = (0..n_probes)
            .map(|_| RealHV::random(hd_dim, rng).normalized())
            .collect();
        InputPreservingBridge {
            probes,
            n_probes,
            hd_dim,
            alpha_max,
        }
    }
}

impl Bridge for InputPreservingBridge {
    fn encode_for_lnn(&self, state: &RealHV, input: &RealHV) -> Vec<f32> {
        // Features: similarity of (state, input, combined) with each probe
        let combined = RealHV::bundle_normalized(&[state, input]);
        let mut features = Vec::with_capacity(self.n_probes);
        for probe in &self.probes {
            features.push(RealHV::cosine_similarity(&combined, probe));
        }
        features
    }

    fn decode_from_lnn(&self, lnn_output: &[f32], current_state: &RealHV) -> RealHV {
        // The LNN output is interpreted as a mixing signal.
        // Mean of output → sigmoid → alpha
        let mean_output: f32 = if lnn_output.is_empty() {
            0.0
        } else {
            lnn_output.iter().sum::<f32>() / lnn_output.len() as f32
        };
        let alpha = crate::lnn::neuron::sigmoid(mean_output) * self.alpha_max;

        // The "input" was already blended into the combined signal during encode.
        // We need the original input. Since we don't have it here, we use a trick:
        // the state from encode was bundle(state, input). We can approximate
        // the input direction as: 2*combined - state (since combined ≈ (state+input)/norm)
        // But this is lossy. Instead, we store the input in the bridge.
        //
        // Actually, in this architecture the decode doesn't have access to the
        // original input. So we make the decode a simple damping/inertia controller:
        // new_state = (1 - alpha) * current_state
        //
        // The actual input injection happens at a higher level (SemanticField).
        // We return a scaled version of current_state.
        let mut result = current_state.scale(1.0 - alpha);
        result.normalize();
        result
    }

    fn name(&self) -> &str { "InputPreserving" }
    fn lnn_input_size(&self) -> usize { self.n_probes }
    fn lnn_output_size(&self) -> usize { self.n_probes }
    fn hd_dim(&self) -> usize { self.hd_dim }
}

// =============================================================================
// Strategy F: Semantic DualTrack
// =============================================================================

/// Bridge combining semantic probes with input-preserving mixing.
///
/// The LNN receives rich features: similarity with semantic probes PLUS
/// similarity between state and input. Its output controls:
/// - alpha: how much to mix input into state (input-preserving)
/// - beta: how much to adjust via probe-guided delta (DualTrack-style)
///
/// Update rule:
///   delta = weighted_sum(probes, ncp_weights)
///   new_state = normalize((1-alpha-beta)*state + alpha*input + beta*delta)
pub struct SemanticDualTrackBridge {
    /// Semantic probes (can be role vectors, category vectors, etc.)
    probes: Vec<RealHV>,
    n_probes: usize,
    hd_dim: usize,
}

impl SemanticDualTrackBridge {
    /// Creates with random probes (will be replaced by semantic ones).
    pub fn new(hd_dim: usize, n_probes: usize, rng: &mut impl Rng) -> Self {
        let probes: Vec<RealHV> = (0..n_probes)
            .map(|_| RealHV::random(hd_dim, rng).normalized())
            .collect();
        SemanticDualTrackBridge { probes, n_probes, hd_dim }
    }

    /// Creates with pre-defined semantic probes.
    pub fn with_probes(hd_dim: usize, probes: Vec<RealHV>) -> Self {
        let n_probes = probes.len();
        SemanticDualTrackBridge { probes, n_probes, hd_dim }
    }

    /// Get/set probe parameters for training.
    pub fn probe_param_count(&self) -> usize {
        self.n_probes * self.hd_dim
    }

    pub fn get_probe_params(&self) -> Vec<f32> {
        let mut params = Vec::with_capacity(self.probe_param_count());
        for probe in &self.probes {
            params.extend_from_slice(&probe.data);
        }
        params
    }

    pub fn set_probe_params(&mut self, params: &[f32]) {
        assert_eq!(params.len(), self.probe_param_count());
        for (i, probe) in self.probes.iter_mut().enumerate() {
            let offset = i * self.hd_dim;
            probe.data.copy_from_slice(&params[offset..offset + self.hd_dim]);
            probe.normalize();
        }
    }
}

impl Bridge for SemanticDualTrackBridge {
    fn encode_for_lnn(&self, state: &RealHV, input: &RealHV) -> Vec<f32> {
        // Features: probe similarities + state-input similarity + input norm signal
        let combined = RealHV::bundle_normalized(&[state, input]);
        let mut features = Vec::with_capacity(self.n_probes);
        for probe in &self.probes {
            features.push(RealHV::cosine_similarity(&combined, probe));
        }
        features
    }

    fn decode_from_lnn(&self, lnn_output: &[f32], current_state: &RealHV) -> RealHV {
        // Split output: first half = probe weights, last 2 = alpha, beta
        let n_weights = if self.n_probes > 2 { self.n_probes - 2 } else { 0 };

        let raw_alpha = if n_weights < lnn_output.len() {
            lnn_output[n_weights]
        } else {
            0.0
        };
        let raw_beta = if n_weights + 1 < lnn_output.len() {
            lnn_output[n_weights + 1]
        } else {
            0.0
        };

        let alpha = crate::lnn::neuron::sigmoid(raw_alpha) * 0.2; // input mixing [0, 0.2]
        let beta = crate::lnn::neuron::sigmoid(raw_beta) * 0.1;   // probe adjustment [0, 0.1]

        // Probe-guided delta
        let mut delta = RealHV::zero(self.hd_dim);
        for i in 0..n_weights.min(lnn_output.len()) {
            let scaled = self.probes[i].scale(lnn_output[i]);
            delta = RealHV::add(&delta, &scaled);
        }
        if delta.norm() > 1e-8 {
            delta.normalize();
        }

        // Combined update: state*(1-alpha-beta) + input_proxy*alpha + delta*beta
        // Note: we don't have input here, so the input part is handled in SemanticField.step()
        // Here we just apply the probe-guided adjustment
        let kept = current_state.scale(1.0 - beta);
        let adjustment = delta.scale(beta);
        let mut result = RealHV::add(&kept, &adjustment);
        if result.norm() > 1e-8 {
            result.normalize();
        }
        result
    }

    fn name(&self) -> &str { "SemanticDualTrack" }
    fn lnn_input_size(&self) -> usize { self.n_probes }
    fn lnn_output_size(&self) -> usize { self.n_probes }
    fn hd_dim(&self) -> usize { self.hd_dim }
}

// =============================================================================
// Strategy R6: Adaptive DualTrack (LVQ-style probe learning)
// =============================================================================

/// Bridge where probes adapt via LVQ-style winner-take-all learning.
///
/// Like `DualTrackBridge`, the LNN operates on similarity features extracted
/// via probes and produces parameters for HDC operations. However, during
/// each `encode_for_lnn` call the probe most similar to the input is updated
/// to move toward the input. Over time, probes converge to category centroids
/// in HD space, providing better feature extraction without external training.
///
/// Uses `RefCell` for interior mutability so that probe adaptation can happen
/// inside `encode_for_lnn(&self, ...)`.
pub struct AdaptiveDualTrackBridge {
    probes: RefCell<Vec<RealHV>>,
    n_probes: usize,
    hd_dim: usize,
    probe_lr: f32,
}

// SAFETY: AdaptiveDualTrackBridge uses RefCell which is !Sync. However, we
// need Send + Sync for the Bridge trait. SemanticField is used single-threaded,
// so we manually implement Sync. The RefCell is never shared across threads.
unsafe impl Sync for AdaptiveDualTrackBridge {}

impl AdaptiveDualTrackBridge {
    pub fn new(hd_dim: usize, n_probes: usize, probe_lr: f32, rng: &mut impl Rng) -> Self {
        let probes: Vec<RealHV> = (0..n_probes)
            .map(|_| RealHV::random(hd_dim, rng).normalized())
            .collect();
        AdaptiveDualTrackBridge {
            probes: RefCell::new(probes),
            n_probes,
            hd_dim,
            probe_lr,
        }
    }

    /// Update the probe most similar to `input` toward it (LVQ winner-take-all).
    pub fn adapt_probes(&self, input: &RealHV) {
        let mut probes = self.probes.borrow_mut();
        // Find the probe most similar to input
        let mut best_idx = 0;
        let mut best_sim = f32::NEG_INFINITY;
        for (i, probe) in probes.iter().enumerate() {
            let sim = RealHV::cosine_similarity(probe, input);
            if sim > best_sim {
                best_sim = sim;
                best_idx = i;
            }
        }
        // Update winner probe toward input: p' = normalize((1-lr)*p + lr*input)
        let old = &probes[best_idx];
        let moved = RealHV::add(
            &old.scale(1.0 - self.probe_lr),
            &input.scale(self.probe_lr),
        );
        probes[best_idx] = moved.normalized();
    }

    /// Number of trainable probe parameters.
    pub fn probe_param_count(&self) -> usize {
        self.n_probes * self.hd_dim
    }

    /// Extract probe data as flat vector.
    pub fn get_probe_params(&self) -> Vec<f32> {
        let probes = self.probes.borrow();
        let mut params = Vec::with_capacity(self.probe_param_count());
        for probe in probes.iter() {
            params.extend_from_slice(&probe.data);
        }
        params
    }

    /// Set probe data from flat vector.
    pub fn set_probe_params(&self, params: &[f32]) {
        assert_eq!(params.len(), self.probe_param_count());
        let mut probes = self.probes.borrow_mut();
        for (i, probe) in probes.iter_mut().enumerate() {
            let offset = i * self.hd_dim;
            probe.data.copy_from_slice(&params[offset..offset + self.hd_dim]);
            probe.normalize();
        }
    }

    /// Get a snapshot of the current probes (for analysis/testing).
    pub fn get_probes(&self) -> Vec<RealHV> {
        self.probes.borrow().clone()
    }
}

impl Bridge for AdaptiveDualTrackBridge {
    fn encode_for_lnn(&self, state: &RealHV, input: &RealHV) -> Vec<f32> {
        // Adapt probes toward the input (LVQ step)
        // Only adapt if input is non-trivial (not a zero vector)
        if input.norm() > 1e-8 {
            self.adapt_probes(input);
        }

        let combined = RealHV::bundle_normalized(&[state, input]);
        let probes = self.probes.borrow();
        probes
            .iter()
            .map(|probe| RealHV::cosine_similarity(&combined, probe))
            .collect()
    }

    fn decode_from_lnn(&self, lnn_output: &[f32], current_state: &RealHV) -> RealHV {
        let probes = self.probes.borrow();
        // Last value is alpha (mixing parameter)
        let n_weights = self.n_probes - 1;
        let raw_alpha = if n_weights < lnn_output.len() {
            lnn_output[n_weights]
        } else {
            0.1
        };
        let alpha = crate::lnn::neuron::sigmoid(raw_alpha) * 0.5; // Keep alpha in [0, 0.5]

        // Weighted sum of probes
        let mut delta = RealHV::zero(self.hd_dim);
        let weight_count = n_weights.min(lnn_output.len());
        for i in 0..weight_count {
            let scaled = probes[i].scale(lnn_output[i]);
            delta = RealHV::add(&delta, &scaled);
        }
        delta.normalize();

        // Mix: new_state = (1-alpha)*current + alpha*delta
        let kept = current_state.scale(1.0 - alpha);
        let added = delta.scale(alpha);
        let mut result = RealHV::add(&kept, &added);
        result.normalize();
        result
    }

    fn name(&self) -> &str { "AdaptiveDualTrack" }
    fn lnn_input_size(&self) -> usize { self.n_probes }
    fn lnn_output_size(&self) -> usize { self.n_probes }
    fn hd_dim(&self) -> usize { self.hd_dim }
}

// =============================================================================
// Factory function
// =============================================================================

use crate::lnn::ncp::NCPConfig;

/// Creates a bridge for the given strategy and configuration.
pub fn create_bridge(
    strategy: BridgeStrategy,
    hd_dim: usize,
    ncp_config: &NCPConfig,
    rng: &mut impl Rng,
) -> Box<dyn Bridge> {
    match strategy {
        BridgeStrategy::RandomProjection => Box::new(RandomProjectionBridge::new(
            hd_dim,
            ncp_config.sensory_size,
            ncp_config.motor_size,
            rng,
        )),
        BridgeStrategy::DirectHD => Box::new(DirectHDBridge::new(hd_dim)),
        BridgeStrategy::DualTrack => Box::new(DualTrackBridge::new(
            hd_dim,
            ncp_config.sensory_size,
            rng,
        )),
        BridgeStrategy::SparseHD => Box::new(SparseHDBridge::new(
            hd_dim,
            16,  // n_neurons
            2,   // overlap_factor
            rng,
        )),
        BridgeStrategy::InputPreserving => Box::new(InputPreservingBridge::new(
            hd_dim,
            ncp_config.sensory_size,
            0.3,
            rng,
        )),
        BridgeStrategy::SemanticDualTrack => Box::new(SemanticDualTrackBridge::new(
            hd_dim,
            ncp_config.sensory_size,
            rng,
        )),
        BridgeStrategy::AdaptiveDualTrack => Box::new(AdaptiveDualTrackBridge::new(
            hd_dim,
            ncp_config.sensory_size,
            0.005, // default probe learning rate
            rng,
        )),
    }
}
