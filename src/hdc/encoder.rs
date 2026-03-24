//! Compositional encoding of structures into hypervectors.
//!
//! The encoder transforms composite structures (sequences, records, sets)
//! into hypervectors using HDC operations: bind, bundle, and permute.

use rand::rngs::StdRng;
use rand::SeedableRng;
use thiserror::Error;

use super::binary::BinaryHV;
use super::memory::ItemMemory;

/// Errors that can occur during encoding.
#[derive(Debug, Error)]
pub enum EncoderError {
    /// Atom not found in vocabulary.
    #[error("Atom not registered: {0}")]
    AtomNotFound(String),
    /// Role index out of bounds.
    #[error("Role index {0} exceeds maximum {1}")]
    RoleOutOfBounds(usize, usize),
}

/// Encodes composite structures into hypervectors using HDC algebra.
pub struct Encoder {
    /// Dimensionality of all hypervectors.
    dim: usize,
    /// Vocabulary of atomic symbols.
    vocab: ItemMemory,
    /// Pre-generated role hypervectors (ROLE_0, ROLE_1, ...).
    roles: ItemMemory,
    /// RNG for reproducibility.
    rng: StdRng,
    /// Number of pre-generated roles.
    _n_roles: usize,
}

impl Encoder {
    /// Creates a new encoder with empty vocabulary and pre-generated roles.
    pub fn new(dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let n_roles = 32;
        let mut roles = ItemMemory::new();
        for i in 0..n_roles {
            roles.insert(format!("ROLE_{i}"), BinaryHV::random(dim, &mut rng));
        }

        Encoder {
            dim,
            vocab: ItemMemory::new(),
            roles,
            rng,
            _n_roles: n_roles,
        }
    }

    /// Registers an atom with a random hypervector. Returns existing if already registered.
    pub fn register(&mut self, label: &str) -> &BinaryHV {
        if !self.vocab.contains(label) {
            let hv = BinaryHV::random(self.dim, &mut self.rng);
            self.vocab.insert(label.to_string(), hv);
        }
        self.vocab.get(label).unwrap()
    }

    /// Returns the hypervector for a registered atom.
    pub fn encode_atom(&self, label: &str) -> Result<BinaryHV, EncoderError> {
        self.vocab
            .get(label)
            .cloned()
            .ok_or_else(|| EncoderError::AtomNotFound(label.to_string()))
    }

    /// Encodes an ordered sequence using permutation to capture order.
    ///
    /// `seq = bundle([permute(encode(label_i), i) for i in 0..n])`
    pub fn encode_sequence(&mut self, labels: &[&str]) -> BinaryHV {
        let permuted: Vec<BinaryHV> = labels
            .iter()
            .enumerate()
            .map(|(i, label)| {
                let hv = self.register(label).clone();
                hv.permute(i)
            })
            .collect();

        let refs: Vec<&BinaryHV> = permuted.iter().collect();
        BinaryHV::bundle(&refs, &mut self.rng)
    }

    /// Encodes a record (set of role-value pairs) using binding.
    ///
    /// `rec = bundle([bind(role_i, value_i) for each pair])`
    pub fn encode_record(&mut self, pairs: &[(&str, &str)]) -> BinaryHV {
        let bound: Vec<BinaryHV> = pairs
            .iter()
            .enumerate()
            .map(|(i, (role, value))| {
                let role_hv = if let Some(r) = self.roles.get(role) {
                    r.clone()
                } else {
                    self.roles
                        .get(&format!("ROLE_{i}"))
                        .expect("role index out of bounds")
                        .clone()
                };
                let value_hv = self.register(value).clone();
                BinaryHV::bind(&role_hv, &value_hv)
            })
            .collect();

        let refs: Vec<&BinaryHV> = bound.iter().collect();
        BinaryHV::bundle(&refs, &mut self.rng)
    }

    /// Encodes a record using named roles from the vocabulary.
    ///
    /// Both role and value labels are looked up (or created) in the **vocab**,
    /// ensuring that `encode_atom(role)` returns the same vector used here.
    /// This guarantees consistent encoding/decoding.
    pub fn encode_record_named(&mut self, pairs: &[(&str, &str)]) -> BinaryHV {
        let bound: Vec<BinaryHV> = pairs
            .iter()
            .map(|(role, value)| {
                // Use vocab for both role and value — same pool as encode_atom
                let role_hv = self.register(role).clone();
                let value_hv = self.register(value).clone();
                BinaryHV::bind(&role_hv, &value_hv)
            })
            .collect();

        let refs: Vec<&BinaryHV> = bound.iter().collect();
        BinaryHV::bundle(&refs, &mut self.rng)
    }

    /// Encodes an unordered set.
    ///
    /// `set = bundle([encode(label_i) for each label])`
    pub fn encode_set(&mut self, labels: &[&str]) -> BinaryHV {
        let hvs: Vec<BinaryHV> = labels
            .iter()
            .map(|label| self.register(label).clone())
            .collect();

        let refs: Vec<&BinaryHV> = hvs.iter().collect();
        BinaryHV::bundle(&refs, &mut self.rng)
    }

    /// Decodes a role from a record by unbinding, then searching the vocabulary.
    ///
    /// Returns the top-k vocabulary entries most similar to the unbound result.
    pub fn decode_role(&self, record_hv: &BinaryHV, role_label: &str) -> Vec<(String, f32)> {
        // Use vocab (same pool as encode_record_named) for consistent encode/decode
        let role_hv = self
            .vocab
            .get(role_label)
            .expect("Role not found in vocab");
        let unbound = BinaryHV::bind(record_hv, role_hv);
        self.vocab.nearest_k(&unbound, 5)
    }

    /// Returns top-k vocabulary entries most similar to the query.
    pub fn similarity_query(&self, hv: &BinaryHV) -> Vec<(String, f32)> {
        self.vocab.nearest_k(hv, 10)
    }

    /// Decodes a role from a RealHV record using real-space search.
    ///
    /// This avoids the lossy RealHV→BinaryHV conversion and searches
    /// directly in real-valued space using cosine similarity.
    pub fn decode_role_real(&self, record_hv: &super::real::RealHV, role_label: &str, k: usize) -> Vec<(String, f32)> {
        let role_bin = self.vocab.get(role_label).expect("Role not found in vocab");
        let role_real = role_bin.to_real();
        let unbound = super::real::RealHV::bind(record_hv, &role_real.inverse());

        // Search in real space against all vocab entries
        let mut sims: Vec<(String, f32)> = self.vocab.all_similarities_real(&unbound);
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sims.truncate(k);
        sims
    }

    /// Provides read access to the vocabulary.
    pub fn vocab(&self) -> &ItemMemory {
        &self.vocab
    }

    /// Provides mutable access to the vocabulary.
    pub fn vocab_mut(&mut self) -> &mut ItemMemory {
        &mut self.vocab
    }

    /// Provides read access to the roles.
    pub fn roles(&self) -> &ItemMemory {
        &self.roles
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::hypervector::HyperVector;

    fn run_at_dims(test_fn: impl Fn(usize)) {
        for &dim in &[1024, 4096] {
            test_fn(dim);
        }
    }

    #[test]
    fn test_sequence_order() {
        run_at_dims(|dim| {
            let mut enc = Encoder::new(dim, 42);
            // Register atoms first so both sequences use the same vectors
            enc.register("a");
            enc.register("b");
            enc.register("c");
            let s1 = enc.encode_sequence(&["a", "b", "c"]);
            let s2 = enc.encode_sequence(&["c", "b", "a"]);
            let sim = s1.similarity(&s2);
            assert!(
                sim < 0.5,
                "D={dim}: same-elements different-order similarity = {sim}, expected < 0.5"
            );
        });
    }

    #[test]
    fn test_record_decode() {
        run_at_dims(|dim| {
            let mut enc = Encoder::new(dim, 42);
            let record = enc.encode_record_named(&[
                ("soggetto", "gatto"),
                ("azione", "mangia"),
                ("oggetto", "topo"),
            ]);
            let decoded = enc.decode_role(&record, "soggetto");
            assert!(
                !decoded.is_empty(),
                "D={dim}: decode returned empty"
            );
            assert_eq!(
                decoded[0].0, "gatto",
                "D={dim}: expected 'gatto' as top-1, got '{}'",
                decoded[0].0
            );

            let decoded_action = enc.decode_role(&record, "azione");
            assert_eq!(
                decoded_action[0].0, "mangia",
                "D={dim}: expected 'mangia' as top-1, got '{}'",
                decoded_action[0].0
            );

            let decoded_obj = enc.decode_role(&record, "oggetto");
            assert_eq!(
                decoded_obj[0].0, "topo",
                "D={dim}: expected 'topo' as top-1, got '{}'",
                decoded_obj[0].0
            );
        });
    }

    #[test]
    fn test_set_membership() {
        run_at_dims(|dim| {
            let mut enc = Encoder::new(dim, 42);
            enc.register("d");
            let set = enc.encode_set(&["a", "b", "c"]);
            let a = enc.encode_atom("a").unwrap();
            let _b = enc.encode_atom("b").unwrap();
            let _c = enc.encode_atom("c").unwrap();
            let d = enc.encode_atom("d").unwrap();

            let sim_a = set.similarity(&a);
            let sim_d = set.similarity(&d);

            assert!(
                sim_a > 0.2,
                "D={dim}: set similarity to member 'a' = {sim_a}"
            );
            assert!(
                sim_d.abs() < sim_a,
                "D={dim}: non-member 'd' ({sim_d}) should be less similar than member 'a' ({sim_a})"
            );
        });
    }

    #[test]
    fn test_analogy_mechanical() {
        run_at_dims(|dim| {
            let mut enc = Encoder::new(dim, 42);
            enc.register("re");
            enc.register("italia");
            enc.register("francia");
            enc.register("presidente");

            let re = enc.encode_atom("re").unwrap();
            let italia = enc.encode_atom("italia").unwrap();
            let francia = enc.encode_atom("francia").unwrap();
            let presidente = enc.encode_atom("presidente").unwrap();

            // Mechanical analogy: bind(bind(re, inverse(italia)), francia)
            // With random vectors this should NOT produce meaningful results
            let analogy = BinaryHV::bind(&BinaryHV::bind(&re, &italia.inverse()), &francia);

            assert_eq!(analogy.dim, dim, "D={dim}: analogy dim mismatch");

            // Similarity with presidente should be ~0.0 (random)
            let sim = analogy.similarity(&presidente);
            assert!(
                sim.abs() < 0.15,
                "D={dim}: random analogy sim to 'presidente' = {sim}, expected ~0.0"
            );
        });
    }

    #[test]
    fn test_composition_depth() {
        run_at_dims(|dim| {
            let mut enc = Encoder::new(dim, 42);

            // Level 1: bind(role_a, val_x)
            let inner = enc.encode_record_named(&[("inner_role", "inner_val")]);

            // Register the inner record as an atom by inserting it
            enc.vocab_mut()
                .insert("inner_record".to_string(), inner.clone());

            // Level 2: bind(outer_role, inner_record)
            let outer = enc.encode_record_named(&[
                ("outer_role", "inner_record"),
                ("other", "something"),
            ]);

            // Decode outer_role → should recover inner_record
            let decoded = enc.decode_role(&outer, "outer_role");
            assert!(
                !decoded.is_empty(),
                "D={dim}: outer decode returned empty"
            );
            assert_eq!(
                decoded[0].0, "inner_record",
                "D={dim}: expected 'inner_record' as top-1"
            );
        });
    }
}
