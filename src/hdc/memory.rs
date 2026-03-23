//! Item Memory: associative memory store for hypervectors.
//!
//! Maps string labels to binary hypervectors and supports nearest-neighbor search.

use rayon::prelude::*;

use super::binary::BinaryHV;
use super::hypervector::HyperVector;

/// Associative memory store mapping labels to binary hypervectors.
///
/// All stored hypervectors must have the same dimensionality.
pub struct ItemMemory {
    items: Vec<(String, BinaryHV)>,
    /// Dimensionality of stored vectors (None if empty).
    dim: Option<usize>,
    /// Threshold for switching to parallel search.
    parallel_threshold: usize,
}

impl ItemMemory {
    /// Creates a new empty item memory.
    pub fn new() -> Self {
        ItemMemory {
            items: Vec::new(),
            dim: None,
            parallel_threshold: 1000,
        }
    }

    /// Sets the threshold for parallel search.
    pub fn with_parallel_threshold(mut self, threshold: usize) -> Self {
        self.parallel_threshold = threshold;
        self
    }

    /// Returns the number of stored items.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Returns true if the memory is empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Returns true if the memory contains a vector with the given label.
    pub fn contains(&self, label: &str) -> bool {
        self.items.iter().any(|(l, _)| l == label)
    }

    /// Inserts or updates a hypervector with the given label.
    ///
    /// Panics if the dimensionality doesn't match existing vectors.
    pub fn insert(&mut self, label: String, hv: BinaryHV) {
        if let Some(d) = self.dim {
            assert_eq!(
                hv.dim, d,
                "Dimension mismatch: inserting {} into memory with dim {}",
                hv.dim, d
            );
        } else {
            self.dim = Some(hv.dim);
        }

        if let Some(pos) = self.items.iter().position(|(l, _)| l == &label) {
            self.items[pos].1 = hv;
        } else {
            self.items.push((label, hv));
        }
    }

    /// Retrieves a hypervector by label.
    pub fn get(&self, label: &str) -> Option<&BinaryHV> {
        self.items.iter().find(|(l, _)| l == label).map(|(_, hv)| hv)
    }

    /// Removes a hypervector by label. Returns true if it was present.
    pub fn remove(&mut self, label: &str) -> bool {
        if let Some(pos) = self.items.iter().position(|(l, _)| l == label) {
            self.items.swap_remove(pos);
            if self.items.is_empty() {
                self.dim = None;
            }
            true
        } else {
            false
        }
    }

    /// Returns the most similar item to the query.
    pub fn nearest(&self, query: &BinaryHV) -> Option<(String, f32)> {
        if self.items.is_empty() {
            return None;
        }

        if self.items.len() >= self.parallel_threshold {
            self.items
                .par_iter()
                .map(|(label, hv)| (label.clone(), hv.similarity(query)))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        } else {
            self.items
                .iter()
                .map(|(label, hv)| (label.clone(), hv.similarity(query)))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        }
    }

    /// Returns the k most similar items to the query, sorted by similarity (descending).
    pub fn nearest_k(&self, query: &BinaryHV, k: usize) -> Vec<(String, f32)> {
        let mut all = self.all_similarities(query);
        all.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        all.truncate(k);
        all
    }

    /// Returns all items with their similarity to the query.
    pub fn all_similarities(&self, query: &BinaryHV) -> Vec<(String, f32)> {
        if self.items.len() >= self.parallel_threshold {
            self.items
                .par_iter()
                .map(|(label, hv)| (label.clone(), hv.similarity(query)))
                .collect()
        } else {
            self.items
                .iter()
                .map(|(label, hv)| (label.clone(), hv.similarity(query)))
                .collect()
        }
    }

    /// Returns all items with cosine similarity to a RealHV query.
    ///
    /// Converts each stored BinaryHV to RealHV for comparison.
    /// This avoids the lossy RealHV→BinaryHV conversion.
    pub fn all_similarities_real(&self, query: &super::real::RealHV) -> Vec<(String, f32)> {
        self.items
            .iter()
            .map(|(label, hv)| {
                let hv_real = hv.to_real();
                (label.clone(), super::real::RealHV::cosine_similarity(query, &hv_real))
            })
            .collect()
    }
}

impl Default for ItemMemory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn run_at_dims(test_fn: impl Fn(usize, &mut StdRng)) {
        for &dim in &[1024, 4096] {
            let mut rng = StdRng::seed_from_u64(42);
            test_fn(dim, &mut rng);
        }
    }

    #[test]
    fn test_insert_retrieve() {
        run_at_dims(|dim, rng| {
            let mut mem = ItemMemory::new();
            let hv = BinaryHV::random(dim, rng);
            mem.insert("test".into(), hv.clone());
            assert_eq!(mem.get("test"), Some(&hv));
        });
    }

    #[test]
    fn test_nearest_exact() {
        run_at_dims(|dim, rng| {
            let mut mem = ItemMemory::new();
            let a = BinaryHV::random(dim, rng);
            mem.insert("a".into(), a.clone());
            // Add some distractors
            for i in 0..10 {
                mem.insert(format!("d{i}"), BinaryHV::random(dim, rng));
            }
            let (label, sim) = mem.nearest(&a).unwrap();
            assert_eq!(label, "a");
            assert!((sim - 1.0).abs() < 1e-6, "sim = {sim}");
        });
    }

    #[test]
    fn test_nearest_noisy() {
        run_at_dims(|dim, rng| {
            let mut mem = ItemMemory::new();
            let a = BinaryHV::random(dim, rng);
            mem.insert("a".into(), a.clone());
            for i in 0..20 {
                mem.insert(format!("d{i}"), BinaryHV::random(dim, rng));
            }

            // Create noisy version: flip ~10% of bits
            let mut noisy = a.clone();
            for word in noisy.data.iter_mut() {
                for bit in 0..64 {
                    if rng.gen::<f32>() < 0.1 {
                        *word ^= 1u64 << bit;
                    }
                }
            }
            // Clean padding
            if dim % 64 != 0 {
                let last_mask = (1u64 << (dim % 64)) - 1;
                if let Some(last) = noisy.data.last_mut() {
                    *last &= last_mask;
                }
            }

            let (label, _) = mem.nearest(&noisy).unwrap();
            assert_eq!(label, "a", "D={dim}: nearest of noisy A should be A");
        });
    }

    #[test]
    fn test_nearest_k() {
        run_at_dims(|dim, rng| {
            let mut mem = ItemMemory::new();
            let mut targets = Vec::new();
            for i in 0..100 {
                let hv = BinaryHV::random(dim, rng);
                mem.insert(format!("v{i}"), hv.clone());
                if i < 3 {
                    targets.push(hv);
                }
            }
            let refs: Vec<&BinaryHV> = targets.iter().collect();
            let bundle = BinaryHV::bundle(&refs, rng);
            let top3 = mem.nearest_k(&bundle, 3);
            let top_labels: Vec<&str> = top3.iter().map(|(l, _)| l.as_str()).collect();
            for i in 0..3 {
                assert!(
                    top_labels.contains(&format!("v{i}").as_str()),
                    "D={dim}: v{i} not in top-3: {:?}",
                    top_labels
                );
            }
        });
    }

    #[test]
    fn test_capacity() {
        let dim = 4096;
        let mut rng = StdRng::seed_from_u64(42);
        let mut mem = ItemMemory::new().with_parallel_threshold(500);
        for i in 0..10000 {
            mem.insert(format!("v{i}"), BinaryHV::random(dim, &mut rng));
        }
        let query = BinaryHV::random(dim, &mut rng);
        // Run 3 times, take best (avoids flaky failures under system load)
        let mut best_ms = u128::MAX;
        for _ in 0..3 {
            let start = std::time::Instant::now();
            let _ = mem.nearest(&query);
            let ms = start.elapsed().as_millis();
            if ms < best_ms { best_ms = ms; }
        }
        let limit_ms: u128 = if cfg!(debug_assertions) { 500 } else { 10 };
        assert!(
            best_ms < limit_ms,
            "nearest on 10000 items took {}ms (best of 3)",
            best_ms
        );
    }

    #[test]
    #[should_panic(expected = "Dimension mismatch")]
    fn test_dimension_consistency() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut mem = ItemMemory::new();
        mem.insert("a".into(), BinaryHV::random(1024, &mut rng));
        mem.insert("b".into(), BinaryHV::random(4096, &mut rng));
    }
}
