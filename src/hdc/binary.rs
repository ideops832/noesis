//! Binary hypervectors represented as `Vec<u64>` with runtime dimensionality.
//!
//! Each bit represents one dimension. The storage uses `Vec<u64>` where
//! the number of words is `ceil(dim / 64)`. Padding bits in the last word
//! are always kept at 0.

use std::fmt;
use std::ops::BitXor;

use rand::Rng;
use serde::{Deserialize, Serialize};

use super::hypervector::HyperVector;

/// A binary hypervector with runtime dimensionality.
///
/// Internally stored as a `Vec<u64>` where each bit is one dimension.
/// The effective dimensionality is `dim`, and padding bits in the last
/// word are always 0.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BinaryHV {
    /// The raw bit data, stored as 64-bit words.
    pub data: Vec<u64>,
    /// The logical dimensionality (number of meaningful bits).
    pub dim: usize,
}

impl BinaryHV {
    /// Number of u64 words needed for the given dimensionality.
    #[inline]
    fn n_words(dim: usize) -> usize {
        dim.div_ceil(64)
    }

    /// Bitmask for the last word to zero out padding bits.
    #[inline]
    fn last_word_mask(dim: usize) -> u64 {
        let remainder = dim % 64;
        if remainder == 0 {
            u64::MAX
        } else {
            (1u64 << remainder) - 1
        }
    }

    /// Ensures padding bits are zeroed in the last word.
    #[inline]
    fn clean_padding(&mut self) {
        if let Some(last) = self.data.last_mut() {
            *last &= Self::last_word_mask(self.dim);
        }
    }

    /// Generates a random binary hypervector with uniform 50/50 distribution.
    pub fn random(dim: usize, rng: &mut impl Rng) -> Self {
        let n = Self::n_words(dim);
        let data: Vec<u64> = (0..n).map(|_| rng.gen::<u64>()).collect();
        let mut hv = BinaryHV { data, dim };
        hv.clean_padding();
        hv
    }

    /// Creates a zero hypervector (all bits 0).
    pub fn zero(dim: usize) -> Self {
        let n = Self::n_words(dim);
        BinaryHV {
            data: vec![0u64; n],
            dim,
        }
    }

    /// Creates a ones hypervector (all meaningful bits set to 1).
    pub fn ones(dim: usize) -> Self {
        let n = Self::n_words(dim);
        let mut data = vec![u64::MAX; n];
        // Zero out padding bits in the last word
        if let Some(last) = data.last_mut() {
            *last &= Self::last_word_mask(dim);
        }
        BinaryHV { data, dim }
    }

    /// Binding operation: XOR of two hypervectors.
    ///
    /// Properties:
    /// - `bind(a, a) = zero`
    /// - `bind(a, b) = bind(b, a)` (commutative)
    /// - `bind(bind(a, b), a) = b` (self-inverse)
    #[inline]
    pub fn bind(a: &Self, b: &Self) -> Self {
        assert_eq!(
            a.dim, b.dim,
            "Dimension mismatch: {} vs {}",
            a.dim, b.dim
        );
        let data: Vec<u64> = a.data.iter().zip(&b.data).map(|(x, y)| x ^ y).collect();
        BinaryHV { data, dim: a.dim }
    }

    /// Bundle operation: majority vote across multiple hypervectors.
    ///
    /// For each bit position, counts how many vectors have a 1;
    /// if majority → 1, otherwise → 0. For even counts with a tie,
    /// uses random tie-breaking.
    pub fn bundle(vecs: &[&Self], rng: &mut impl Rng) -> Self {
        assert!(!vecs.is_empty(), "Cannot bundle empty slice");
        let dim = vecs[0].dim;
        for v in vecs.iter().skip(1) {
            assert_eq!(
                v.dim, dim,
                "Dimension mismatch: {} vs {}",
                v.dim, dim
            );
        }

        let n_words = Self::n_words(dim);
        let n_vecs = vecs.len();
        let threshold = n_vecs / 2;
        let is_even = n_vecs.is_multiple_of(2);

        // Word-level counting: for each word, count set bits per position
        let mut result = vec![0u64; n_words];

        for (word_idx, result_word) in result.iter_mut().enumerate() {
            // Count bits set per position across all vectors
            let mut counts = [0u16; 64];
            for v in vecs.iter() {
                let w = v.data[word_idx];
                if w == 0 {
                    continue;
                }
                for (bit, count) in counts.iter_mut().enumerate() {
                    if w & (1u64 << bit) != 0 {
                        *count += 1;
                    }
                }
            }

            let mut word = 0u64;
            for (bit, count) in counts.iter().enumerate() {
                let c = *count as usize;
                let set = if is_even && c == threshold {
                    rng.gen::<bool>()
                } else {
                    c > threshold
                };
                if set {
                    word |= 1u64 << bit;
                }
            }
            *result_word = word;
        }

        let mut hv = BinaryHV { data: result, dim };
        hv.clean_padding();
        hv
    }

    /// Cyclic left shift of the entire bit sequence by `n` positions.
    ///
    /// Property: `permute(v, 1)` is dissimilar from `v`.
    /// Property: `permute(permute(v, a), b) = permute(v, a + b)` (mod dim).
    pub fn permute(&self, n: usize) -> Self {
        if self.dim == 0 {
            return self.clone();
        }
        let shift = n % self.dim;
        if shift == 0 {
            return self.clone();
        }

        // Collect all bits, shift cyclically, then repack.
        // Uses word-level operations: shift by word_offset words,
        // then shift within words by bit_offset.
        let n_words = Self::n_words(self.dim);

        // We treat the bit stream as a flat array of `dim` bits.
        // Left cyclic shift by `shift` means: new[i] = old[(i + shift) % dim].
        // Equivalently: the bit at position `j` in old goes to position `(j - shift + dim) % dim` in new.

        // For efficiency, work word by word: extract 64 consecutive bits starting
        // at offset `shift` (wrapping around), place them into result word.
        let mut result = vec![0u64; n_words];

        for dst_bit in (0..self.dim).step_by(64) {
            let bits_in_this_word = std::cmp::min(64, self.dim - dst_bit);
            let src_start = (dst_bit + shift) % self.dim;

            let mut word = 0u64;
            for b in 0..bits_in_this_word {
                let src_pos = (src_start + b) % self.dim;
                let sw = src_pos / 64;
                let sb = src_pos % 64;
                if self.data[sw] & (1u64 << sb) != 0 {
                    word |= 1u64 << b;
                }
            }
            result[dst_bit / 64] = word;
        }

        BinaryHV { data: result, dim: self.dim }
    }

    /// Inverse operation for binary hypervectors.
    ///
    /// For binary XOR-based binding, every vector is its own inverse:
    /// `bind(a, inverse(a)) = bind(a, a) = zero`
    #[inline]
    pub fn inverse(&self) -> Self {
        self.clone()
    }

    /// Counts the number of bits set to 1 among the first `dim` bits.
    #[inline]
    pub fn popcount(&self) -> u32 {
        if self.data.is_empty() {
            return 0;
        }
        let full_words = if self.dim.is_multiple_of(64) {
            self.data.len()
        } else {
            self.data.len() - 1
        };

        let mut count: u32 = self.data[..full_words]
            .iter()
            .map(|w| w.count_ones())
            .sum();

        // Last word: already cleaned of padding, so count_ones is correct
        if !self.dim.is_multiple_of(64) {
            count += self.data[full_words].count_ones();
        }

        count
    }

    /// Hamming distance between two binary hypervectors.
    #[inline]
    pub fn hamming_distance(a: &Self, b: &Self) -> u32 {
        assert_eq!(
            a.dim, b.dim,
            "Dimension mismatch: {} vs {}",
            a.dim, b.dim
        );
        a.data
            .iter()
            .zip(&b.data)
            .map(|(x, y)| (x ^ y).count_ones())
            .sum()
    }

    /// Converts this binary hypervector to a real-valued one.
    ///
    /// Mapping: 0 → +1.0, 1 → -1.0 (i.e., (-1)^bit).
    /// This ensures that XOR in binary corresponds to element-wise multiply
    /// in real space: (-1)^(a XOR b) = (-1)^a * (-1)^b.
    pub fn to_real(&self) -> super::real::RealHV {
        let mut data = Vec::with_capacity(self.dim);
        for i in 0..self.dim {
            let word_idx = i / 64;
            let bit_pos = i % 64;
            if self.data[word_idx] & (1u64 << bit_pos) != 0 {
                data.push(-1.0f32);
            } else {
                data.push(1.0f32);
            }
        }
        super::real::RealHV { data, dim: self.dim }
    }
}

impl HyperVector for BinaryHV {
    #[inline]
    fn dim(&self) -> usize {
        self.dim
    }

    /// Normalized Hamming similarity.
    ///
    /// Returns a value in [-1, 1] where:
    /// - 1.0 = identical
    /// - 0.0 = orthogonal (50% bits differ)
    /// - -1.0 = complementary (all bits differ)
    #[inline]
    fn similarity(&self, other: &Self) -> f32 {
        assert_eq!(
            self.dim, other.dim,
            "Dimension mismatch: {} vs {}",
            self.dim, other.dim
        );
        let hd = Self::hamming_distance(self, other) as f32;
        1.0 - 2.0 * hd / self.dim as f32
    }
}

impl BitXor for &BinaryHV {
    type Output = BinaryHV;

    #[inline]
    fn bitxor(self, rhs: Self) -> BinaryHV {
        BinaryHV::bind(self, rhs)
    }
}

impl fmt::Display for BinaryHV {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BinaryHV(dim={}", self.dim)?;
        if !self.data.is_empty() {
            write!(f, ", {:064b}...", self.data[0])?;
        }
        write!(f, ")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn run_at_dims(test_fn: impl Fn(usize, &mut StdRng)) {
        for &dim in &[1024, 4096, 10048] {
            let mut rng = StdRng::seed_from_u64(42);
            test_fn(dim, &mut rng);
        }
    }

    #[test]
    fn test_random_orthogonality() {
        run_at_dims(|dim, rng| {
            let n = 1000;
            let vecs: Vec<BinaryHV> = (0..n).map(|_| BinaryHV::random(dim, rng)).collect();
            let mut total_sim = 0.0f64;
            let mut count = 0u64;
            for i in 0..100 {
                for j in (i + 1)..100 {
                    total_sim += vecs[i].similarity(&vecs[j]) as f64;
                    count += 1;
                }
            }
            let mean_sim = total_sim / count as f64;
            assert!(
                mean_sim.abs() < 0.05,
                "D={dim}: mean similarity = {mean_sim}, expected ~0.0"
            );
        });
    }

    #[test]
    fn test_bind_dissimilarity() {
        run_at_dims(|dim, rng| {
            let a = BinaryHV::random(dim, rng);
            let b = BinaryHV::random(dim, rng);
            let c = BinaryHV::bind(&a, &b);
            let sim_a = c.similarity(&a);
            let sim_b = c.similarity(&b);
            assert!(
                sim_a.abs() < 0.1,
                "D={dim}: bind(a,b) sim to a = {sim_a}"
            );
            assert!(
                sim_b.abs() < 0.1,
                "D={dim}: bind(a,b) sim to b = {sim_b}"
            );
        });
    }

    #[test]
    fn test_bind_inverse() {
        run_at_dims(|dim, rng| {
            let a = BinaryHV::random(dim, rng);
            let b = BinaryHV::random(dim, rng);
            let c = BinaryHV::bind(&a, &b);
            let recovered = BinaryHV::bind(&c, &a);
            let sim = recovered.similarity(&b);
            assert!(
                (sim - 1.0).abs() < 1e-6,
                "D={dim}: recovered sim to b = {sim}, expected 1.0"
            );
        });
    }

    #[test]
    fn test_bundle_similarity() {
        run_at_dims(|dim, rng| {
            let a = BinaryHV::random(dim, rng);
            let b = BinaryHV::random(dim, rng);
            let c = BinaryHV::random(dim, rng);
            let bundle = BinaryHV::bundle(&[&a, &b, &c], rng);
            assert!(
                bundle.similarity(&a) > 0.3,
                "D={dim}: bundle sim to a = {}",
                bundle.similarity(&a)
            );
            assert!(
                bundle.similarity(&b) > 0.3,
                "D={dim}: bundle sim to b = {}",
                bundle.similarity(&b)
            );
            assert!(
                bundle.similarity(&c) > 0.3,
                "D={dim}: bundle sim to c = {}",
                bundle.similarity(&c)
            );
        });
    }

    #[test]
    fn test_permute_dissimilarity() {
        run_at_dims(|dim, rng| {
            let v = BinaryHV::random(dim, rng);
            let p = v.permute(1);
            let sim = v.similarity(&p);
            assert!(
                sim.abs() < 0.1,
                "D={dim}: permute(v,1) sim to v = {sim}"
            );
        });
    }

    #[test]
    fn test_permute_inverse() {
        run_at_dims(|dim, rng| {
            let v = BinaryHV::random(dim, rng);
            let p = v.permute(1).permute(dim - 1);
            assert_eq!(v, p, "D={dim}: permute(permute(v,1), dim-1) != v");
        });
    }

    #[test]
    fn test_bind_commutativity() {
        run_at_dims(|dim, rng| {
            let a = BinaryHV::random(dim, rng);
            let b = BinaryHV::random(dim, rng);
            assert_eq!(
                BinaryHV::bind(&a, &b),
                BinaryHV::bind(&b, &a),
                "D={dim}: bind not commutative"
            );
        });
    }

    #[test]
    fn test_bind_associativity() {
        run_at_dims(|dim, rng| {
            let a = BinaryHV::random(dim, rng);
            let b = BinaryHV::random(dim, rng);
            let c = BinaryHV::random(dim, rng);
            let ab_c = BinaryHV::bind(&BinaryHV::bind(&a, &b), &c);
            let a_bc = BinaryHV::bind(&a, &BinaryHV::bind(&b, &c));
            assert_eq!(ab_c, a_bc, "D={dim}: bind not associative");
        });
    }

    #[test]
    fn test_high_dimensionality() {
        let dim = 10048;
        let mut rng = StdRng::seed_from_u64(42);
        let n = 10000;
        let vecs: Vec<BinaryHV> = (0..n).map(|_| BinaryHV::random(dim, &mut rng)).collect();
        // Check max similarity among a random sample of pairs
        let mut max_sim: f32 = 0.0;
        for i in 0..500 {
            for j in (i + 1)..500 {
                let sim = vecs[i].similarity(&vecs[j]).abs();
                if sim > max_sim {
                    max_sim = sim;
                }
            }
        }
        assert!(
            max_sim < 0.1,
            "D={dim}: max similarity among 500 pairs = {max_sim}, expected < 0.1"
        );
    }

    #[test]
    #[should_panic(expected = "Dimension mismatch")]
    fn test_dimension_mismatch_bind() {
        let mut rng = StdRng::seed_from_u64(42);
        let a = BinaryHV::random(1024, &mut rng);
        let b = BinaryHV::random(4096, &mut rng);
        BinaryHV::bind(&a, &b);
    }

    #[test]
    #[should_panic(expected = "Dimension mismatch")]
    fn test_dimension_mismatch_similarity() {
        let mut rng = StdRng::seed_from_u64(42);
        let a = BinaryHV::random(1024, &mut rng);
        let b = BinaryHV::random(4096, &mut rng);
        a.similarity(&b);
    }
}
