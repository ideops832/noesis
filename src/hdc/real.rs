//! Real-valued hypervectors represented as `Vec<f32>` with runtime dimensionality.
//!
//! Operations are component-wise and O(dim).

use std::ops::{Add, Mul};

use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use super::hypervector::HyperVector;

/// A real-valued hypervector with runtime dimensionality.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RealHV {
    /// The component data.
    pub data: Vec<f32>,
    /// The logical dimensionality (== data.len()).
    pub dim: usize,
}

impl RealHV {
    /// Generates a random bipolar hypervector with components in {-1.0, +1.0}.
    pub fn random(dim: usize, rng: &mut impl Rng) -> Self {
        let data: Vec<f32> = (0..dim)
            .map(|_| if rng.gen::<bool>() { 1.0 } else { -1.0 })
            .collect();
        RealHV { data, dim }
    }

    /// Generates a random hypervector with components from N(0, 1/sqrt(dim)).
    pub fn random_normal(dim: usize, rng: &mut impl Rng) -> Self {
        let std_dev = 1.0 / (dim as f32).sqrt();
        let normal = Normal::new(0.0f32, std_dev).unwrap();
        let data: Vec<f32> = (0..dim).map(|_| normal.sample(rng)).collect();
        RealHV { data, dim }
    }

    /// Binding: element-wise multiplication.
    #[inline]
    pub fn bind(a: &Self, b: &Self) -> Self {
        assert_eq!(a.dim, b.dim, "Dimension mismatch: {} vs {}", a.dim, b.dim);
        let data: Vec<f32> = a.data.iter().zip(&b.data).map(|(x, y)| x * y).collect();
        RealHV { data, dim: a.dim }
    }

    /// Bundle: element-wise sum (not normalized).
    pub fn bundle(vecs: &[&Self]) -> Self {
        assert!(!vecs.is_empty(), "Cannot bundle empty slice");
        let dim = vecs[0].dim;
        for v in vecs.iter().skip(1) {
            assert_eq!(v.dim, dim, "Dimension mismatch: {} vs {}", v.dim, dim);
        }
        let mut data = vec![0.0f32; dim];
        for v in vecs {
            for (d, s) in data.iter_mut().zip(&v.data) {
                *d += s;
            }
        }
        RealHV { data, dim }
    }

    /// Bundle with normalization to unit norm.
    pub fn bundle_normalized(vecs: &[&Self]) -> Self {
        let mut result = Self::bundle(vecs);
        let n = result.norm();
        if n > 1e-10 {
            for d in &mut result.data {
                *d /= n;
            }
        }
        result
    }

    /// Cyclic left shift by n positions.
    pub fn permute(&self, n: usize) -> Self {
        if self.dim == 0 {
            return self.clone();
        }
        let shift = n % self.dim;
        if shift == 0 {
            return self.clone();
        }
        let mut data = Vec::with_capacity(self.dim);
        data.extend_from_slice(&self.data[shift..]);
        data.extend_from_slice(&self.data[..shift]);
        RealHV { data, dim: self.dim }
    }

    /// Inverse for element-wise multiplication binding.
    ///
    /// For bipolar {-1, +1} vectors, inverse == self.
    /// For general f32: inverse[i] = 1.0 / v[i], with near-zero protection.
    pub fn inverse(&self) -> Self {
        const EPSILON: f32 = 1e-7;
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| if x.abs() < EPSILON { 0.0 } else { 1.0 / x })
            .collect();
        RealHV { data, dim: self.dim }
    }

    /// Cosine similarity.
    #[inline]
    pub fn cosine_similarity(a: &Self, b: &Self) -> f32 {
        assert_eq!(a.dim, b.dim, "Dimension mismatch: {} vs {}", a.dim, b.dim);
        let dot: f32 = a.data.iter().zip(&b.data).map(|(x, y)| x * y).sum();
        let na = a.norm();
        let nb = b.norm();
        if na < 1e-10 || nb < 1e-10 {
            return 0.0;
        }
        dot / (na * nb)
    }

    /// Normalizes to unit L2 norm (in place).
    pub fn normalize(&mut self) {
        let n = self.norm();
        if n > 1e-10 {
            for d in &mut self.data {
                *d /= n;
            }
        }
    }

    /// Returns a normalized copy.
    pub fn normalized(&self) -> Self {
        let mut result = self.clone();
        result.normalize();
        result
    }

    /// Scalar multiplication.
    pub fn scale(&self, s: f32) -> Self {
        let data: Vec<f32> = self.data.iter().map(|x| x * s).collect();
        RealHV { data, dim: self.dim }
    }

    /// Element-wise addition.
    pub fn add(a: &Self, b: &Self) -> Self {
        assert_eq!(a.dim, b.dim, "Dimension mismatch: {} vs {}", a.dim, b.dim);
        let data: Vec<f32> = a.data.iter().zip(&b.data).map(|(x, y)| x + y).collect();
        RealHV { data, dim: a.dim }
    }

    /// Dot product.
    #[inline]
    pub fn dot(a: &Self, b: &Self) -> f32 {
        assert_eq!(a.dim, b.dim, "Dimension mismatch: {} vs {}", a.dim, b.dim);
        a.data.iter().zip(&b.data).map(|(x, y)| x * y).sum()
    }

    /// L2 norm.
    #[inline]
    pub fn norm(&self) -> f32 {
        self.data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Zero vector.
    pub fn zero(dim: usize) -> Self {
        RealHV {
            data: vec![0.0; dim],
            dim,
        }
    }

    /// Creates a RealHV from a raw data slice.
    pub fn from_slice(data: &[f32]) -> Self {
        RealHV {
            data: data.to_vec(),
            dim: data.len(),
        }
    }

    /// Converts to a BinaryHV: positive → 1, non-positive → 0.
    pub fn to_binary(&self) -> super::binary::BinaryHV {
        let n_words = self.dim.div_ceil(64);
        let mut data = vec![0u64; n_words];
        for (i, &val) in self.data.iter().enumerate() {
            if val > 0.0 {
                data[i / 64] |= 1u64 << (i % 64);
            }
        }
        super::binary::BinaryHV {
            data,
            dim: self.dim,
        }
    }
}

impl HyperVector for RealHV {
    fn dim(&self) -> usize {
        self.dim
    }

    fn similarity(&self, other: &Self) -> f32 {
        Self::cosine_similarity(self, other)
    }
}

impl Add for &RealHV {
    type Output = RealHV;
    fn add(self, rhs: Self) -> RealHV {
        RealHV::add(self, rhs)
    }
}

impl Mul for &RealHV {
    type Output = RealHV;
    fn mul(self, rhs: Self) -> RealHV {
        RealHV::bind(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn run_at_dims(test_fn: impl Fn(usize, &mut StdRng)) {
        for &dim in &[1024, 4096] {
            let mut rng = StdRng::seed_from_u64(42);
            test_fn(dim, &mut rng);
        }
    }

    #[test]
    fn test_real_hv_bind() {
        run_at_dims(|dim, rng| {
            let a = RealHV::random(dim, rng);
            let b = RealHV::random(dim, rng);
            let c = RealHV::bind(&a, &b);
            // Bipolar * bipolar = bipolar
            assert!(
                c.data.iter().all(|&x| (x - 1.0).abs() < 1e-6 || (x + 1.0).abs() < 1e-6),
                "D={dim}: bind of bipolar vectors should be bipolar"
            );
        });
    }

    #[test]
    fn test_real_hv_bind_inverse() {
        run_at_dims(|dim, rng| {
            let a = RealHV::random(dim, rng);
            let b = RealHV::random(dim, rng);
            let ab = RealHV::bind(&a, &b);
            let recovered = RealHV::bind(&ab, &a.inverse());
            let sim = RealHV::cosine_similarity(&recovered, &b);
            assert!(
                sim > 0.99,
                "D={dim}: bind inverse recovery sim = {sim}"
            );
        });
    }

    #[test]
    fn test_real_hv_bundle_similarity() {
        run_at_dims(|dim, rng| {
            let a = RealHV::random(dim, rng);
            let b = RealHV::random(dim, rng);
            let c = RealHV::random(dim, rng);
            let bundle = RealHV::bundle(&[&a, &b, &c]);
            assert!(
                RealHV::cosine_similarity(&bundle, &a) > 0.3,
                "D={dim}: bundle sim to a"
            );
            assert!(
                RealHV::cosine_similarity(&bundle, &b) > 0.3,
                "D={dim}: bundle sim to b"
            );
            assert!(
                RealHV::cosine_similarity(&bundle, &c) > 0.3,
                "D={dim}: bundle sim to c"
            );
        });
    }

    #[test]
    fn test_real_hv_orthogonality() {
        run_at_dims(|dim, rng| {
            let vecs: Vec<RealHV> = (0..1000).map(|_| RealHV::random(dim, rng)).collect();
            let mut total = 0.0f64;
            let mut count = 0u64;
            for i in 0..100 {
                for j in (i + 1)..100 {
                    total += RealHV::cosine_similarity(&vecs[i], &vecs[j]) as f64;
                    count += 1;
                }
            }
            let mean = total / count as f64;
            assert!(mean.abs() < 0.05, "D={dim}: mean sim = {mean}");
        });
    }

    #[test]
    fn test_real_hv_permute() {
        run_at_dims(|dim, rng| {
            let v = RealHV::random(dim, rng);
            let p = v.permute(1);
            let sim = RealHV::cosine_similarity(&v, &p);
            assert!(sim.abs() < 0.1, "D={dim}: permute sim = {sim}");
        });
    }

    #[test]
    fn test_binary_to_real_roundtrip() {
        run_at_dims(|dim, rng| {
            use crate::hdc::binary::BinaryHV;
            let b = BinaryHV::random(dim, rng);
            let r = b.to_real();
            let b2 = r.to_binary();
            assert_eq!(b, b2, "D={dim}: roundtrip failed");
        });
    }

    #[test]
    fn test_real_hv_similarity_matches_binary() {
        run_at_dims(|dim, rng| {
            use crate::hdc::binary::BinaryHV;
            let a = BinaryHV::random(dim, rng);
            let b = BinaryHV::random(dim, rng);
            let bin_sim = a.similarity(&b);
            let real_sim = RealHV::cosine_similarity(&a.to_real(), &b.to_real());
            let diff = (bin_sim - real_sim).abs();
            assert!(
                diff < 0.01,
                "D={dim}: binary sim {bin_sim} vs real sim {real_sim}, diff = {diff}"
            );
        });
    }

    #[test]
    fn test_real_hv_normalize() {
        run_at_dims(|dim, rng| {
            let v = RealHV::random(dim, rng);
            let n = v.normalized();
            assert!(
                (n.norm() - 1.0).abs() < 1e-5,
                "D={dim}: normalized norm = {}",
                n.norm()
            );
        });
    }
}
