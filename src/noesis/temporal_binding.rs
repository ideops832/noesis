//! Temporal binding: binding with a temporal dimension.
//!
//! Extends HDC binding with a notion of time. A temporal binding encodes
//! both content and temporal position, allowing ordered sequences to be
//! represented and later queried. The temporal component is created via
//! permutation, which acts as a time-shift operator in HD space.

use crate::hdc::real::RealHV;

/// Bind a hypervector with a temporal index.
///
/// Uses cyclic permutation to create a time-dependent representation:
///   temporal_bind(v, t) = permute(v, t)
///
/// This preserves the information in `v` while making it orthogonal to
/// the same content at different time steps.
///
/// Multiple temporal bindings can be bundled to form a sequence representation:
///   sequence = bundle([ temporal_bind(v0, 0), temporal_bind(v1, 1), ... ])
pub fn temporal_bind(vector: &RealHV, time_step: usize) -> RealHV {
    vector.permute(time_step)
}

/// Measure the quality of temporal unbinding.
///
/// Given a sequence (bundle of temporal bindings), a query vector, and a
/// time step, returns the cosine similarity between the inverse-permuted
/// sequence and the original query. High similarity indicates the query
/// content was present at that time step.
///
/// Unbinding:
///   recovered = permute(sequence, dim - time_step)  // inverse permutation
///   quality = cosine_similarity(recovered, query)
pub fn temporal_unbind_quality(
    sequence: &RealHV,
    query: &RealHV,
    time_step: usize,
) -> f32 {
    let dim = sequence.dim;
    // Inverse of permute(_, t) is permute(_, dim - t)
    let inverse_shift = if time_step.is_multiple_of(dim) { 0 } else { dim - (time_step % dim) };
    let recovered = sequence.permute(inverse_shift);
    RealHV::cosine_similarity(&recovered, query)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::real::RealHV;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    const D: usize = 1024;

    #[test]
    fn test_temporal_binding_works() {
        let mut rng = StdRng::seed_from_u64(42);

        // Create three distinct vectors
        let v0 = RealHV::random(D, &mut rng);
        let v1 = RealHV::random(D, &mut rng);
        let v2 = RealHV::random(D, &mut rng);

        // Bind each with a temporal index
        let tb0 = temporal_bind(&v0, 0);
        let tb1 = temporal_bind(&v1, 1);
        let tb2 = temporal_bind(&v2, 2);

        // Bundle into a sequence
        let sequence = RealHV::bundle(&[&tb0, &tb1, &tb2]);

        // Unbinding should recover each item at the correct time step
        let q0 = temporal_unbind_quality(&sequence, &v0, 0);
        let q1 = temporal_unbind_quality(&sequence, &v1, 1);
        let q2 = temporal_unbind_quality(&sequence, &v2, 2);

        assert!(
            q0 > 0.3,
            "Should recover v0 at t=0: quality = {}",
            q0
        );
        assert!(
            q1 > 0.3,
            "Should recover v1 at t=1: quality = {}",
            q1
        );
        assert!(
            q2 > 0.3,
            "Should recover v2 at t=2: quality = {}",
            q2
        );

        // Cross-time queries should have low similarity
        let cross_01 = temporal_unbind_quality(&sequence, &v0, 1);
        let cross_10 = temporal_unbind_quality(&sequence, &v1, 0);
        assert!(
            cross_01.abs() < q0,
            "Cross-time query (v0 at t=1) should be weaker than correct: {} vs {}",
            cross_01.abs(),
            q0
        );
        assert!(
            cross_10.abs() < q1,
            "Cross-time query (v1 at t=0) should be weaker than correct: {} vs {}",
            cross_10.abs(),
            q1
        );
    }

    #[test]
    fn test_temporal_binding_decays() {
        // As the sequence grows longer, the quality of unbinding each element
        // should decrease because each element's contribution is diluted.
        let mut rng = StdRng::seed_from_u64(42);
        let target = RealHV::random(D, &mut rng);

        // Short sequence: 3 items
        let short_items: Vec<RealHV> = (0..3).map(|_| RealHV::random(D, &mut rng)).collect();
        let mut short_bindings: Vec<RealHV> = short_items
            .iter()
            .enumerate()
            .map(|(t, v)| temporal_bind(v, t))
            .collect();
        // Insert target at position 0
        short_bindings[0] = temporal_bind(&target, 0);

        let short_refs: Vec<&RealHV> = short_bindings.iter().collect();
        let short_seq = RealHV::bundle(&short_refs);
        let short_quality = temporal_unbind_quality(&short_seq, &target, 0);

        // Long sequence: 30 items
        let mut rng2 = StdRng::seed_from_u64(42);
        let _target2 = RealHV::random(D, &mut rng2); // same target
        let long_items: Vec<RealHV> = (0..30).map(|_| RealHV::random(D, &mut rng2)).collect();
        let mut long_bindings: Vec<RealHV> = long_items
            .iter()
            .enumerate()
            .map(|(t, v)| temporal_bind(v, t))
            .collect();
        long_bindings[0] = temporal_bind(&target, 0);

        let long_refs: Vec<&RealHV> = long_bindings.iter().collect();
        let long_seq = RealHV::bundle(&long_refs);
        let long_quality = temporal_unbind_quality(&long_seq, &target, 0);

        assert!(
            short_quality > long_quality,
            "Short sequence should have better unbinding quality: short={}, long={}",
            short_quality,
            long_quality
        );

        // Both should still be positive (signal present, just diluted)
        assert!(
            short_quality > 0.0,
            "Short sequence quality should be positive: {}",
            short_quality
        );
        assert!(
            long_quality > 0.0,
            "Long sequence quality should be positive: {}",
            long_quality
        );
    }
}
