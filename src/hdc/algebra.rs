//! Algebraic properties of HDC operations: empirical verification.
//!
//! Verifies group structure, commutativity, associativity, and distributivity
//! of the HDC operations on binary hypervectors.

#[cfg(test)]
mod tests {
    use crate::hdc::binary::BinaryHV;
    use crate::hdc::hypervector::HyperVector;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    const DIM: usize = 4096;

    #[test]
    fn test_bind_is_group() {
        let mut rng = StdRng::seed_from_u64(42);
        let a = BinaryHV::random(DIM, &mut rng);
        let b = BinaryHV::random(DIM, &mut rng);
        let c = BinaryHV::random(DIM, &mut rng);

        // Closure: bind(a,b) has same dim
        let ab = BinaryHV::bind(&a, &b);
        assert_eq!(ab.dim, DIM);

        // Associativity: (a⊗b)⊗c = a⊗(b⊗c)
        let ab_c = BinaryHV::bind(&ab, &c);
        let bc = BinaryHV::bind(&b, &c);
        let a_bc = BinaryHV::bind(&a, &bc);
        assert_eq!(ab_c, a_bc, "Associativity failed");

        // Identity element: zero vector (XOR with zero = self)
        let zero = BinaryHV::zero(DIM);
        let a_zero = BinaryHV::bind(&a, &zero);
        assert_eq!(a_zero, a, "Zero is not identity for bind");

        // Inverse: bind(a, a) = zero (self-inverse)
        let a_a = BinaryHV::bind(&a, &a);
        assert_eq!(a_a, zero, "Self-inverse failed");
    }

    #[test]
    fn test_bundle_commutativity() {
        let mut rng = StdRng::seed_from_u64(42);
        let a = BinaryHV::random(DIM, &mut rng);
        let b = BinaryHV::random(DIM, &mut rng);
        let c = BinaryHV::random(DIM, &mut rng);

        // Bundle is commutative (majority vote is order-independent)
        let mut rng1 = StdRng::seed_from_u64(99);
        let mut rng2 = StdRng::seed_from_u64(99);
        let bundle_abc = BinaryHV::bundle(&[&a, &b, &c], &mut rng1);
        let bundle_cba = BinaryHV::bundle(&[&c, &b, &a], &mut rng2);
        // With odd number of vectors (3), no tie-breaking needed → exactly equal
        assert_eq!(bundle_abc, bundle_cba, "Bundle not commutative for odd count");
    }

    #[test]
    fn test_permute_compose() {
        let mut rng = StdRng::seed_from_u64(42);
        let v = BinaryHV::random(DIM, &mut rng);

        // permute(v, a+b) == permute(permute(v, a), b)
        let a = 7;
        let b = 13;
        let p_ab = v.permute(a + b);
        let p_a_then_b = v.permute(a).permute(b);
        assert_eq!(p_ab, p_a_then_b, "Permute composition failed");
    }

    #[test]
    fn test_distributivity() {
        // Does bind distribute over bundle?
        // bind(a, bundle(b, c)) ≈? bundle(bind(a,b), bind(a,c))
        // For binary XOR-based bind over majority bundle, this does NOT hold exactly.
        let mut rng = StdRng::seed_from_u64(42);
        let a = BinaryHV::random(DIM, &mut rng);
        let b = BinaryHV::random(DIM, &mut rng);
        let c = BinaryHV::random(DIM, &mut rng);
        let d = BinaryHV::random(DIM, &mut rng); // odd count for bundle

        let bundle_bcd = BinaryHV::bundle(&[&b, &c, &d], &mut rng);
        let lhs = BinaryHV::bind(&a, &bundle_bcd);

        let ab = BinaryHV::bind(&a, &b);
        let ac = BinaryHV::bind(&a, &c);
        let ad = BinaryHV::bind(&a, &d);
        let rhs = BinaryHV::bundle(&[&ab, &ac, &ad], &mut rng);

        let sim = lhs.similarity(&rhs);

        // Document the result: XOR distributes over majority approximately
        // because XOR is bitwise and majority is bitwise.
        // Actually for binary: XOR(a, majority(b,c,d)) = majority(XOR(a,b), XOR(a,c), XOR(a,d))
        // This should hold EXACTLY because both are bitwise operations.
        println!("Distributivity similarity: {sim}");
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "Distributivity: bind over bundle sim = {sim} (expected 1.0 for binary)"
        );
    }
}
