//! Experiment 2: Emergent syntactic structure.
//!
//! Tests whether sentence encoding preserves meaning similarity:
//! pairs of sentences with similar meaning (but different structure)
//! should have higher cosine similarity than pairs with different meaning.

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;

    use crate::language::composer::Composer;

    const D: usize = 1024;

    #[test]
    fn test_exp2_syntax() {
        println!("\n=== Experiment 2: Syntactic Structure ===\n");

        let mut composer = Composer::new(D, 42);

        // Pairs with similar meaning (paraphrases / near-synonymous)
        let similar_pairs = vec![
            ("Il gatto mangia il pesce", "Il felino mangia il pesce"),
            ("Il cane corre nel parco", "Il cucciolo corre nel parco"),
            ("La mamma cucina la cena", "La madre prepara la cena"),
            ("Il bambino gioca nel giardino", "Il bambino gioca nel cortile"),
            ("Il sole splende nel cielo", "Il sole brilla nel cielo"),
        ];

        // Pairs with different meaning
        let different_pairs = vec![
            ("Il gatto mangia il pesce", "Il treno parte dalla stazione"),
            ("Il cane corre nel parco", "La pasta cuoce nella pentola"),
            ("La mamma cucina la cena", "Il vento soffia tra gli alberi"),
            ("Il bambino gioca nel giardino", "Il lavoratore arriva in ufficio"),
            ("Il sole splende nel cielo", "Il gatto dorme sul divano"),
        ];

        println!("--- Similar Meaning Pairs ---");
        let mut sim_scores = Vec::new();
        for (a, b) in &similar_pairs {
            let sim = composer.sentence_similarity(a, b);
            sim_scores.push(sim);
            println!("  {:.4}  \"{}\" vs \"{}\"", sim, a, b);
        }
        let avg_similar: f32 = sim_scores.iter().sum::<f32>() / sim_scores.len() as f32;

        println!("\n--- Different Meaning Pairs ---");
        let mut diff_scores = Vec::new();
        for (a, b) in &different_pairs {
            let sim = composer.sentence_similarity(a, b);
            diff_scores.push(sim);
            println!("  {:.4}  \"{}\" vs \"{}\"", sim, a, b);
        }
        let avg_different: f32 = diff_scores.iter().sum::<f32>() / diff_scores.len() as f32;

        println!("\n--- Summary ---");
        println!("  Average similarity (similar meaning):    {:.4}", avg_similar);
        println!("  Average similarity (different meaning):  {:.4}", avg_different);
        println!("  Difference:                              {:.4}", avg_similar - avg_different);

        // --- CSV Export ---
        {
            let mut f = File::create("results/exp2_syntax.csv")
                .expect("Failed to create exp2_syntax.csv");
            writeln!(f, "type,sentence_a,sentence_b,similarity").unwrap();
            for ((a, b), sim) in similar_pairs.iter().zip(sim_scores.iter()) {
                writeln!(f, "similar,\"{}\",\"{}\",{:.6}", a, b, sim).unwrap();
            }
            for ((a, b), sim) in different_pairs.iter().zip(diff_scores.iter()) {
                writeln!(f, "different,\"{}\",\"{}\",{:.6}", a, b, sim).unwrap();
            }
        }

        println!("\n[CSV] Saved results/exp2_syntax.csv");

        assert!(
            avg_similar > avg_different,
            "Similar-meaning pairs ({:.4}) should have higher avg similarity than different-meaning pairs ({:.4})",
            avg_similar,
            avg_different
        );

        println!("\n[PASS] avg_similar ({:.4}) > avg_different ({:.4})", avg_similar, avg_different);
        println!("\n=== Experiment 2 Complete ===\n");
    }
}
