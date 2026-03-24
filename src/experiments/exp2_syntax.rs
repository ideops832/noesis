//! Experiment 2: Emergent syntactic structure.
//!
//! Expanded protocol:
//! 1. Word order sensitivity: same words in different order yield different encodings
//! 2. Negation sensitivity: "X does Y" vs "X does not Y"
//! 3. Structure vs meaning: same SVO structure, different semantics
//! 4. Export CSV results to results/experiments/

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;

    use crate::hdc::hypervector::HyperVector;
    use crate::hdc::real::RealHV;
    use crate::language::tokenizer::Tokenizer;
    use crate::language::vocabulary::Vocabulary;
    use crate::language::composer::Composer;
    use crate::utils::corpus;

    const D: usize = 1024;

    #[test]
    fn test_exp2_research() {
        println!("\n{}", "=".repeat(60));
        println!("=== Experiment 2: Syntactic Structure — Expanded Protocol ===");
        println!("{}\n", "=".repeat(60));

        // ── 1. Build trained composer ──────────────────────────────────────
        println!("--- Phase 1: Build trained composer ---");

        let tokenizer = Tokenizer::new();
        let mut full_corpus = corpus::expanded_corpus();
        full_corpus.extend(corpus::synonym_parallel_corpus());
        let sentences: Vec<Vec<String>> = full_corpus.iter().map(|s| tokenizer.tokenize(s)).collect();

        let mut composer = Composer::new(D, 42);
        // Pre-populate vocabulary
        for sentence in &sentences {
            for word in sentence {
                composer.vocabulary.get_or_create(word);
            }
        }
        // Train with momentum
        for _ in 0..3 {
            composer.vocabulary.learn_from_context_momentum(&sentences, 3, 0.7);
        }
        println!("  Vocab: {} words, 3 passes momentum=0.7\n", composer.vocabulary.len());

        // ── 2. Word order test ─────────────────────────────────────────────
        println!("--- Phase 2: Word order sensitivity ---\n");

        let word_order_pairs: Vec<(&str, &str)> = vec![
            ("gatto mangia topo", "topo mangia gatto"),
            ("cane insegue gatto", "gatto insegue cane"),
            ("mamma chiama bambino", "bambino chiama mamma"),
            ("sole scalda terra", "terra scalda sole"),
            ("vento muove nuvola", "nuvola muove vento"),
            ("pesce nuota mare", "mare nuota pesce"),
            ("uccello canta albero", "albero canta uccello"),
            ("cuoco prepara cena", "cena prepara cuoco"),
            ("nonno racconta storia", "storia racconta nonno"),
            ("fratello aiuta sorella", "sorella aiuta fratello"),
        ];

        struct PairResult {
            a: String,
            b: String,
            similarity: f32,
        }

        let mut wo_results: Vec<PairResult> = Vec::new();

        println!("  {:35} | {:35} | {:>8}", "Sentence A", "Sentence B", "Sim");
        println!("  {:-<35}-+-{:-<35}-+-{:-<8}", "", "", "");

        for (a, b) in &word_order_pairs {
            let sim = composer.sentence_similarity(a, b);
            println!("  {:35} | {:35} | {:>8.4}", a, b, sim);
            wo_results.push(PairResult { a: a.to_string(), b: b.to_string(), similarity: sim });
        }

        let avg_wo: f32 = wo_results.iter().map(|r| r.similarity).sum::<f32>() / wo_results.len() as f32;
        println!("\n  Average word-order similarity: {:.4}", avg_wo);

        // ── 3. Negation test ───────────────────────────────────────────────
        println!("\n--- Phase 3: Negation sensitivity ---\n");

        let negation_pairs: Vec<(&str, &str)> = vec![
            ("gatto dorme", "gatto non dorme"),
            ("cane mangia", "cane non mangia"),
            ("bambino gioca", "bambino non gioca"),
            ("sole splende", "sole non splende"),
            ("pioggia cade", "pioggia non cade"),
        ];

        let mut neg_results: Vec<PairResult> = Vec::new();

        println!("  {:25} | {:25} | {:>8}", "Affirmative", "Negative", "Sim");
        println!("  {:-<25}-+-{:-<25}-+-{:-<8}", "", "", "");

        for (a, b) in &negation_pairs {
            let sim = composer.sentence_similarity(a, b);
            println!("  {:25} | {:25} | {:>8.4}", a, b, sim);
            neg_results.push(PairResult { a: a.to_string(), b: b.to_string(), similarity: sim });
        }

        let avg_neg: f32 = neg_results.iter().map(|r| r.similarity).sum::<f32>() / neg_results.len() as f32;
        println!("\n  Average negation similarity: {:.4}", avg_neg);
        println!("  (High similarity expected: 'non' is just one extra token)");

        // ── 4. Structure vs meaning ────────────────────────────────────────
        println!("\n--- Phase 4: Structure vs meaning ---\n");

        let structure_pairs: Vec<(&str, &str)> = vec![
            ("gatto mangia topo", "sole scalda terra"),
            ("cane corre parco", "pioggia bagna strada"),
            ("mamma cucina cena", "vento muove nuvola"),
            ("bambino gioca palla", "treno parte stazione"),
            ("pesce nuota mare", "uccello vola cielo"),
            ("nonno racconta storia", "cuoco prepara torta"),
            ("fratello legge libro", "neve copre montagna"),
            ("gatto dorme divano", "luna illumina notte"),
            ("cane beve acqua", "sole tramonta sera"),
            ("bambino ride forte", "pioggia cade piano"),
        ];

        let mut sm_results: Vec<PairResult> = Vec::new();

        println!("  {:30} | {:30} | {:>8}", "Sentence A", "Sentence B", "Sim");
        println!("  {:-<30}-+-{:-<30}-+-{:-<8}", "", "", "");

        for (a, b) in &structure_pairs {
            let sim = composer.sentence_similarity(a, b);
            println!("  {:30} | {:30} | {:>8.4}", a, b, sim);
            sm_results.push(PairResult { a: a.to_string(), b: b.to_string(), similarity: sim });
        }

        let avg_sm: f32 = sm_results.iter().map(|r| r.similarity).sum::<f32>() / sm_results.len() as f32;
        println!("\n  Average structure-vs-meaning similarity: {:.4}", avg_sm);

        // ── 5. Summary table ───────────────────────────────────────────────
        println!("\n--- Phase 5: Summary ---\n");
        println!("  {:30} | {:>10} | {:>10}", "Sub-test", "Avg Sim", "N pairs");
        println!("  {:-<30}-+-{:-<10}-+-{:-<10}", "", "", "");
        println!("  {:30} | {:>10.4} | {:>10}", "Word order (reversed SVO)", avg_wo, wo_results.len());
        println!("  {:30} | {:>10.4} | {:>10}", "Negation (X vs not X)", avg_neg, neg_results.len());
        println!("  {:30} | {:>10.4} | {:>10}", "Structure vs meaning", avg_sm, sm_results.len());

        // ── 6. Save CSVs ──────────────────────────────────────────────────
        println!("\n--- Phase 6: Saving CSV results ---");

        // exp2_word_order.csv
        {
            let mut f = File::create("results/experiments/exp2_word_order.csv")
                .expect("Failed to create exp2_word_order.csv");
            writeln!(f, "sentence_a,sentence_b,similarity").unwrap();
            for r in &wo_results {
                writeln!(f, "\"{}\",\"{}\",{:.6}", r.a, r.b, r.similarity).unwrap();
            }
            println!("  Saved results/experiments/exp2_word_order.csv");
        }

        // exp2_negation.csv
        {
            let mut f = File::create("results/experiments/exp2_negation.csv")
                .expect("Failed to create exp2_negation.csv");
            writeln!(f, "affirmative,negative,similarity").unwrap();
            for r in &neg_results {
                writeln!(f, "\"{}\",\"{}\",{:.6}", r.a, r.b, r.similarity).unwrap();
            }
            println!("  Saved results/experiments/exp2_negation.csv");
        }

        // exp2_structure_vs_meaning.csv
        {
            let mut f = File::create("results/experiments/exp2_structure_vs_meaning.csv")
                .expect("Failed to create exp2_structure_vs_meaning.csv");
            writeln!(f, "sentence_a,sentence_b,similarity").unwrap();
            for r in &sm_results {
                writeln!(f, "\"{}\",\"{}\",{:.6}", r.a, r.b, r.similarity).unwrap();
            }
            println!("  Saved results/experiments/exp2_structure_vs_meaning.csv");
        }

        // ── 7. Assertions ─────────────────────────────────────────────────
        println!("\n--- Phase 7: Assertions ---");

        println!("  Word order avg similarity: {:.4} (threshold < 0.7)", avg_wo);
        assert!(
            avg_wo < 0.7,
            "Average word-order similarity ({:.4}) should be < 0.7 (word order matters)",
            avg_wo
        );
        println!("  [PASS] Word order sensitivity: avg={:.4} < 0.7", avg_wo);

        println!("  Structure-vs-meaning avg similarity: {:.4} (threshold < 0.3)", avg_sm);
        assert!(
            avg_sm < 0.3,
            "Average structure-vs-meaning similarity ({:.4}) should be < 0.3 (different semantics)",
            avg_sm
        );
        println!("  [PASS] Structure vs meaning: avg={:.4} < 0.3", avg_sm);

        println!("\n=== Experiment 2 Complete ===\n");
    }
}
