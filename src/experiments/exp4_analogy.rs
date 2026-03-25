//! Experiment 4: Analogical reasoning.
//!
//! Expanded protocol:
//! 1. Train vocabulary on expanded + parallel corpus (5 passes, momentum=0.7)
//! 2. Test analogies using both vocab.analogy() (vector arithmetic) and vocab.analogy_hdc() (binding)
//! 3. At least 8 analogies, report top-5 for each, mark HIT/MISS
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
    fn test_exp4_research() {
        println!("\n{}", "=".repeat(60));
        println!("=== Experiment 4: Analogical Reasoning — Expanded Protocol ===");
        println!("{}\n", "=".repeat(60));

        // ── 1. Build trained vocabulary ────────────────────────────────────
        println!("--- Phase 1: Build trained vocabulary ---");

        let tokenizer = Tokenizer::new();
        let mut full_corpus = corpus::expanded_corpus();
        full_corpus.extend(corpus::synonym_parallel_corpus());
        full_corpus.extend(corpus::relational_corpus());
        let sentences: Vec<Vec<String>> = full_corpus.iter().map(|s| tokenizer.tokenize(s)).collect();

        let mut vocab = Vocabulary::new(D, 42);
        for sentence in &sentences {
            for word in sentence {
                vocab.get_or_create(word);
            }
        }

        for pass in 0..8 {
            vocab.learn_from_context_with_bigrams(&sentences, 3, 0.7, 0.5);
            if (pass + 1) % 2 == 0 || pass == 7 {
                println!("  Pass {}/8 complete", pass + 1);
            }
        }
        println!("  Vocabulary: {} words, D={}\n", vocab.len(), D);

        // ── 2. Define analogies ────────────────────────────────────────────
        // Format: (a, b, c, expected_answers, description)
        let analogies: Vec<(&str, &str, &str, Vec<&str>, &str)> = vec![
            ("mamma", "papà", "nonna", vec!["nonno"], "family gender: mother:father :: grandmother:?"),
            ("gatto", "felino", "cane", vec!["cucciolo", "animale"], "animal synonym: cat:feline :: dog:?"),
            ("sole", "giorno", "luna", vec!["notte", "sera"], "celestial time: sun:day :: moon:?"),
            ("cuoco", "cucina", "programmatore", vec!["codice", "ufficio", "computer"], "profession place: cook:kitchen :: programmer:?"),
            ("grande", "piccolo", "alto", vec!["basso"], "antonym scale: big:small :: tall:?"),
            ("gatto", "micio", "cane", vec!["cucciolo"], "animal diminutive: cat:kitty :: dog:?"),
            ("mamma", "cucina", "papà", vec!["lavoro", "ufficio", "gioca"], "family activity: mom:cooks :: dad:?"),
            ("gatto", "dorme", "cane", vec!["corre", "gioca", "abbaia"], "animal action: cat:sleeps :: dog:?"),
            ("bambino", "gioca", "nonno", vec!["racconta", "dorme", "legge"], "age activity: child:plays :: grandpa:?"),
            ("pioggia", "bagna", "sole", vec!["scalda", "splende", "illumina"], "element action: rain:wets :: sun:?"),
        ];

        // ── 3. Run analogies with both methods ─────────────────────────────
        println!("--- Phase 2: Analogy tests ---\n");

        struct AnalogyResult {
            a: String,
            b: String,
            c: String,
            expected: Vec<String>,
            method: String,
            rank1: String,
            rank1_sim: f32,
            hit: bool,
            top5: Vec<(String, f32)>,
        }

        let mut all_results: Vec<AnalogyResult> = Vec::new();
        let mut arith_hits = 0u32;
        let mut hdc_hits = 0u32;
        let total = analogies.len();

        for (a, b, c, expected, desc) in &analogies {
            println!("  {} : {} :: {} : ?  (expect: {:?})", a, b, c, expected);
            println!("    {}", desc);

            // --- Arithmetic method ---
            let arith_results = vocab.analogy(a, b, c, 5);
            let arith_hit = arith_results.iter().any(|(w, _)| expected.contains(&w.as_str()));
            if arith_hit { arith_hits += 1; }

            println!("    [ARITHMETIC] {}", if arith_hit { "HIT" } else { "MISS" });
            for (rank, (word, sim)) in arith_results.iter().enumerate() {
                let marker = if expected.contains(&word.as_str()) { " <--" } else { "" };
                println!("      {:2}. {:15} {:.4}{}", rank + 1, word, sim, marker);
            }

            let (arith_r1, arith_r1_sim) = arith_results.first()
                .map(|(w, s)| (w.clone(), *s))
                .unwrap_or(("".to_string(), 0.0));

            all_results.push(AnalogyResult {
                a: a.to_string(),
                b: b.to_string(),
                c: c.to_string(),
                expected: expected.iter().map(|s| s.to_string()).collect(),
                method: "arithmetic".to_string(),
                rank1: arith_r1,
                rank1_sim: arith_r1_sim,
                hit: arith_hit,
                top5: arith_results.clone(),
            });

            // --- HDC binding method ---
            let hdc_results = vocab.analogy_hdc(a, b, c, 5);
            let hdc_hit = hdc_results.iter().any(|(w, _)| expected.contains(&w.as_str()));
            if hdc_hit { hdc_hits += 1; }

            println!("    [HDC]        {}", if hdc_hit { "HIT" } else { "MISS" });
            for (rank, (word, sim)) in hdc_results.iter().enumerate() {
                let marker = if expected.contains(&word.as_str()) { " <--" } else { "" };
                println!("      {:2}. {:15} {:.4}{}", rank + 1, word, sim, marker);
            }

            let (hdc_r1, hdc_r1_sim) = hdc_results.first()
                .map(|(w, s)| (w.clone(), *s))
                .unwrap_or(("".to_string(), 0.0));

            all_results.push(AnalogyResult {
                a: a.to_string(),
                b: b.to_string(),
                c: c.to_string(),
                expected: expected.iter().map(|s| s.to_string()).collect(),
                method: "hdc".to_string(),
                rank1: hdc_r1,
                rank1_sim: hdc_r1_sim,
                hit: hdc_hit,
                top5: hdc_results.clone(),
            });

            println!();
        }

        // ── 4. Failure analysis ────────────────────────────────────────────
        println!("--- Phase 3: Failure analysis ---\n");

        let arith_misses: Vec<&AnalogyResult> = all_results.iter()
            .filter(|r| r.method == "arithmetic" && !r.hit)
            .collect();

        if arith_misses.is_empty() {
            println!("  No arithmetic misses to analyze!");
        } else {
            println!("  Arithmetic misses ({}):", arith_misses.len());
            for r in &arith_misses {
                println!("    {} : {} :: {} : ?  expected={:?}, got={}",
                    r.a, r.b, r.c, r.expected, r.rank1);
                // Check if expected words even exist in vocab
                for exp in &r.expected {
                    let in_vocab = vocab.words.contains_key(exp.as_str());
                    if !in_vocab {
                        println!("      -> '{}' NOT in vocabulary (impossible to find)", exp);
                    } else {
                        // Check similarity of expected to the analogy target
                        let sim_a_b = vocab.similarity(&r.a, &r.b);
                        let sim_c_exp = vocab.similarity(&r.c, exp);
                        println!("      -> '{}' in vocab, sim(a,b)={:.4}, sim(c,exp)={:.4}",
                            exp, sim_a_b, sim_c_exp);
                    }
                }
            }
        }

        // ── 5. Summary ────────────────────────────────────────────────────
        println!("\n--- Phase 4: Summary ---\n");

        let arith_acc = arith_hits as f32 / total as f32 * 100.0;
        let hdc_acc = hdc_hits as f32 / total as f32 * 100.0;

        // Combined: at least one method hits
        let combined_hits: u32 = analogies.iter().enumerate().map(|(i, _)| {
            let arith_hit = all_results[i * 2].hit;
            let hdc_hit = all_results[i * 2 + 1].hit;
            if arith_hit || hdc_hit { 1 } else { 0 }
        }).sum();
        let combined_acc = combined_hits as f32 / total as f32 * 100.0;

        println!("  {:20} | {:>10} | {:>10}", "Method", "Hits", "Accuracy");
        println!("  {:-<20}-+-{:-<10}-+-{:-<10}", "", "", "");
        println!("  {:20} | {:>10} | {:>9.1}%", "Arithmetic (b-a+c)", format!("{}/{}", arith_hits, total), arith_acc);
        println!("  {:20} | {:>10} | {:>9.1}%", "HDC (bind/unbind)", format!("{}/{}", hdc_hits, total), hdc_acc);
        println!("  {:20} | {:>10} | {:>9.1}%", "Combined (either)", format!("{}/{}", combined_hits, total), combined_acc);

        println!("\n  Note: Low accuracy expected with small corpus and distributional learning alone.");
        println!("  HDC analogies require strong distributional signal.");

        // ── 6. Save CSV ────────────────────────────────────────────────────
        println!("\n--- Phase 5: Saving CSV results ---");

        {
            let mut f = File::create("results/experiments/exp4_analogies.csv")
                .expect("Failed to create exp4_analogies.csv");
            writeln!(f, "a,b,c,expected,method,rank1,rank1_sim,hit").unwrap();
            for r in &all_results {
                writeln!(f, "{},{},{},\"{:?}\",{},{},{:.6},{}",
                    r.a, r.b, r.c, r.expected, r.method, r.rank1, r.rank1_sim, r.hit).unwrap();
            }
            println!("  Saved results/experiments/exp4_analogies.csv");
        }

        // ── 7. Assertions ─────────────────────────────────────────────────
        println!("\n--- Phase 6: Assertions ---");

        // At least 1 analogy should work (mechanism produces meaningful results)
        let any_hit = all_results.iter().any(|r| r.hit);
        println!("  Any analogy hit: {}", any_hit);
        println!("  Arithmetic accuracy: {:.1}%", arith_acc);
        println!("  HDC accuracy: {:.1}%", hdc_acc);

        assert!(
            any_hit,
            "At least 1 analogy should work across both methods (got 0 hits out of {} tests)",
            all_results.len()
        );

        println!("  [PASS] At least 1 analogy works");
        println!("\n=== Experiment 4 Complete ===\n");
    }
}
