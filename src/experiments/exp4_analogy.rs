//! Experiment 4: Analogical reasoning.
//!
//! After contextual learning on the Italian corpus, tests analogies like
//! "re:italia :: ? : francia" using vector arithmetic (b - a + c).
//! Reports accuracy honestly, even if low.

#[cfg(test)]
mod tests {
    use crate::language::tokenizer::Tokenizer;
    use crate::language::vocabulary::Vocabulary;

    const D: usize = 2048;

    #[test]
    fn test_exp4_analogy() {
        println!("\n=== Experiment 4: Analogical Reasoning ===\n");

        let tokenizer = Tokenizer::new();
        let corpus = crate::utils::corpus::expanded_corpus();

        // Tokenize corpus
        let sentences: Vec<Vec<String>> = corpus
            .iter()
            .map(|s| tokenizer.tokenize(s.as_str()))
            .collect();

        // Build vocabulary with distributional learning (higher D for analogies)
        let mut vocab = Vocabulary::new(2048, 42);

        // Ensure all words exist
        for sentence in &sentences {
            for word in sentence {
                vocab.get_or_create(word);
            }
        }

        // 5 passes of contextual learning on expanded corpus
        for pass in 0..5 {
            vocab.learn_from_context(&sentences, 3);
            println!("Learning pass {} complete", pass + 1);
        }

        println!("Vocabulary size: {} words\n", vocab.len());

        // Define analogy tests using words from the corpus
        // Format: (a, b, c, expected_answers) where b - a + c should give expected
        let analogies: Vec<(&str, &str, &str, Vec<&str>)> = vec![
            // Animal synonyms: gatto is to felino as cane is to ?
            ("gatto", "felino", "cane", vec!["cucciolo"]),
            // Actions: gatto is to dorme as cane is to ?
            ("gatto", "dorme", "cane", vec!["dorme", "corre", "gioca"]),
            // Family: mamma is to cucina as papa is to ?
            ("mamma", "cucina", "papà", vec!["cucina", "prepara", "gioca"]),
            // Place: gatto is to divano as cane is to ?
            ("gatto", "divano", "cane", vec!["parco", "cortile"]),
            // Animals: gatto is to micio as cane is to ?
            ("gatto", "micio", "cane", vec!["cucciolo"]),
        ];

        let mut correct = 0;
        let total = analogies.len();

        for (a, b, c, expected) in &analogies {
            let results = vocab.analogy(a, b, c, 5);

            println!("  {} : {} :: {} : ?", a, b, c);
            println!("    Expected: {:?}", expected);

            if results.is_empty() {
                println!("    Got: (no results - words not in vocabulary)");
            } else {
                println!("    Top 5 results:");
                for (rank, (word, sim)) in results.iter().enumerate() {
                    let marker = if expected.contains(&word.as_str()) { " <--" } else { "" };
                    println!("      {:2}. {:15} : {:.4}{}", rank + 1, word, sim, marker);
                }

                // Check if any expected answer is in top 5
                let found = results.iter().any(|(w, _)| expected.contains(&w.as_str()));
                if found {
                    correct += 1;
                    println!("    [HIT] Found expected answer in top 5");
                } else {
                    println!("    [MISS] Expected answer not in top 5");
                }
            }
            println!();
        }

        let accuracy = correct as f32 / total as f32;
        println!("--- Summary ---");
        println!("  Accuracy (top-5): {}/{} = {:.1}%", correct, total, accuracy * 100.0);
        println!("  Note: Low accuracy is expected with distributional learning alone.");
        println!("  HDC analogy requires strong distributional signal which a small");
        println!("  corpus may not provide sufficiently.");

        // Soft assertion: we just verify the analogy mechanism runs and produces results.
        // Accuracy may be low and that is documented honestly.
        let all_words_exist = ["gatto", "felino", "cane", "cucciolo"]
            .iter()
            .all(|w| vocab.words.contains_key(*w));
        assert!(all_words_exist, "Key vocabulary words should exist after learning");

        // Verify analogy produces non-empty results for words that exist
        let results = vocab.analogy("gatto", "felino", "cane", 5);
        assert!(
            !results.is_empty(),
            "Analogy should produce results for known words"
        );

        println!("\n[PASS] Analogy mechanism works; accuracy = {:.1}%", accuracy * 100.0);

        // --- CSV Export ---
        use std::fs::File;
        use std::io::Write;
        let mut csv_file = File::create("results/exp4_analogy.csv")
            .expect("Failed to create results/exp4_analogy.csv");
        writeln!(csv_file, "a,b,c,expected,rank1_word,rank1_sim,rank2_word,rank2_sim,rank3_word,rank3_sim,hit")
            .unwrap();
        for (a, b, c, expected) in &analogies {
            let results = vocab.analogy(a, b, c, 5);
            let hit = results.iter().any(|(w, _)| expected.contains(&w.as_str()));
            let get = |idx: usize| -> (String, String) {
                if let Some((w, s)) = results.get(idx) {
                    (w.clone(), format!("{:.6}", s))
                } else {
                    ("".to_string(), "".to_string())
                }
            };
            let (w1, s1) = get(0);
            let (w2, s2) = get(1);
            let (w3, s3) = get(2);
            writeln!(
                csv_file,
                "{},{},{},{:?},{},{},{},{},{},{},{}",
                a, b, c, expected, w1, s1, w2, s2, w3, s3, hit
            ).unwrap();
        }
        println!("CSV saved to results/exp4_analogy.csv");

        println!("\n=== Experiment 4 Complete ===\n");
    }
}
