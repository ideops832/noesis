//! Experiment 5: Productive compositionality.
//!
//! Splits the corpus into train (first 250) and test (last 75) sentences.
//! Encodes all sentences. Verifies that test sentences sharing words with
//! train sentences are more similar to those related train sentences than
//! to unrelated ones, demonstrating productive generalization.

#[cfg(test)]
mod tests {
    use crate::hdc::real::RealHV;
    use crate::language::composer::Composer;
    use crate::language::tokenizer::Tokenizer;
    use crate::utils::corpus::italian_corpus;

    const D: usize = 1024;

    #[test]
    fn test_exp5_productivity() {
        println!("\n=== Experiment 5: Productive Compositionality ===\n");

        let corpus = italian_corpus();
        let tokenizer = Tokenizer::new();
        let mut composer = Composer::new(D, 42);

        // Split into train and test
        let train_sentences: Vec<&str> = corpus[..250].to_vec();
        let test_sentences: Vec<&str> = corpus[250..].to_vec();
        println!("Train sentences: {}", train_sentences.len());
        println!("Test sentences: {}", test_sentences.len());

        // Encode train sentences
        let train_encoded: Vec<(usize, RealHV)> = train_sentences
            .iter()
            .enumerate()
            .filter_map(|(i, s)| composer.encode_sentence(s).map(|hv| (i, hv)))
            .collect();

        // Encode test sentences
        let test_encoded: Vec<(usize, RealHV)> = test_sentences
            .iter()
            .enumerate()
            .filter_map(|(i, s)| composer.encode_sentence(s).map(|hv| (i, hv)))
            .collect();

        println!("Encoded train: {}", train_encoded.len());
        println!("Encoded test: {}", test_encoded.len());

        // Tokenize train and test
        let train_tokens: Vec<Vec<String>> = train_sentences
            .iter()
            .map(|s| tokenizer.tokenize(s))
            .collect();
        let test_tokens: Vec<Vec<String>> = test_sentences
            .iter()
            .map(|s| tokenizer.tokenize(s))
            .collect();

        // For each test sentence, find train sentences that share at least one
        // content word (related) vs those that share none (unrelated).
        // Average similarity should be higher for related pairs.
        let mut related_sims: Vec<f32> = Vec::new();
        let mut unrelated_sims: Vec<f32> = Vec::new();
        let mut examples_shown = 0;

        for (test_idx, test_hv) in &test_encoded {
            let test_toks: std::collections::HashSet<&str> = test_tokens[*test_idx]
                .iter()
                .map(|s| s.as_str())
                .collect();

            if test_toks.is_empty() {
                continue;
            }

            for (train_idx, train_hv) in &train_encoded {
                let train_toks: std::collections::HashSet<&str> = train_tokens[*train_idx]
                    .iter()
                    .map(|s| s.as_str())
                    .collect();

                let shared: Vec<&&str> = test_toks.intersection(&train_toks).collect();
                let sim = RealHV::cosine_similarity(test_hv, train_hv);

                if !shared.is_empty() {
                    related_sims.push(sim);

                    if examples_shown < 5 {
                        println!(
                            "\n  Related pair (shared: {:?}):",
                            shared.iter().take(3).collect::<Vec<_>>()
                        );
                        println!("    Test:  \"{}\"", test_sentences[*test_idx]);
                        println!("    Train: \"{}\"", train_sentences[*train_idx]);
                        println!("    Sim:   {:.4}", sim);
                        examples_shown += 1;
                    }
                } else {
                    unrelated_sims.push(sim);
                }
            }
        }

        let avg_related = if related_sims.is_empty() {
            0.0
        } else {
            related_sims.iter().sum::<f32>() / related_sims.len() as f32
        };
        let avg_unrelated = if unrelated_sims.is_empty() {
            0.0
        } else {
            unrelated_sims.iter().sum::<f32>() / unrelated_sims.len() as f32
        };

        println!("\n--- Summary ---");
        println!("  Related pairs:   {} (avg sim: {:.4})", related_sims.len(), avg_related);
        println!("  Unrelated pairs: {} (avg sim: {:.4})", unrelated_sims.len(), avg_unrelated);
        println!("  Difference:      {:.4}", avg_related - avg_unrelated);

        assert!(
            avg_related > avg_unrelated,
            "Related pairs ({:.4}) should have higher avg similarity than unrelated ({:.4})",
            avg_related,
            avg_unrelated
        );

        println!("\n[PASS] avg_related ({:.4}) > avg_unrelated ({:.4})", avg_related, avg_unrelated);

        // --- CSV Export (sample: first 500 related + 500 unrelated) ---
        use std::fs::File;
        use std::io::Write;
        {
            let mut csv_file = File::create("results/exp5_productivity.csv")
                .expect("Failed to create results/exp5_productivity.csv");
            writeln!(csv_file, "type,test_idx,train_idx,similarity").unwrap();

            let mut related_count = 0usize;
            let mut unrelated_count = 0usize;
            let max_sample = 500;

            for (test_idx, test_hv) in &test_encoded {
                let test_toks: std::collections::HashSet<&str> = test_tokens[*test_idx]
                    .iter()
                    .map(|s| s.as_str())
                    .collect();
                if test_toks.is_empty() {
                    continue;
                }
                for (train_idx, train_hv) in &train_encoded {
                    if related_count >= max_sample && unrelated_count >= max_sample {
                        break;
                    }
                    let train_toks: std::collections::HashSet<&str> = train_tokens[*train_idx]
                        .iter()
                        .map(|s| s.as_str())
                        .collect();
                    let shared: Vec<&&str> = test_toks.intersection(&train_toks).collect();
                    let sim = RealHV::cosine_similarity(test_hv, train_hv);
                    if !shared.is_empty() {
                        if related_count < max_sample {
                            writeln!(csv_file, "related,{},{},{:.6}", test_idx, train_idx, sim).unwrap();
                            related_count += 1;
                        }
                    } else if unrelated_count < max_sample {
                        writeln!(csv_file, "unrelated,{},{},{:.6}", test_idx, train_idx, sim).unwrap();
                        unrelated_count += 1;
                    }
                }
                if related_count >= max_sample && unrelated_count >= max_sample {
                    break;
                }
            }
            println!("CSV saved to results/exp5_productivity.csv ({} related, {} unrelated rows)",
                related_count, unrelated_count);
        }

        println!("\n=== Experiment 5 Complete ===\n");
    }
}
