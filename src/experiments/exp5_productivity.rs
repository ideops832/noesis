//! Experiment 5: Productive compositionality with grouped test sets.
//!
//! Trains vocabulary from expanded corpus, then tests generalization using
//! three groups of test sentences (creative, close-to-training, absurd Chomsky)
//! plus a discrimination test on confusable sentence pairs.

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;

    use crate::hdc::hypervector::HyperVector;
    use crate::hdc::real::RealHV;
    use crate::language::composer::Composer;
    use crate::language::tokenizer::Tokenizer;
    use crate::utils::corpus;

    const D: usize = 1024;

    #[test]
    fn test_exp5_research() {
        println!("\n=== Experiment 5: Productive Compositionality (Expanded) ===\n");

        // ------------------------------------------------------------------
        // 1. Build trained vocabulary from expanded corpus (3 passes, window=3, IDF)
        // ------------------------------------------------------------------
        let tokenizer = Tokenizer::new();
        let expanded = corpus::expanded_corpus();
        let expanded_sents: Vec<Vec<String>> = expanded
            .iter()
            .map(|s| tokenizer.tokenize(s.as_str()))
            .collect();

        let mut composer = Composer::new(D, 42);

        // Pre-populate vocabulary with all words from expanded corpus
        for s in &expanded_sents {
            for w in s {
                composer.vocabulary.get_or_create(w);
            }
        }

        // Distributional learning: 3 passes with window=3 and IDF weighting
        for pass in 0..3 {
            composer.vocabulary.learn_from_context(&expanded_sents, 3);
            println!("  Vocab training pass {}/3 done (vocab size: {})", pass + 1, composer.vocabulary.len());
        }

        // ------------------------------------------------------------------
        // 2. TRAIN = first 250 sentences of italian_corpus()
        // ------------------------------------------------------------------
        let italian = corpus::italian_corpus();
        let train_sentences: Vec<&str> = italian[..250].to_vec();
        println!("  Train sentences: {}", train_sentences.len());

        let train_encoded: Vec<(usize, &str, RealHV)> = train_sentences
            .iter()
            .enumerate()
            .filter_map(|(i, &s)| composer.encode_sentence(s).map(|hv| (i, s, hv)))
            .collect();
        println!("  Train encoded: {}", train_encoded.len());

        // ------------------------------------------------------------------
        // 3. TEST = 3 groups
        // ------------------------------------------------------------------
        let group_a: Vec<&str> = vec![
            "il pesce dorme nella cucina",
            "il programmatore cucina la pasta",
            "il gatto scrive un programma",
            "la nonna insegue il cane",
            "il sole mangia la neve",
            "il cuoco programma il computer",
            "la pioggia cucina la cena",
            "il bambino guida il treno",
            "il nonno nuota nel cielo",
            "il vento dorme sul divano",
        ];

        let group_b: Vec<&str> = vec![
            "il gatto dorme nel giardino",
            "il cane corre nel prato",
            "la mamma prepara il pranzo",
            "il sole splende forte",
            "il bambino gioca nel cortile",
            "il cuoco cucina il pesce",
            "la nonna racconta una storia",
            "il vento soffia nel parco",
            "il programmatore lavora al progetto",
            "la pioggia cade leggera",
        ];

        let group_c: Vec<&str> = vec![
            "le idee verdi dormono furiosamente",
            "la matematica viola corre veloce",
            "il silenzio azzurro mangia il tempo",
            "la tristezza rotonda nuota nel pensiero",
            "il coraggio liquido dorme sul nulla",
        ];

        // ------------------------------------------------------------------
        // 4. Encode all test sentences, find top-3 most similar TRAIN
        // ------------------------------------------------------------------
        struct TestResult {
            group: String,
            sentence: String,
            top1_train: String,
            top1_sim: f32,
            top2_train: String,
            top2_sim: f32,
            top3_train: String,
            top3_sim: f32,
        }

        let mut all_results: Vec<TestResult> = Vec::new();

        let groups: Vec<(&str, &[&str])> = vec![
            ("A_creative", &group_a[..]),
            ("B_close", &group_b[..]),
            ("C_chomsky", &group_c[..]),
        ];

        for (group_name, sentences) in &groups {
            println!("\n--- Group {} ({} sentences) ---", group_name, sentences.len());
            for &sent in *sentences {
                let hv = match composer.encode_sentence(sent) {
                    Some(h) => h,
                    None => {
                        println!("  [SKIP] Could not encode: {}", sent);
                        continue;
                    }
                };

                // Compute similarities to all train sentences
                let mut sims: Vec<(usize, &str, f32)> = train_encoded
                    .iter()
                    .map(|(i, s, thv)| (*i, *s, RealHV::cosine_similarity(&hv, thv)))
                    .collect();
                sims.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

                let top1 = sims.get(0).map(|x| (x.1.to_string(), x.2)).unwrap_or(("?".into(), 0.0));
                let top2 = sims.get(1).map(|x| (x.1.to_string(), x.2)).unwrap_or(("?".into(), 0.0));
                let top3 = sims.get(2).map(|x| (x.1.to_string(), x.2)).unwrap_or(("?".into(), 0.0));

                println!("  \"{}\"", sent);
                println!("    top1: {:.4} \"{}\"", top1.1, &top1.0[..top1.0.len().min(50)]);

                all_results.push(TestResult {
                    group: group_name.to_string(),
                    sentence: sent.to_string(),
                    top1_train: top1.0.clone(),
                    top1_sim: top1.1,
                    top2_train: top2.0.clone(),
                    top2_sim: top2.1,
                    top3_train: top3.0.clone(),
                    top3_sim: top3.1,
                });
            }
        }

        // ------------------------------------------------------------------
        // 5. Compute avg top-1 similarity per group; assert B > A > C
        // ------------------------------------------------------------------
        let avg_group = |grp: &str| -> f32 {
            let vals: Vec<f32> = all_results
                .iter()
                .filter(|r| r.group == grp)
                .map(|r| r.top1_sim)
                .collect();
            if vals.is_empty() { 0.0 } else { vals.iter().sum::<f32>() / vals.len() as f32 }
        };

        let avg_a = avg_group("A_creative");
        let avg_b = avg_group("B_close");
        let avg_c = avg_group("C_chomsky");

        println!("\n--- Group Average Top-1 Similarity ---");
        println!("  Group A (creative):  {:.4}", avg_a);
        println!("  Group B (close):     {:.4}", avg_b);
        println!("  Group C (Chomsky):   {:.4}", avg_c);

        assert!(
            avg_b > avg_a,
            "Group B avg ({:.4}) should be > Group A avg ({:.4})",
            avg_b, avg_a
        );
        assert!(
            avg_a > avg_c,
            "Group A avg ({:.4}) should be > Group C avg ({:.4})",
            avg_a, avg_c
        );
        println!("  [PASS] B ({:.4}) > A ({:.4}) > C ({:.4})", avg_b, avg_a, avg_c);

        // ------------------------------------------------------------------
        // 6. Discrimination: confusable pairs (word order swapped)
        // ------------------------------------------------------------------
        println!("\n--- Discrimination Test (Confusable Pairs) ---");

        // Pairs where subject/object swap changes meaning.
        // Words must be in different distributional clusters to avoid collapse.
        let confusable_pairs: Vec<(&str, &str)> = vec![
            ("il gatto mangia il pesce", "il pesce mangia il gatto"),
            ("il cane insegue il gatto", "il gatto insegue il cane"),
            ("il cuoco prepara la pasta", "la pasta prepara il cuoco"),
            ("il dottore cura il paziente", "il paziente cura il dottore"),
            ("il maestro insegna al bambino", "il bambino insegna al maestro"),
        ];

        let mut disc_results: Vec<(String, String, f32)> = Vec::new();

        for (s1, s2) in &confusable_pairs {
            let hv1 = composer.encode_sentence(s1).expect("encode s1");
            let hv2 = composer.encode_sentence(s2).expect("encode s2");
            let sim = RealHV::cosine_similarity(&hv1, &hv2);
            println!("  \"{s1}\" vs \"{s2}\" => sim = {:.4}", sim);
            disc_results.push((s1.to_string(), s2.to_string(), sim));
        }

        // Assert average discrimination similarity < 0.8
        let avg_disc: f32 = disc_results.iter().map(|x| x.2).sum::<f32>() / disc_results.len() as f32;
        println!("  Average discrimination similarity: {:.4}", avg_disc);

        // Also assert each pair is < 1.0 (not identical) — confirms order matters
        for (s1, s2, sim) in &disc_results {
            assert!(
                *sim < 0.99,
                "Confusable pair should not be nearly identical: \"{s1}\" vs \"{s2}\" => {:.4}",
                sim
            );
        }
        assert!(
            avg_disc < 0.80,
            "Average discrimination similarity should be < 0.8, got {:.4}",
            avg_disc
        );
        println!("  [PASS] Avg discrimination similarity ({:.4}) < 0.80", avg_disc);

        // ------------------------------------------------------------------
        // 7. Save CSV files
        // ------------------------------------------------------------------
        std::fs::create_dir_all("results/experiments").ok();

        // exp5_productivity.csv
        {
            let mut f = File::create("results/experiments/exp5_productivity.csv")
                .expect("create exp5_productivity.csv");
            writeln!(f, "group,sentence,top1_train,top1_sim,top2_train,top2_sim,top3_train,top3_sim").unwrap();
            for r in &all_results {
                writeln!(
                    f,
                    "{},\"{}\",\"{}\",{:.6},\"{}\",{:.6},\"{}\",{:.6}",
                    r.group, r.sentence,
                    r.top1_train, r.top1_sim,
                    r.top2_train, r.top2_sim,
                    r.top3_train, r.top3_sim,
                )
                .unwrap();
            }
            println!("\n  CSV saved: results/experiments/exp5_productivity.csv");
        }

        // exp5_discrimination.csv
        {
            let mut f = File::create("results/experiments/exp5_discrimination.csv")
                .expect("create exp5_discrimination.csv");
            writeln!(f, "sentence_a,sentence_b,similarity").unwrap();
            for (a, b, sim) in &disc_results {
                writeln!(f, "\"{}\",\"{}\",{:.6}", a, b, sim).unwrap();
            }
            println!("  CSV saved: results/experiments/exp5_discrimination.csv");
        }

        // ------------------------------------------------------------------
        // 8. Summary table
        // ------------------------------------------------------------------
        println!("\n{}", "=".repeat(70));
        println!("  SUMMARY — Experiment 5: Productive Compositionality");
        println!("{}", "=".repeat(70));
        println!("  {:25} {:>12} {:>12}", "Metric", "Value", "Status");
        println!("  {:25} {:>12} {:>12}", "-----", "-----", "------");
        println!("  {:25} {:>12.4} {:>12}", "Avg top-1 Group A", avg_a, "");
        println!("  {:25} {:>12.4} {:>12}", "Avg top-1 Group B", avg_b, if avg_b > avg_a { "OK" } else { "FAIL" });
        println!("  {:25} {:>12.4} {:>12}", "Avg top-1 Group C", avg_c, if avg_a > avg_c { "OK" } else { "FAIL" });
        let max_disc = disc_results.iter().map(|x| x.2).fold(0.0f32, f32::max);
        println!("  {:25} {:>12.4} {:>12}", "Max disc. sim", max_disc, if max_disc < 0.8 { "OK" } else { "FAIL" });
        println!("{}", "=".repeat(70));

        println!("\n=== Experiment 5 Complete ===\n");
    }
}
