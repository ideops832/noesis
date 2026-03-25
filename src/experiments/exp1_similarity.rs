//! Experiment 1: Semantic similarity via distributional learning.
//!
//! Expanded protocol:
//! 1. Train vocabulary on expanded + synonym-parallel corpus (5 passes, momentum=0.7, window=3)
//! 2. Measure intra- vs inter-cluster similarity across 5 semantic groups
//! 3. Compare trained vs random baseline
//! 4. Corpus scaling analysis (100, 300, 1000, 3000, all sentences)
//! 5. Export CSV results to results/experiments/

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

    /// Helper: build a trained vocabulary from the given sentences with momentum learning.
    fn build_trained_vocab(sentences: &[Vec<String>], passes: usize, momentum: f32, window: usize) -> Vocabulary {
        let mut vocab = Vocabulary::new(D, 42);
        for sentence in sentences {
            for word in sentence {
                vocab.get_or_create(word);
            }
        }
        for _ in 0..passes {
            vocab.learn_from_context_momentum(sentences, window, momentum);
        }
        vocab
    }

    /// Helper: compute average pairwise similarity among a set of words.
    fn avg_pairwise_similarity(vocab: &Vocabulary, words: &[&str]) -> f32 {
        let mut sum = 0.0f32;
        let mut count = 0u32;
        for i in 0..words.len() {
            for j in (i + 1)..words.len() {
                let sim = vocab.similarity(words[i], words[j]);
                sum += sim;
                count += 1;
            }
        }
        if count == 0 { 0.0 } else { sum / count as f32 }
    }

    /// Helper: compute average similarity between all pairs across two groups.
    fn avg_cross_similarity(vocab: &Vocabulary, group_a: &[&str], group_b: &[&str]) -> f32 {
        let mut sum = 0.0f32;
        let mut count = 0u32;
        for a in group_a {
            for b in group_b {
                sum += vocab.similarity(a, b);
                count += 1;
            }
        }
        if count == 0 { 0.0 } else { sum / count as f32 }
    }

    #[test]
    fn test_exp1_research() {
        println!("\n{}", "=".repeat(60));
        println!("=== Experiment 1: Semantic Similarity — Expanded Protocol ===");
        println!("{}\n", "=".repeat(60));

        let tokenizer = Tokenizer::new();

        // ── 1. Build trained vocabulary ────────────────────────────────────
        println!("--- Phase 1: Build trained vocabulary ---");
        let mut full_corpus = corpus::expanded_corpus();
        full_corpus.extend(corpus::synonym_parallel_corpus());
        full_corpus.extend(corpus::relational_corpus());
        let full_sentences: Vec<Vec<String>> = full_corpus.iter().map(|s| tokenizer.tokenize(s)).collect();

        let vocab = build_trained_vocab(&full_sentences, 5, 0.7, 3);
        println!("  Corpus: {} sentences, Vocab: {} words", full_sentences.len(), vocab.len());
        println!("  Training: 5 passes, momentum=0.7, window=3\n");

        // ── 2. Semantic clusters ───────────────────────────────────────────
        println!("--- Phase 2: Semantic cluster analysis ---\n");

        let clusters: Vec<(&str, Vec<&str>)> = vec![
            ("ANIMALI",  vec!["gatto", "micio", "felino", "cane", "cucciolo", "uccello", "pesce"]),
            ("CIBO",     vec!["pasta", "pane", "cena", "cuoco", "cucina", "torta"]),
            ("NATURA",   vec!["sole", "pioggia", "vento", "neve", "mare", "cielo"]),
            ("FAMIGLIA", vec!["mamma", "papà", "nonna", "nonno", "bambino", "fratello"]),
            ("LAVORO",   vec!["lavoro", "ufficio", "ingegnere", "programmatore", "scrive", "progetta"]),
        ];

        // Filter words that actually exist in the vocabulary
        let clusters_filtered: Vec<(&str, Vec<&str>)> = clusters.iter().map(|(name, words)| {
            let existing: Vec<&str> = words.iter().copied().filter(|w| vocab.words.contains_key(*w)).collect();
            (*name, existing)
        }).collect();

        // Compute intra-cluster and inter-cluster similarities
        struct ClusterResult {
            name: String,
            intra_sim: f32,
            inter_sim: f32,
            ratio: f32,
        }

        let mut cluster_results: Vec<ClusterResult> = Vec::new();

        for (name, words) in &clusters_filtered {
            if words.len() < 2 {
                println!("  [SKIP] Cluster {} has < 2 words in vocab", name);
                continue;
            }

            let intra_sim = avg_pairwise_similarity(&vocab, words);

            // Inter-cluster: average similarity to ALL words in OTHER clusters
            let mut inter_sims = Vec::new();
            for (other_name, other_words) in &clusters_filtered {
                if *other_name == *name || other_words.is_empty() { continue; }
                inter_sims.push(avg_cross_similarity(&vocab, words, other_words));
            }
            let inter_sim = if inter_sims.is_empty() { 0.0 } else {
                inter_sims.iter().sum::<f32>() / inter_sims.len() as f32
            };

            let ratio = if inter_sim.abs() < 1e-9 { f32::INFINITY } else { intra_sim / inter_sim };

            println!("  {:10} | intra={:.4} | inter={:.4} | ratio={:.2} | words={:?}",
                name, intra_sim, inter_sim, ratio, words);

            cluster_results.push(ClusterResult {
                name: name.to_string(),
                intra_sim,
                inter_sim,
                ratio,
            });
        }

        // ── 3. Random baseline comparison ──────────────────────────────────
        println!("\n--- Phase 3: Random baseline comparison ---\n");

        let mut baseline_vocab = Vocabulary::new(D, 999);
        // Create same words but NO training
        for sentence in &full_sentences {
            for word in sentence {
                baseline_vocab.get_or_create(word);
            }
        }

        println!("  {:20} | {:>10} | {:>10}", "Pair", "Trained", "Random");
        println!("  {:-<20}-+-{:-<10}-+-{:-<10}", "", "", "");

        let test_pairs = vec![
            ("gatto", "micio"),
            ("gatto", "felino"),
            ("cane", "cucciolo"),
            ("mamma", "papà"),
            ("sole", "pioggia"),
        ];

        for (a, b) in &test_pairs {
            let trained_sim = vocab.similarity(a, b);
            let random_sim = baseline_vocab.similarity(a, b);
            println!("  {:10}-{:9} | {:>10.4} | {:>10.4}", a, b, trained_sim, random_sim);
        }

        // ── 4. Corpus scaling ──────────────────────────────────────────────
        println!("\n--- Phase 4: Corpus scaling analysis ---\n");

        let scale_sizes = vec![100, 300, 1000, 3000, full_sentences.len()];

        struct ScaleResult {
            n_sentences: usize,
            gatto_micio: f32,
            gatto_felino: f32,
            cane_cucciolo: f32,
        }

        let mut scale_results: Vec<ScaleResult> = Vec::new();

        for &n in &scale_sizes {
            let n_actual = n.min(full_sentences.len());
            let subset = &full_sentences[..n_actual];

            let v = build_trained_vocab(subset, 5, 0.7, 3);

            let gm = v.similarity("gatto", "micio");
            let gf = v.similarity("gatto", "felino");
            let cc = v.similarity("cane", "cucciolo");

            println!("  n={:5} | gatto-micio={:.4} | gatto-felino={:.4} | cane-cucciolo={:.4}",
                n_actual, gm, gf, cc);

            scale_results.push(ScaleResult {
                n_sentences: n_actual,
                gatto_micio: gm,
                gatto_felino: gf,
                cane_cucciolo: cc,
            });
        }

        // ── 5. Save CSVs ──────────────────────────────────────────────────
        println!("\n--- Phase 5: Saving CSV results ---");

        // exp1_clusters.csv
        {
            let mut f = File::create("results/experiments/exp1_clusters.csv")
                .expect("Failed to create exp1_clusters.csv");
            writeln!(f, "cluster,intra_sim,inter_sim,ratio").unwrap();
            for cr in &cluster_results {
                writeln!(f, "{},{:.6},{:.6},{:.4}", cr.name, cr.intra_sim, cr.inter_sim, cr.ratio).unwrap();
            }
            println!("  Saved results/experiments/exp1_clusters.csv");
        }

        // exp1_corpus_scaling.csv
        {
            let mut f = File::create("results/experiments/exp1_corpus_scaling.csv")
                .expect("Failed to create exp1_corpus_scaling.csv");
            writeln!(f, "n_sentences,gatto_micio,gatto_felino,cane_cucciolo").unwrap();
            for sr in &scale_results {
                writeln!(f, "{},{:.6},{:.6},{:.6}", sr.n_sentences, sr.gatto_micio, sr.gatto_felino, sr.cane_cucciolo).unwrap();
            }
            println!("  Saved results/experiments/exp1_corpus_scaling.csv");
        }

        // ── 6. Assertions ─────────────────────────────────────────────────
        println!("\n--- Phase 6: Assertions ---");

        // Count clusters where intra/inter ratio > 2.0
        let good_clusters = cluster_results.iter()
            .filter(|cr| cr.ratio > 2.0)
            .count();

        println!("  Clusters with ratio > 2.0: {}/{}", good_clusters, cluster_results.len());
        for cr in &cluster_results {
            let status = if cr.ratio > 2.0 { "PASS" } else { "FAIL" };
            println!("    [{}] {} ratio={:.2}", status, cr.name, cr.ratio);
        }

        assert!(
            good_clusters >= 3,
            "At least 3 clusters should have intra/inter ratio > 2.0, got {}",
            good_clusters
        );

        println!("\n[PASS] {} clusters have intra/inter ratio > 2.0", good_clusters);
        println!("\n=== Experiment 1 Complete ===\n");
    }
}
