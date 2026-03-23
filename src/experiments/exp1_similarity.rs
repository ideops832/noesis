//! Experiment 1: Emergent semantic similarity.
//!
//! Builds a vocabulary from the Italian corpus using distributional learning
//! and verifies that words appearing in similar contexts become more similar
//! in hypervector space.

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;

    use crate::hdc::hypervector::HyperVector;
    use crate::language::tokenizer::Tokenizer;
    use crate::language::vocabulary::Vocabulary;
    use crate::utils::corpus::italian_corpus;

    const D: usize = 1024;

    #[test]
    fn test_exp1_similarity() {
        println!("\n=== Experiment 1: Semantic Similarity Emergence ===\n");

        let tokenizer = Tokenizer::new();
        let corpus = italian_corpus();
        println!("Corpus size: {} sentences", corpus.len());

        // Tokenize corpus
        let sentences: Vec<Vec<String>> = corpus
            .iter()
            .map(|s| tokenizer.tokenize(s))
            .collect();

        // Build vocabulary with distributional learning
        let mut vocab = Vocabulary::new(D, 42);

        // Ensure all words exist first
        for sentence in &sentences {
            for word in sentence {
                vocab.get_or_create(word);
            }
        }
        println!("Vocabulary size: {} words", vocab.len());

        // 3 passes of contextual learning with window=3
        for pass in 0..3 {
            vocab.learn_from_context(&sentences, 3);
            println!("Learning pass {} complete", pass + 1);
        }

        // Print top-20 most similar word pairs
        println!("\n--- Top 20 Most Similar Word Pairs ---");
        let words: Vec<String> = vocab.words.keys().cloned().collect();
        let mut all_pairs: Vec<(String, String, f32)> = Vec::new();

        for i in 0..words.len() {
            for j in (i + 1)..words.len() {
                let sim = vocab.similarity(&words[i], &words[j]);
                all_pairs.push((words[i].clone(), words[j].clone(), sim));
            }
        }
        all_pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        for (rank, (w1, w2, sim)) in all_pairs.iter().take(20).enumerate() {
            println!("  {:2}. {:15} - {:15} : {:.4}", rank + 1, w1, w2, sim);
        }

        // Core assertion: gatto-felino should be more similar than gatto-treno
        let sim_gatto_felino = vocab.similarity("gatto", "felino");
        let sim_gatto_treno = vocab.similarity("gatto", "treno");
        let sim_gatto_micio = vocab.similarity("gatto", "micio");
        let sim_cane_cucciolo = vocab.similarity("cane", "cucciolo");

        println!("\n--- Key Similarity Pairs ---");
        println!("  gatto - felino:   {:.4}", sim_gatto_felino);
        println!("  gatto - micio:    {:.4}", sim_gatto_micio);
        println!("  cane  - cucciolo: {:.4}", sim_cane_cucciolo);
        println!("  gatto - treno:    {:.4}", sim_gatto_treno);

        // Most similar to 'gatto'
        let most_sim = vocab.most_similar("gatto", 10);
        println!("\n--- Most Similar to 'gatto' ---");
        for (word, sim) in &most_sim {
            println!("  {:15} : {:.4}", word, sim);
        }

        // --- CSV Export ---
        // Top-100 similarity matrix
        {
            let mut f = File::create("results/exp1_similarity_matrix.csv")
                .expect("Failed to create exp1_similarity_matrix.csv");
            writeln!(f, "word_a,word_b,similarity").unwrap();
            for (w1, w2, sim) in all_pairs.iter().take(100) {
                writeln!(f, "{},{},{:.6}", w1, w2, sim).unwrap();
            }
        }

        // Top-20 pairs
        {
            let mut f = File::create("results/exp1_top_pairs.csv")
                .expect("Failed to create exp1_top_pairs.csv");
            writeln!(f, "rank,word_a,word_b,similarity").unwrap();
            for (rank, (w1, w2, sim)) in all_pairs.iter().take(20).enumerate() {
                writeln!(f, "{},{},{},{:.6}", rank + 1, w1, w2, sim).unwrap();
            }
        }

        // Key similarity pairs
        {
            let mut f = File::create("results/exp1_key_pairs.csv")
                .expect("Failed to create exp1_key_pairs.csv");
            writeln!(f, "word_a,word_b,similarity").unwrap();
            writeln!(f, "gatto,felino,{:.6}", sim_gatto_felino).unwrap();
            writeln!(f, "gatto,micio,{:.6}", sim_gatto_micio).unwrap();
            writeln!(f, "cane,cucciolo,{:.6}", sim_cane_cucciolo).unwrap();
            writeln!(f, "gatto,treno,{:.6}", sim_gatto_treno).unwrap();
        }

        println!("\n[CSV] Saved results/exp1_similarity_matrix.csv, exp1_top_pairs.csv, exp1_key_pairs.csv");

        assert!(
            sim_gatto_felino > sim_gatto_treno,
            "gatto-felino ({:.4}) should be more similar than gatto-treno ({:.4})",
            sim_gatto_felino,
            sim_gatto_treno
        );

        println!("\n[PASS] gatto-felino ({:.4}) > gatto-treno ({:.4})", sim_gatto_felino, sim_gatto_treno);
        println!("\n=== Experiment 1 Complete ===\n");
    }

    #[test]
    fn test_exp1_expanded_corpus() {
        println!("\n=== Experiment 1b: Expanded Corpus ({} sentences) ===\n",
            crate::utils::corpus::expanded_corpus().len());

        let tokenizer = Tokenizer::new();
        let expanded = crate::utils::corpus::expanded_corpus();
        let sentences: Vec<Vec<String>> = expanded
            .iter()
            .map(|s| tokenizer.tokenize(s))
            .collect();

        let mut vocab = Vocabulary::new(2048, 42);
        for sentence in &sentences {
            for word in sentence {
                vocab.get_or_create(word);
            }
        }
        println!("Vocabulary size: {} words", vocab.len());

        // 5 passes with window=3
        for pass in 0..5 {
            vocab.learn_from_context(&sentences, 3);
            if (pass + 1) % 2 == 0 {
                println!("Pass {} complete", pass + 1);
            }
        }

        let sim_gatto_felino = vocab.similarity("gatto", "felino");
        let sim_gatto_micio = vocab.similarity("gatto", "micio");
        let sim_cane_cucciolo = vocab.similarity("cane", "cucciolo");
        let sim_gatto_treno = vocab.similarity("gatto", "treno");
        let sim_gatto_pasta = vocab.similarity("gatto", "pasta");
        let sim_mamma_papa = vocab.similarity("mamma", "papà");

        println!("\n--- Key Pairs (expanded corpus) ---");
        println!("  gatto - felino:   {:.4}", sim_gatto_felino);
        println!("  gatto - micio:    {:.4}", sim_gatto_micio);
        println!("  cane  - cucciolo: {:.4}", sim_cane_cucciolo);
        println!("  mamma - papà:     {:.4}", sim_mamma_papa);
        println!("  gatto - treno:    {:.4}", sim_gatto_treno);
        println!("  gatto - pasta:    {:.4}", sim_gatto_pasta);

        // Most similar to gatto
        let most_sim = vocab.most_similar("gatto", 10);
        println!("\n--- Most Similar to 'gatto' ---");
        for (word, sim) in &most_sim {
            println!("  {:15} : {:.4}", word, sim);
        }

        // Save CSV
        {
            let mut f = File::create("results/exp1b_expanded_key_pairs.csv")
                .expect("create csv");
            writeln!(f, "word_a,word_b,similarity").unwrap();
            writeln!(f, "gatto,felino,{:.6}", sim_gatto_felino).unwrap();
            writeln!(f, "gatto,micio,{:.6}", sim_gatto_micio).unwrap();
            writeln!(f, "cane,cucciolo,{:.6}", sim_cane_cucciolo).unwrap();
            writeln!(f, "mamma,papà,{:.6}", sim_mamma_papa).unwrap();
            writeln!(f, "gatto,treno,{:.6}", sim_gatto_treno).unwrap();
            writeln!(f, "gatto,pasta,{:.6}", sim_gatto_pasta).unwrap();
        }

        assert!(
            sim_gatto_felino > sim_gatto_treno,
            "gatto-felino ({:.4}) should exceed gatto-treno ({:.4})",
            sim_gatto_felino, sim_gatto_treno
        );

        println!("\n[PASS] Expanded corpus: gatto-felino ({:.4}) > gatto-treno ({:.4})",
            sim_gatto_felino, sim_gatto_treno);
        println!("=== Experiment 1b Complete ===\n");
    }
}
