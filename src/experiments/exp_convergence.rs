//! Experiment: Vocabulary convergence with 5→30 passes.
//!
//! Tests the hypothesis that distributional similarity at D=1024
//! requires 20+ passes of momentum learning to emerge above noise.
//! Runs an incremental sweep: train 5 passes, measure, add 5 more, etc.
//! Stops early if semantic_gap > 0.3.

#[cfg(test)]
mod tests {
    use std::fs::{create_dir_all, File};
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::Instant;

    use crate::hdc::real::RealHV;
    use crate::language::corpus_loader::{CorpusConfig, EnglishTokenizer};
    use crate::language::vocabulary::Vocabulary;
    use crate::noesis::deliberation::{
        generate_vocabulary_candidates, DeliberationConfig,
    };
    use crate::noesis::volition::{GoalLevel, VolitionLayer};

    const D: usize = 1024;
    const RESULTS_DIR: &str = "results/convergence";
    const CORPUS_BASE: &str = "/home/gabriele/Progetti/hermes/data/corpus_balanced";

    fn ensure_dir() { let _ = create_dir_all(RESULTS_DIR); }

    /// Synonym pairs expected to be close.
    const SYNONYM_PAIRS: &[(&str, &str)] = &[
        ("happy", "joyful"),
        ("fast", "quick"),
        ("large", "big"),
        ("begin", "start"),
        ("dark", "night"),
        ("beautiful", "lovely"),
    ];

    /// Unrelated pairs expected to be distant.
    const UNRELATED_PAIRS: &[(&str, &str)] = &[
        ("cat", "mathematics"),
        ("house", "philosophy"),
        ("water", "politics"),
    ];

    /// Load corpus sentences (tokenized) from narrative slot.
    fn load_corpus_sentences() -> Vec<Vec<String>> {
        let tok = EnglishTokenizer::new();
        let dir = format!("{}/slot_narrative", CORPUS_BASE);
        let mut all_sentences = Vec::new();

        if let Ok(entries) = std::fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map(|e| e == "txt").unwrap_or(false) {
                    if let Ok(text) = std::fs::read_to_string(&path) {
                        let capped: String = text.chars().take(2_000_000).collect();
                        let sents = tok.tokenize_into_sentences(&capped);
                        all_sentences.extend(sents);
                    }
                }
            }
        }
        all_sentences
    }

    /// Filter sentences to only words with freq >= min_freq.
    fn filter_by_frequency(
        sentences: &[Vec<String>],
        min_freq: usize,
    ) -> (Vec<Vec<String>>, std::collections::HashSet<String>) {
        let mut freq: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for sent in sentences {
            for w in sent {
                *freq.entry(w.clone()).or_insert(0) += 1;
            }
        }
        let allowed: std::collections::HashSet<String> = freq
            .into_iter()
            .filter(|(_, f)| *f >= min_freq)
            .map(|(w, _)| w)
            .collect();
        let filtered = sentences
            .iter()
            .map(|s| s.iter().filter(|w| allowed.contains(w.as_str())).cloned().collect())
            .filter(|s: &Vec<String>| s.len() >= 2)
            .collect();
        (filtered, allowed)
    }

    /// Measure average similarity for a set of word pairs.
    fn avg_pair_sim(vocab: &Vocabulary, pairs: &[(&str, &str)]) -> (f32, usize) {
        let mut sum = 0.0f32;
        let mut n = 0;
        for (a, b) in pairs {
            if let (Some(ha), Some(hb)) = (vocab.words.get(*a), vocab.words.get(*b)) {
                sum += RealHV::cosine_similarity(ha, hb);
                n += 1;
            }
        }
        if n > 0 { (sum / n as f32, n) } else { (0.0, 0) }
    }

    /// Vocabulary-candidates score: use an in-vocab goal.
    fn vocab_candidates_score(vocab: &Vocabulary) -> f32 {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(0xBEEF);

        let n_trials = 10;
        let mut sum = 0.0f32;
        let goal_words: Vec<&String> = vocab.words.keys().take(n_trials).collect();

        for gw in &goal_words {
            let goal = vocab.words.get(*gw).unwrap();
            let noise = RealHV::random_normal(D, &mut rng);
            let fast = RealHV::add(goal, &noise.scale(0.3)).normalized();
            let cands = generate_vocabulary_candidates(&fast, vocab, 5);
            let best = cands.iter()
                .map(|(hv, _)| RealHV::cosine_similarity(hv, goal).max(0.0))
                .fold(0.0f32, f32::max);
            sum += best;
        }
        sum / n_trials as f32
    }

    #[test]
    fn test_convergence_sweep() {
        println!("\n=== Experiment: Vocabulary Convergence Sweep ===\n");
        ensure_dir();

        // 1. Load + filter corpus.
        eprintln!("Loading corpus...");
        let raw_sentences = load_corpus_sentences();
        eprintln!("  {} raw sentences", raw_sentences.len());

        let (filtered, allowed) = filter_by_frequency(&raw_sentences, 5);
        eprintln!("  {} filtered sentences, {} allowed words", filtered.len(), allowed.len());

        // 2. Create vocabulary, seed all allowed words.
        let mut vocab = Vocabulary::new(D, 42);
        for w in &allowed {
            vocab.get_or_create(w);
        }
        // Trim to 10k by frequency if needed.
        if vocab.len() > 10_000 {
            let mut freq: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
            for sent in &filtered {
                for w in sent { *freq.entry(w.clone()).or_insert(0) += 1; }
            }
            let mut by_freq: Vec<(String, usize)> = freq.into_iter().collect();
            by_freq.sort_by(|a, b| b.1.cmp(&a.1));
            let keep: std::collections::HashSet<String> = by_freq.into_iter().take(10_000).map(|(w, _)| w).collect();
            vocab.words.retain(|w, _| keep.contains(w));
        }
        eprintln!("  Vocabulary: {} words", vocab.len());

        // 3. Incremental sweep.
        let checkpoints = [5, 10, 15, 20, 25, 30];
        let passes_per_step = 5;
        let start = Instant::now();

        let csv_path = format!("{}/passes_sweep.csv", RESULTS_DIR);
        let mut csv = File::create(&csv_path).unwrap();
        writeln!(
            csv,
            "passes,avg_sim_synonyms,avg_sim_unrelated,semantic_gap,\
             vocab_candidates_score,training_time_cumulative_secs,\
             n_synonym_pairs,n_unrelated_pairs"
        ).unwrap();

        let mut best_gap = 0.0f32;
        let mut best_passes = 0usize;
        let mut best_vocab_score = 0.0f32;

        for &target_passes in &checkpoints {
            // Train incrementally: 5 more passes per step.
            for _ in 0..passes_per_step {
                vocab.learn_from_context_momentum(&filtered, 3, 0.7);
            }

            let elapsed = start.elapsed().as_secs_f32();

            // Measure.
            let (avg_syn, n_syn) = avg_pair_sim(&vocab, SYNONYM_PAIRS);
            let (avg_unr, n_unr) = avg_pair_sim(&vocab, UNRELATED_PAIRS);
            let gap = avg_syn - avg_unr;
            let vc_score = vocab_candidates_score(&vocab);

            println!(
                "passes={:2}  syn={:.4}({}) unr={:.4}({}) gap={:.4}  vc={:.4}  ({:.1}s)",
                target_passes, avg_syn, n_syn, avg_unr, n_unr, gap, vc_score, elapsed
            );

            writeln!(
                csv,
                "{},{:.6},{:.6},{:.6},{:.6},{:.1},{},{}",
                target_passes, avg_syn, avg_unr, gap, vc_score, elapsed, n_syn, n_unr
            ).unwrap();

            if gap > best_gap { best_gap = gap; best_passes = target_passes; }
            if vc_score > best_vocab_score { best_vocab_score = vc_score; }

            // Save checkpoint vocab.
            let vpath = format!("data/vocabulary_{}p.vocab", target_passes);
            let _ = vocab.save(std::path::Path::new(&vpath));

            // Early stop.
            if gap > 0.3 {
                println!("  *** Early stop: gap > 0.3 at {} passes ***", target_passes);
                break;
            }
        }

        // Save best vocab.
        let _ = vocab.save(std::path::Path::new("data/vocabulary_best.vocab"));

        // Per-pair direction check: for each synonym pair (a,b), count how
        // many have sim > 0.1 (above noise). This is more robust than aggregate
        // gap which is distorted by high-frequency unrelated words (e.g.
        // "cat" from Alice in Wonderland).
        let (final_syn, n_syn) = avg_pair_sim(&vocab, SYNONYM_PAIRS);
        let n_above_01: usize = SYNONYM_PAIRS.iter()
            .filter_map(|(a, b)| {
                let ha = vocab.words.get(*a)?;
                let hb = vocab.words.get(*b)?;
                let s = RealHV::cosine_similarity(ha, hb);
                if s > 0.1 { Some(()) } else { None }
            })
            .count();

        // Verdict based on individual pair strength.
        let strong = n_above_01 >= 4 && final_syn > 0.15;
        let moderate = n_above_01 >= 3 && final_syn > 0.08;
        let verdict = if strong { "STRONG" }
            else if moderate { "MODERATE" }
            else { "WEAK" };

        let smp = format!("{}/summary.md", RESULTS_DIR);
        let mut sm = File::create(&smp).unwrap();
        writeln!(sm, "# Vocabulary Convergence — Sweep Summary").unwrap();
        writeln!(sm).unwrap();
        writeln!(sm, "## Results").unwrap();
        writeln!(sm, "- synonym pairs with sim > 0.1: {}/{}", n_above_01, SYNONYM_PAIRS.len()).unwrap();
        writeln!(sm, "- avg synonym sim at {} passes: {:.4}", checkpoints.last().unwrap_or(&30), final_syn).unwrap();
        writeln!(sm, "- best semantic gap (aggregate): {:.4} at {} passes", best_gap, best_passes).unwrap();
        writeln!(sm, "- vocabulary size: {}", vocab.len()).unwrap();
        writeln!(sm).unwrap();

        // Detailed synonym pairs at best checkpoint.
        writeln!(sm, "## Synonym pairs (at {} passes)", best_passes).unwrap();
        for (a, b) in SYNONYM_PAIRS {
            if let (Some(ha), Some(hb)) = (vocab.words.get(*a), vocab.words.get(*b)) {
                writeln!(sm, "- {} / {} : {:.4}", a, b, RealHV::cosine_similarity(ha, hb)).unwrap();
            }
        }
        for (a, b) in UNRELATED_PAIRS {
            if let (Some(ha), Some(hb)) = (vocab.words.get(*a), vocab.words.get(*b)) {
                writeln!(sm, "- {} / {} : {:.4} (unrelated)", a, b, RealHV::cosine_similarity(ha, hb)).unwrap();
            }
        }
        writeln!(sm).unwrap();
        writeln!(sm, "## Verdict: **{}**", verdict).unwrap();
        writeln!(sm).unwrap();
        writeln!(sm, "Criteria:").unwrap();
        writeln!(sm, "- STRONG: ≥ 4/6 synonym pairs with sim > 0.1 AND avg_syn > 0.15").unwrap();
        writeln!(sm, "- MODERATE: ≥ 3/6 synonym pairs with sim > 0.1 AND avg_syn > 0.08").unwrap();
        writeln!(sm, "- WEAK: otherwise").unwrap();

        println!("\nVerdict: {}", verdict);
        println!("Best gap: {:.4} at {} passes", best_gap, best_passes);
        println!("Best vc_score: {:.4}", best_vocab_score);

        assert!(std::path::Path::new(&csv_path).exists());
        assert!(std::path::Path::new(&smp).exists());
    }
}
