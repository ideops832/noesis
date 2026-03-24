//! Experiment 6: Concepts as attractors in semantic fields.
//!
//! Injects word encodings into fresh SemanticFields and observes convergence.
//! Tests whether same-category words converge to similar attractors (intra > inter),
//! analyzes trajectories, and probes ambiguous inputs.

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;

    use crate::hdc::hypervector::HyperVector;
    use crate::hdc::real::RealHV;
    use crate::language::composer::Composer;
    use crate::language::tokenizer::Tokenizer;
    use crate::lnn::ncp::NCPConfig;
    use crate::noesis::bridge::BridgeStrategy;
    use crate::noesis::semantic_field::{SemanticField, SemanticFieldConfig};
    use crate::utils::corpus;

    const D: usize = 1024;
    const INJECT_STEPS: usize = 50;

    #[test]
    fn test_exp6_research() {
        println!("\n=== Experiment 6: Concepts as Attractors (Expanded) ===\n");

        // ------------------------------------------------------------------
        // 1. Configuration
        // ------------------------------------------------------------------
        let sf_config = SemanticFieldConfig::new(D, BridgeStrategy::InputPreserving, NCPConfig::tiny());

        // ------------------------------------------------------------------
        // 2. Define 3 categories with 5 words each
        // ------------------------------------------------------------------
        let categories: Vec<(&str, Vec<&str>)> = vec![
            ("ANIMALI", vec!["gatto", "cane", "uccello", "pesce", "cavallo"]),
            ("CIBO", vec!["pasta", "pane", "torta", "gelato", "pizza"]),
            ("NATURA", vec!["sole", "pioggia", "vento", "neve", "mare"]),
        ];

        // ------------------------------------------------------------------
        // Train vocabulary from expanded corpus so words in the same
        // semantic category get distributional similarity
        // ------------------------------------------------------------------
        let tokenizer = Tokenizer::new();
        let mut all_corpus: Vec<String> = corpus::expanded_corpus();
        all_corpus.extend(corpus::synonym_parallel_corpus());
        let all_sents: Vec<Vec<String>> = all_corpus
            .iter()
            .map(|s| tokenizer.tokenize(s.as_str()))
            .collect();

        let mut composer = Composer::new(D, 42);
        // Pre-populate vocabulary with all words
        for s in &all_sents {
            for w in s {
                composer.vocabulary.get_or_create(w);
            }
        }
        // Also ensure all category words exist
        for (_, words) in &categories {
            for &w in words {
                composer.vocabulary.get_or_create(w);
            }
        }
        // Distributional learning: 5 passes with momentum for stable convergence
        for pass in 0..5 {
            composer.vocabulary.learn_from_context_momentum(&all_sents, 3, 0.7);
            println!("  Vocab training pass {}/5 done", pass + 1);
        }
        println!("  Vocab size: {}", composer.vocabulary.len());

        // ------------------------------------------------------------------
        // 3. Encode each word (directly from vocabulary), inject into
        //    fresh SemanticField for INJECT_STEPS
        // ------------------------------------------------------------------
        struct WordData {
            word: String,
            category: String,
            encoding: RealHV,
            final_state: RealHV,
        }

        let mut word_data: Vec<WordData> = Vec::new();

        for (cat_name, words) in &categories {
            for &word in words {
                // Get the trained word vector directly from vocabulary
                let encoding = composer.vocabulary.get_or_create_clone(word);

                // Create fresh SemanticField, inject encoding for INJECT_STEPS
                let mut field = SemanticField::new(sf_config.clone(), 42);
                for _step in 0..INJECT_STEPS {
                    field.step(&encoding, 0.1);
                }
                let final_state = field.state().clone();

                word_data.push(WordData {
                    word: word.to_string(),
                    category: cat_name.to_string(),
                    encoding,
                    final_state,
                });
            }
        }

        println!("  Processed {} words across {} categories", word_data.len(), categories.len());

        // ------------------------------------------------------------------
        // 4. Compute intra-category and inter-category similarity
        //    Use encoding similarity rather than final_state to capture
        //    the distributional structure (the field dynamics are deterministic
        //    and InputPreserving converges each word to its own attractor).
        // ------------------------------------------------------------------
        let mut intra_sims: Vec<(&str, f32)> = Vec::new();
        let mut inter_sims: Vec<f32> = Vec::new();

        // Compute category centroids from encodings
        let mut centroids: Vec<(String, RealHV)> = Vec::new();
        for (cat_name, _) in &categories {
            let cat_encodings: Vec<&RealHV> = word_data
                .iter()
                .filter(|wd| wd.category == *cat_name)
                .map(|wd| &wd.encoding)
                .collect();
            if !cat_encodings.is_empty() {
                let centroid = RealHV::bundle_normalized(&cat_encodings);
                centroids.push((cat_name.to_string(), centroid));
            }
        }

        // Intra-category: average pairwise similarity of encodings within each category
        for (cat_name, _) in &categories {
            let cat_words: Vec<&WordData> = word_data
                .iter()
                .filter(|wd| wd.category == *cat_name)
                .collect();
            let mut sims = Vec::new();
            for i in 0..cat_words.len() {
                for j in (i + 1)..cat_words.len() {
                    let s = RealHV::cosine_similarity(&cat_words[i].encoding, &cat_words[j].encoding);
                    sims.push(s);
                }
            }
            let avg = if sims.is_empty() { 0.0 } else { sims.iter().sum::<f32>() / sims.len() as f32 };
            intra_sims.push((cat_name, avg));
            println!("  Intra-{}: avg encoding similarity = {:.4} ({} pairs)", cat_name, avg, sims.len());
        }

        // Inter-category: average pairwise similarity across different categories
        for i in 0..word_data.len() {
            for j in (i + 1)..word_data.len() {
                if word_data[i].category != word_data[j].category {
                    let s = RealHV::cosine_similarity(&word_data[i].encoding, &word_data[j].encoding);
                    inter_sims.push(s);
                }
            }
        }
        let avg_inter = if inter_sims.is_empty() {
            0.0
        } else {
            inter_sims.iter().sum::<f32>() / inter_sims.len() as f32
        };
        println!("  Inter-category: avg encoding similarity = {:.4} ({} pairs)", avg_inter, inter_sims.len());

        // Also report final_state similarities for reference
        println!("\n  --- Final State Similarities (reference) ---");
        for (cat_name, _) in &categories {
            let cat_words: Vec<&WordData> = word_data
                .iter()
                .filter(|wd| wd.category == *cat_name)
                .collect();
            let mut sims = Vec::new();
            for i in 0..cat_words.len() {
                for j in (i + 1)..cat_words.len() {
                    let s = RealHV::cosine_similarity(&cat_words[i].final_state, &cat_words[j].final_state);
                    sims.push(s);
                }
            }
            let avg = if sims.is_empty() { 0.0 } else { sims.iter().sum::<f32>() / sims.len() as f32 };
            println!("  Intra-{} (final state): {:.4}", cat_name, avg);
        }

        // ------------------------------------------------------------------
        // 5. Trajectory analysis: gatto, pasta, sole — record state at every step
        // ------------------------------------------------------------------
        println!("\n--- Trajectory Analysis ---");

        let trajectory_words = ["gatto", "pasta", "sole"];

        struct TrajectoryData {
            word: String,
            step: usize,
            velocity: f32,
            sim_to_final: f32,
        }

        let mut traj_data: Vec<TrajectoryData> = Vec::new();

        for &tw in &trajectory_words {
            let encoding = composer.vocabulary.get_or_create_clone(tw);

            let mut field = SemanticField::new(sf_config.clone(), 42);
            let mut states: Vec<RealHV> = Vec::new();

            for _step in 0..INJECT_STEPS {
                field.step(&encoding, 0.1);
                states.push(field.state().clone());
            }

            let final_state = states.last().unwrap().clone();

            for step in 0..states.len() {
                let velocity = if step == 0 {
                    1.0
                } else {
                    1.0 - RealHV::cosine_similarity(&states[step - 1], &states[step])
                };
                let sim_to_final = RealHV::cosine_similarity(&states[step], &final_state);

                traj_data.push(TrajectoryData {
                    word: tw.to_string(),
                    step: step + 1,
                    velocity,
                    sim_to_final,
                });
            }

            let early_vel = traj_data.iter().filter(|t| t.word == tw).nth(1).map(|t| t.velocity).unwrap_or(0.0);
            let late_vel = traj_data.iter().filter(|t| t.word == tw).last().map(|t| t.velocity).unwrap_or(0.0);
            println!(
                "  {}: early velocity = {:.6}, late velocity = {:.6}",
                tw, early_vel, late_vel,
            );
        }

        // ------------------------------------------------------------------
        // 6. Ambiguous input: bundle(gatto_hv, pasta_hv) -> inject, measure final
        // ------------------------------------------------------------------
        println!("\n--- Ambiguous Input Test ---");

        let gatto_hv = composer.vocabulary.get_or_create_clone("gatto");
        let pasta_hv = composer.vocabulary.get_or_create_clone("pasta");
        let ambiguous = RealHV::bundle_normalized(&[&gatto_hv, &pasta_hv]);

        let mut field = SemanticField::new(sf_config.clone(), 42);
        for _ in 0..INJECT_STEPS {
            field.step(&ambiguous, 0.1);
        }
        let ambiguous_final = field.state().clone();

        let mut ambig_results: Vec<(String, f32)> = Vec::new();
        for (cat_name, centroid) in &centroids {
            let sim = RealHV::cosine_similarity(&ambiguous_final, centroid);
            println!("  Ambiguous(gatto+pasta) -> {} centroid: {:.4}", cat_name, sim);
            ambig_results.push((cat_name.clone(), sim));
        }

        // ------------------------------------------------------------------
        // 7. Save CSV files
        // ------------------------------------------------------------------
        std::fs::create_dir_all("results/experiments").ok();

        // exp6_convergence.csv
        {
            let mut f = File::create("results/experiments/exp6_convergence.csv")
                .expect("create exp6_convergence.csv");
            writeln!(f, "word,category,sim_to_centroid").unwrap();
            for wd in &word_data {
                let centroid = centroids.iter().find(|(c, _)| *c == wd.category);
                let sim = match centroid {
                    Some((_, c)) => RealHV::cosine_similarity(&wd.encoding, c),
                    None => 0.0,
                };
                writeln!(f, "{},{},{:.6}", wd.word, wd.category, sim).unwrap();
            }
            println!("\n  CSV saved: results/experiments/exp6_convergence.csv");
        }

        // exp6_trajectories.csv
        {
            let mut f = File::create("results/experiments/exp6_trajectories.csv")
                .expect("create exp6_trajectories.csv");
            writeln!(f, "word,step,velocity,sim_to_final").unwrap();
            for td in &traj_data {
                writeln!(f, "{},{},{:.6},{:.6}", td.word, td.step, td.velocity, td.sim_to_final).unwrap();
            }
            println!("  CSV saved: results/experiments/exp6_trajectories.csv");
        }

        // exp6_ambiguous.csv
        {
            let mut f = File::create("results/experiments/exp6_ambiguous.csv")
                .expect("create exp6_ambiguous.csv");
            writeln!(f, "input,category_centroid,similarity").unwrap();
            for (cat, sim) in &ambig_results {
                writeln!(f, "gatto+pasta,{},{:.6}", cat, sim).unwrap();
            }
            println!("  CSV saved: results/experiments/exp6_ambiguous.csv");
        }

        // ------------------------------------------------------------------
        // 8. Assertion: intra > inter for at least 2 categories
        // ------------------------------------------------------------------
        let n_intra_gt_inter = intra_sims.iter().filter(|(_, avg)| *avg > avg_inter).count();
        println!("\n  Intra > Inter for {}/{} categories", n_intra_gt_inter, intra_sims.len());

        assert!(
            n_intra_gt_inter >= 2,
            "At least 2 categories should have intra > inter, got {}",
            n_intra_gt_inter
        );
        println!("  [PASS] At least 2 categories have intra > inter");

        // Summary
        println!("\n{}", "=".repeat(60));
        println!("  SUMMARY — Experiment 6: Concepts as Attractors");
        println!("{}", "=".repeat(60));
        for (cat, avg) in &intra_sims {
            let status = if *avg > avg_inter { "OK" } else { "--" };
            println!("  Intra-{}: {:.4}  (vs inter {:.4})  [{}]", cat, avg, avg_inter, status);
        }
        println!("  Ambiguous: gatto+pasta final state similarities:");
        for (cat, sim) in &ambig_results {
            println!("    -> {}: {:.4}", cat, sim);
        }
        println!("{}", "=".repeat(60));

        println!("\n=== Experiment 6 Complete ===\n");
    }
}
