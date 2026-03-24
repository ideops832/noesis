//! Experiment 7: Emergent timescales in multi-scale semantic fields.
//!
//! Processes themed sentence sequences through a MultiScaleField,
//! tracking fast/slow/combined state similarities to theme centroids.
//! Compares multi-scale vs single-scale retention of early inputs.

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
    use crate::noesis::multiscale::{MultiScaleConfig, MultiScaleField};
    use crate::noesis::semantic_field::{SemanticField, SemanticFieldConfig};
    use crate::utils::corpus;

    const D: usize = 1024;

    #[test]
    fn test_exp7_research() {
        println!("\n=== Experiment 7: Emergent Timescales (Expanded) ===\n");

        // ------------------------------------------------------------------
        // 1. Create MultiScaleField with default config, D=1024
        // ------------------------------------------------------------------
        let ms_config = MultiScaleConfig::default_for_dim(D);
        let mut ms_field = MultiScaleField::new(ms_config.clone(), 42);

        // ------------------------------------------------------------------
        // 2. Create Composer with trained vocab
        // ------------------------------------------------------------------
        let tokenizer = Tokenizer::new();
        let expanded = corpus::expanded_corpus();
        let expanded_sents: Vec<Vec<String>> = expanded
            .iter()
            .map(|s| tokenizer.tokenize(s.as_str()))
            .collect();

        let mut composer = Composer::new(D, 42);
        for s in &expanded_sents {
            for w in s {
                composer.vocabulary.get_or_create(w);
            }
        }
        for _ in 0..3 {
            composer.vocabulary.learn_from_context(&expanded_sents, 3);
        }
        println!("  Vocab trained: {} words", composer.vocabulary.len());

        // ------------------------------------------------------------------
        // 3. Define 20 sentences with topic changes
        // ------------------------------------------------------------------
        let themes: Vec<(&str, Vec<&str>)> = vec![
            ("animals", vec![
                "il gatto dorme",
                "il cane corre",
                "il cavallo salta",
                "il pesce nuota",
                "il uccello vola",
            ]),
            ("food", vec![
                "la pasta cuoce",
                "il cuoco prepara",
                "il pane lievita",
                "la torta cuoce",
                "il gelato si scioglie",
            ]),
            ("work", vec![
                "il programmatore scrive",
                "il dottore visita",
                "il maestro insegna",
                "il direttore decide",
                "il ricercatore studia",
            ]),
            ("nature", vec![
                "il sole splende",
                "la pioggia cade",
                "il vento soffia",
                "la neve fiocca",
                "il mare ondeggia",
            ]),
        ];

        // Flatten all sentences in order
        let all_sentences: Vec<(&str, &str)> = themes
            .iter()
            .flat_map(|(theme, sents)| sents.iter().map(move |&s| (*theme, s)))
            .collect();
        assert_eq!(all_sentences.len(), 20);

        // Encode all sentences
        let encoded_sentences: Vec<(&str, &str, RealHV)> = all_sentences
            .iter()
            .filter_map(|&(theme, sent)| {
                composer.encode_sentence(sent).map(|hv| (theme, sent, hv))
            })
            .collect();

        // ------------------------------------------------------------------
        // Compute theme centroids (bundle of theme sentences' encodings)
        // ------------------------------------------------------------------
        let theme_names: Vec<&str> = themes.iter().map(|(n, _)| *n).collect();
        let mut theme_centroids: Vec<(&str, RealHV)> = Vec::new();
        for &tn in &theme_names {
            let hvs: Vec<&RealHV> = encoded_sentences
                .iter()
                .filter(|(t, _, _)| *t == tn)
                .map(|(_, _, hv)| hv)
                .collect();
            if !hvs.is_empty() {
                let centroid = RealHV::bundle_normalized(&hvs);
                theme_centroids.push((tn, centroid));
            }
        }
        println!("  Theme centroids computed: {}", theme_centroids.len());

        // ------------------------------------------------------------------
        // 4. Process sentences, record fast/slow/combined state similarities
        // ------------------------------------------------------------------
        struct ThemeTrackRow {
            step: usize,
            theme: String,
            fast_sim: f32,
            slow_sim: f32,
            combined_sim: f32,
        }

        let mut tracking: Vec<ThemeTrackRow> = Vec::new();

        // Also record the encoding of sentence 1 for retention test
        let sentence1_hv = encoded_sentences.first().map(|(_, _, hv)| hv.clone());

        for (step, (theme, sent, hv)) in encoded_sentences.iter().enumerate() {
            ms_field.step(hv);

            // Record similarities to each theme centroid
            for (tn, centroid) in &theme_centroids {
                let fast_sim = RealHV::cosine_similarity(ms_field.fast_state(), centroid);
                let slow_sim = RealHV::cosine_similarity(ms_field.slow_state(), centroid);
                let combined_sim = RealHV::cosine_similarity(ms_field.combined_state(), centroid);

                tracking.push(ThemeTrackRow {
                    step: step + 1,
                    theme: tn.to_string(),
                    fast_sim,
                    slow_sim,
                    combined_sim,
                });
            }

            if step % 5 == 0 || step == 19 {
                println!("  Step {:2} [{}] \"{}\"", step + 1, theme, sent);
                println!("    fast->current_theme:  {:.4}",
                    RealHV::cosine_similarity(ms_field.fast_state(),
                        &theme_centroids.iter().find(|(t, _)| t == theme).unwrap().1));
                println!("    slow->current_theme:  {:.4}",
                    RealHV::cosine_similarity(ms_field.slow_state(),
                        &theme_centroids.iter().find(|(t, _)| t == theme).unwrap().1));
            }
        }

        // ------------------------------------------------------------------
        // 5. Record NCP effective tau (using a SemanticField with NCP access)
        // ------------------------------------------------------------------
        println!("\n--- NCP Effective Tau Sampling ---");
        {
            // Create a separate NCP-based field to probe tau values
            use crate::lnn::ncp::NCP;
            use rand::rngs::StdRng;
            use rand::SeedableRng;

            let ncp_config = NCPConfig::tiny();
            let mut rng = StdRng::seed_from_u64(42);
            let mut ncp = NCP::new(ncp_config, &mut rng);

            // Feed a few representative inputs and report tau
            let dt = 0.1;
            for (step, (_, _, hv)) in encoded_sentences.iter().enumerate() {
                // Use first 8 dimensions as a proxy input for the tiny NCP
                let input: Vec<f32> = hv.data.iter().take(8).cloned().collect();
                let _output = ncp.step(&input, dt);
                let tau = ncp.get_effective_tau(&input);

                if step % 5 == 0 {
                    println!(
                        "  Step {:2}: inter_tau=[{:.2},{:.2},{:.2},{:.2}]  cmd_tau=[{:.2},{:.2},{:.2},{:.2}]",
                        step + 1,
                        tau.inter[0], tau.inter[1], tau.inter[2], tau.inter[3],
                        tau.command[0], tau.command[1], tau.command[2], tau.command[3],
                    );
                }
            }
        }

        // ------------------------------------------------------------------
        // 6. Multi-scale vs single-scale retention comparison
        // ------------------------------------------------------------------
        println!("\n--- Multi-scale vs Single-scale Retention ---");

        let s1_hv = match &sentence1_hv {
            Some(hv) => hv.clone(),
            None => panic!("Sentence 1 encoding missing"),
        };

        // Multi-scale: slow field trace of sentence 1 after all 20 sentences
        let multi_retention = RealHV::cosine_similarity(ms_field.slow_state(), &s1_hv);
        println!("  Multi-scale slow field -> sentence 1: {:.4}", multi_retention);

        // Single-scale: fresh SemanticField, process all 20 sentences
        let sf_config = SemanticFieldConfig::new(D, BridgeStrategy::InputPreserving, NCPConfig::tiny());
        let mut single_field = SemanticField::new(sf_config, 42);
        for (_, _, hv) in &encoded_sentences {
            single_field.step(hv, 0.2); // same dt as fast field
        }
        let single_retention = RealHV::cosine_similarity(single_field.state(), &s1_hv);
        println!("  Single field -> sentence 1: {:.4}", single_retention);

        println!("  Retention advantage (multi - single): {:.4}", multi_retention - single_retention);

        // ------------------------------------------------------------------
        // 7. Save CSV files
        // ------------------------------------------------------------------
        std::fs::create_dir_all("results/experiments").ok();

        // exp7_theme_tracking.csv
        {
            let mut f = File::create("results/experiments/exp7_theme_tracking.csv")
                .expect("create exp7_theme_tracking.csv");
            writeln!(f, "step,theme,fast_sim,slow_sim,combined_sim").unwrap();
            for row in &tracking {
                writeln!(
                    f,
                    "{},{},{:.6},{:.6},{:.6}",
                    row.step, row.theme, row.fast_sim, row.slow_sim, row.combined_sim
                )
                .unwrap();
            }
            println!("\n  CSV saved: results/experiments/exp7_theme_tracking.csv");
        }

        // exp7_scale_comparison.csv
        {
            let mut f = File::create("results/experiments/exp7_scale_comparison.csv")
                .expect("create exp7_scale_comparison.csv");
            writeln!(f, "field_type,sentence1_retention").unwrap();
            writeln!(f, "multi_slow,{:.6}", multi_retention).unwrap();
            writeln!(f, "single,{:.6}", single_retention).unwrap();
            println!("  CSV saved: results/experiments/exp7_scale_comparison.csv");
        }

        // ------------------------------------------------------------------
        // 8. Assertions
        // ------------------------------------------------------------------
        assert!(
            multi_retention > single_retention,
            "Slow field should retain sentence 1 better than single: multi={:.4} single={:.4}",
            multi_retention, single_retention
        );
        println!("\n  [PASS] Multi-scale slow field retains sentence 1 better than single field");

        // Summary
        println!("\n{}", "=".repeat(60));
        println!("  SUMMARY — Experiment 7: Emergent Timescales");
        println!("{}", "=".repeat(60));
        println!("  Multi-scale slow retention: {:.4}", multi_retention);
        println!("  Single-scale retention:     {:.4}", single_retention);
        println!("  Advantage:                  {:.4}", multi_retention - single_retention);
        println!("{}", "=".repeat(60));

        println!("\n=== Experiment 7 Complete ===\n");
    }
}
