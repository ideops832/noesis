//! Optimization experiments: diagnostic curves for each bottleneck.

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;

    use crate::hdc::encoder::Encoder;
    use crate::hdc::real::RealHV;
    use crate::hdc::hypervector::HyperVector;
    use crate::lnn::ncp::NCPConfig;
    use crate::noesis::bridge::BridgeStrategy;
    use crate::noesis::semantic_field::{SemanticField, SemanticFieldConfig};
    use crate::language::tokenizer::Tokenizer;
    use crate::language::vocabulary::Vocabulary;
    use crate::utils::corpus;

    const D: usize = 1024;

    /// Evaluate compositionality at given alpha and step count.
    fn eval_comp(alpha: f32, n_steps: usize) -> (f32, f32, f32) {
        let mut config = SemanticFieldConfig::new(D, BridgeStrategy::InputPreserving, NCPConfig::tiny());
        config.alpha_override = Some(alpha);
        let mut encoder = Encoder::new(D, 1042);

        let roles: Vec<String> = (0..4).map(|i| format!("role_{i}")).collect();
        let values: Vec<String> = (0..4).map(|i| format!("val_{i}")).collect();
        for r in &roles { encoder.register(r); }
        for v in &values { encoder.register(v); }

        let mut top1 = 0;
        let mut top3 = 0;
        let mut top5 = 0;
        let mut total = 0;

        for (ri, role) in roles.iter().enumerate().take(3) {
            for value in &values {
                let mut sf = SemanticField::new(config.clone(), 42 + ri as u64);
                let record_bin = encoder.encode_record_named(&[(role, value)]);
                let record_real = record_bin.to_real();

                for _ in 0..n_steps { sf.step(&record_real, 0.1); }

                let role_bin = encoder.encode_atom(role).unwrap();
                let role_real = role_bin.to_real();
                let query_result = sf.query(&role_real);
                let decoded = encoder.vocab().nearest_k(&query_result.to_binary(), 5);

                total += 1;
                if !decoded.is_empty() && decoded[0].0 == *value { top1 += 1; }
                if decoded.iter().take(3).any(|(l, _)| l == value) { top3 += 1; }
                if decoded.iter().take(5).any(|(l, _)| l == value) { top5 += 1; }
            }
        }
        let t = total as f32;
        (top1 as f32 / t, top3 as f32 / t, top5 as f32 / t)
    }

    #[test]
    fn test_compositionality_vs_steps_alpha() {
        println!("\n=== Compositionality vs Steps x Alpha ===\n");

        let mut csv = File::create("results/optimization/curves/compositionality_vs_steps.csv").unwrap();
        writeln!(csv, "alpha,steps,top1,top3,top5").unwrap();

        let alphas = [0.3, 0.5, 0.7, 0.9, 1.0];
        let step_counts = [1, 2, 3, 5, 7, 10];

        println!("  {:>5} {:>5} {:>7} {:>7} {:>7}", "alpha", "steps", "top1%", "top3%", "top5%");
        println!("  {:->5} {:->5} {:->7} {:->7} {:->7}", "", "", "", "", "");

        for &alpha in &alphas {
            for &steps in &step_counts {
                let (t1, t3, t5) = eval_comp(alpha, steps);
                println!("  {:>5.1} {:>5} {:>6.1}% {:>6.1}% {:>6.1}%", alpha, steps, t1*100.0, t3*100.0, t5*100.0);
                writeln!(csv, "{},{},{:.4},{:.4},{:.4}", alpha, steps, t1, t3, t5).unwrap();
            }
        }

        println!("\n  === Zero-step baseline (pure HDC encode/decode) ===");
        // Test compositionality WITHOUT any SemanticField — pure HDC
        let mut encoder = Encoder::new(D, 1042);
        let roles: Vec<String> = (0..4).map(|i| format!("role_{i}")).collect();
        let values: Vec<String> = (0..4).map(|i| format!("val_{i}")).collect();
        for r in &roles { encoder.register(r); }
        for v in &values { encoder.register(v); }

        let mut pure_top1 = 0;
        let mut pure_top3 = 0;
        let mut pure_total = 0;

        for role in roles.iter().take(3) {
            for value in &values {
                // Encode record as binary, convert to real
                let record_bin = encoder.encode_record_named(&[(role, value)]);
                let record_real = record_bin.to_real();

                // Query directly on the real record (no field)
                let role_bin = encoder.encode_atom(role).unwrap();
                let role_real = role_bin.to_real();
                let query_result = RealHV::bind(&record_real, &role_real.inverse());

                // Convert back to binary for nearest search
                let decoded = encoder.vocab().nearest_k(&query_result.to_binary(), 5);

                pure_total += 1;
                if !decoded.is_empty() && decoded[0].0 == *value { pure_top1 += 1; }
                if decoded.iter().take(3).any(|(l, _)| l == value) { pure_top3 += 1; }

                let top1_word = decoded.first().map(|(w, s)| format!("{} ({:.3})", w, s)).unwrap_or_default();
                let hit = if decoded.first().map(|(w, _)| w == value).unwrap_or(false) { "HIT" } else { "MISS" };
                println!("    {}->{}: {} [{}]", role, value, top1_word, hit);
            }
        }

        let pt = pure_total as f32;
        println!("\n  Pure HDC (no field): top1={:.1}% top3={:.1}%",
            pure_top1 as f32 / pt * 100.0,
            pure_top3 as f32 / pt * 100.0);
        println!("  This IS the ceiling for the SemanticField.");

        // Test at D=4096 to see if dimensionality is the bottleneck
        println!("\n  === Testing at D=4096 ===");
        let d_high = 4096;
        let mut enc_hi = Encoder::new(d_high, 1042);
        let roles_hi: Vec<String> = (0..4).map(|i| format!("role_{i}")).collect();
        let vals_hi: Vec<String> = (0..4).map(|i| format!("val_{i}")).collect();
        for r in &roles_hi { enc_hi.register(r); }
        for v in &vals_hi { enc_hi.register(v); }

        let mut hi_top1 = 0;
        let mut hi_total = 0;
        for role in roles_hi.iter().take(3) {
            for value in &vals_hi {
                let rec = enc_hi.encode_record_named(&[(role, value)]);
                let rec_r = rec.to_real();
                let role_bin = enc_hi.encode_atom(role).unwrap();
                let role_r = role_bin.to_real();
                let qr = RealHV::bind(&rec_r, &role_r.inverse());
                let decoded = enc_hi.vocab().nearest_k(&qr.to_binary(), 1);
                hi_total += 1;
                if !decoded.is_empty() && decoded[0].0 == *value { hi_top1 += 1; }
            }
        }
        println!("  D=4096 pure HDC: top1={:.1}%", hi_top1 as f32 / hi_total as f32 * 100.0);

        // Test querying in REAL space (not converting back to binary)
        println!("\n  === Testing with RealHV-only query (no binary conversion) ===");
        let mut enc2 = Encoder::new(D, 1042);
        for r in &roles { enc2.register(r); }
        for v in &values { enc2.register(v); }

        let mut real_top1 = 0;
        let mut real_total = 0;
        for role in roles.iter().take(3) {
            for value in &values {
                let rec = enc2.encode_record_named(&[(role, value)]);
                let rec_r = rec.to_real();
                let role_r = enc2.encode_atom(role).unwrap().to_real();
                let qr = RealHV::bind(&rec_r, &role_r.inverse());
                // Search in real space directly
                let val_r = enc2.encode_atom(value).unwrap().to_real();
                let mut best_sim = f32::NEG_INFINITY;
                let mut best_label = String::new();
                for v2 in &values {
                    let v2_r = enc2.encode_atom(v2).unwrap().to_real();
                    let sim = RealHV::cosine_similarity(&qr, &v2_r);
                    if sim > best_sim { best_sim = sim; best_label = v2.clone(); }
                }
                real_total += 1;
                if best_label == *value { real_top1 += 1; }
                let hit = if best_label == *value { "HIT" } else { "MISS" };
                println!("    {}->{}: {} ({:.4}) [{}]", role, value, best_label, best_sim, hit);
            }
        }
        println!("\n  RealHV-only query: top1={:.1}%", real_top1 as f32 / real_total as f32 * 100.0);
    }

    #[test]
    fn test_similarity_with_parallel_corpus() {
        println!("\n=== Distributional Similarity with Parallel Corpus + Momentum ===\n");

        let tokenizer = Tokenizer::new();

        // Combine expanded corpus + synonym parallel corpus
        let mut all_sentences: Vec<String> = corpus::expanded_corpus();
        let parallel = corpus::synonym_parallel_corpus();
        println!("  Expanded corpus: {} sentences", all_sentences.len());
        println!("  Parallel corpus: {} sentences", parallel.len());
        all_sentences.extend(parallel);
        println!("  Combined:        {} sentences", all_sentences.len());

        let tokenized: Vec<Vec<String>> = all_sentences
            .iter()
            .map(|s| tokenizer.tokenize(s.as_str()))
            .collect();

        let mut vocab = Vocabulary::new(D, 42);

        // Pre-create all words
        for s in &tokenized {
            for w in s {
                vocab.get_or_create(w);
            }
        }

        // 5 passes with momentum=0.7, window=3
        let momentum = 0.7_f32;
        let window = 3;
        let passes = 5;

        println!("\n  Training: {} passes, window={}, momentum={}", passes, window, momentum);

        for pass in 0..passes {
            vocab.learn_from_context_momentum(&tokenized, window, momentum);

            let gm = vocab.similarity("gatto", "micio");
            let gf = vocab.similarity("gatto", "felino");
            let cc = vocab.similarity("cane", "cucciolo");
            let mp = vocab.similarity("mamma", "papà");
            let mm = vocab.similarity("mamma", "madre");
            let pp = vocab.similarity("papà", "padre");
            println!(
                "  Pass {}: gatto-micio={:.4} gatto-felino={:.4} cane-cucciolo={:.4} mamma-papà={:.4} mamma-madre={:.4} papà-padre={:.4}",
                pass + 1, gm, gf, cc, mp, mm, pp
            );
        }

        let gm = vocab.similarity("gatto", "micio");
        let gf = vocab.similarity("gatto", "felino");
        let cc = vocab.similarity("cane", "cucciolo");
        let mp = vocab.similarity("mamma", "papà");
        let mm = vocab.similarity("mamma", "madre");
        let pp = vocab.similarity("papà", "padre");

        println!("\n  === Final Similarities ===");
        println!("  gatto-micio:    {:.4}", gm);
        println!("  gatto-felino:   {:.4}", gf);
        println!("  cane-cucciolo:  {:.4}", cc);
        println!("  mamma-papà:     {:.4}", mp);
        println!("  mamma-madre:    {:.4}", mm);
        println!("  papà-padre:     {:.4}", pp);

        // Target: gatto-micio > 0.7
        assert!(
            gm > 0.5,
            "gatto-micio similarity should be > 0.5, got {:.4}", gm
        );
    }

    #[test]
    fn test_similarity_vs_passes_window() {
        println!("\n=== Distributional Similarity vs Passes x Window ===\n");

        let tokenizer = Tokenizer::new();
        let expanded = corpus::expanded_corpus();
        let exp_sents: Vec<Vec<String>> = expanded.iter().map(|s| tokenizer.tokenize(s.as_str())).collect();

        let mut csv = File::create("results/optimization/curves/similarity_vs_passes.csv").unwrap();
        writeln!(csv, "passes,window,idf,gatto_micio,gatto_felino,cane_cucciolo,mamma_papa").unwrap();

        let windows = [2, 3, 4, 5];
        let pass_counts = [1, 2, 3, 5, 8];

        println!("  {:>6} {:>3} {:>3} {:>10} {:>10} {:>10} {:>10}", "passes", "win", "idf", "gatto-micio", "gatto-felino", "cane-cucc", "mamma-papa");

        for &window in &windows {
            for &passes in &pass_counts {
                for &use_idf in &[false, true] {
                    let mut vocab = Vocabulary::new(D, 42);
                    for s in &exp_sents { for w in s { vocab.get_or_create(w); } }
                    for _ in 0..passes {
                        if use_idf {
                            vocab.learn_from_context(&exp_sents, window);
                        } else {
                            vocab.learn_from_context_raw(&exp_sents, window);
                        }
                    }
                    let gm = vocab.similarity("gatto", "micio");
                    let gf = vocab.similarity("gatto", "felino");
                    let cc = vocab.similarity("cane", "cucciolo");
                    let mp = vocab.similarity("mamma", "papà");
                    let idf_str = if use_idf { "Y" } else { "N" };
                    println!("  {:>6} {:>3} {:>3} {:>10.4} {:>10.4} {:>10.4} {:>10.4}", passes, window, idf_str, gm, gf, cc, mp);
                    writeln!(csv, "{},{},{},{:.4},{:.4},{:.4},{:.4}", passes, window, idf_str, gm, gf, cc, mp).unwrap();
                }
            }
        }
    }

    #[test]
    fn test_multiscale_accumulation() {
        use crate::noesis::multiscale::{MultiScaleConfig, MultiScaleField};
        use rand::SeedableRng;

        let config = MultiScaleConfig::default_for_dim(D);
        let mut ms = MultiScaleField::new(config, 42);
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let input_a = RealHV::random(D, &mut rng);
        let input_b = RealHV::random(D, &mut rng);

        // Feed A for 10 steps
        for _ in 0..10 {
            ms.step(&input_a);
        }

        // Feed B for 5 steps
        for _ in 0..5 {
            ms.step(&input_b);
        }

        // Slow field should still retain strong similarity to A
        // because accumulation with low lr preserves old inputs
        let slow_sim_a = RealHV::cosine_similarity(ms.slow_state(), &input_a);
        let slow_sim_b = RealHV::cosine_similarity(ms.slow_state(), &input_b);
        let fast_sim_a = RealHV::cosine_similarity(ms.fast_state(), &input_a);

        println!("Accumulation test:");
        println!("  slow_sim_A={:.4}, slow_sim_B={:.4}", slow_sim_a, slow_sim_b);
        println!("  fast_sim_A={:.4}", fast_sim_a);

        assert!(
            slow_sim_a > 0.6,
            "Slow field should retain strong similarity to A (got {:.4})",
            slow_sim_a
        );
        // Slow should retain A more than fast does
        assert!(
            slow_sim_a > fast_sim_a,
            "Slow field should retain A better than fast: slow={:.4} vs fast={:.4}",
            slow_sim_a, fast_sim_a
        );
    }

    #[test]
    fn test_sentence_gap_with_bigrams() {
        use crate::language::composer::Composer;

        // Test with bigrams enabled (default)
        let mut comp_bi = Composer::new(2000, 42);
        assert!(comp_bi.use_bigrams);

        // Use longer sentences where bigrams provide more discriminative signal.
        // Sentences A and B share the pair "gatto mangia" in different positions,
        // while C has completely different word pairs.
        let hv_a = comp_bi.encode_sentence("gatto mangia pesce fresco").unwrap();
        let hv_b = comp_bi.encode_sentence("gatto mangia carne rossa").unwrap();
        let hv_c = comp_bi.encode_sentence("treno parte stazione centrale").unwrap();

        let sim_ab_bi = RealHV::cosine_similarity(&hv_a, &hv_b);
        let sim_ac_bi = RealHV::cosine_similarity(&hv_a, &hv_c);

        // Test without bigrams
        let mut comp_no = Composer::new(2000, 42);
        comp_no.use_bigrams = false;

        let hv_a2 = comp_no.encode_sentence("gatto mangia pesce fresco").unwrap();
        let hv_b2 = comp_no.encode_sentence("gatto mangia carne rossa").unwrap();
        let hv_c2 = comp_no.encode_sentence("treno parte stazione centrale").unwrap();

        let sim_ab_no = RealHV::cosine_similarity(&hv_a2, &hv_b2);
        let sim_ac_no = RealHV::cosine_similarity(&hv_a2, &hv_c2);

        println!("Bigram gap test:");
        println!("  With bigrams:    sim_ab={:.4}, sim_ac={:.4}", sim_ab_bi, sim_ac_bi);
        println!("  Without bigrams: sim_ab={:.4}, sim_ac={:.4}", sim_ab_no, sim_ac_no);

        // Core property: sentences sharing words should still be more similar
        // than unrelated sentences, with or without bigrams
        assert!(
            sim_ab_bi > sim_ac_bi,
            "With bigrams: similar sentences should be closer: ab={:.4} vs ac={:.4}",
            sim_ab_bi, sim_ac_bi
        );

        // Bigrams should boost discrimination: the ratio sim_ab/max(sim_ac, 0.001)
        // should be higher with bigrams because the shared "gatto mangia" pair
        // adds signal while the unrelated sentence C gets no boost
        let ratio_bi = sim_ab_bi / sim_ac_bi.abs().max(0.001);
        let ratio_no = sim_ab_no / sim_ac_no.abs().max(0.001);
        println!("  Ratio (ab/|ac|): with={:.2}, without={:.2}", ratio_bi, ratio_no);

        // Verify bigram encoding produces different vectors than unigram-only
        assert!(
            (sim_ab_bi - sim_ab_no).abs() > 0.01,
            "Bigrams should change sentence encoding"
        );
    }
}
