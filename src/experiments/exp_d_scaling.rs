//! Dimensionality scaling experiment: measures all metrics at D=1024,2048,4096,8192.

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;
    use std::time::Instant;

    use rand::rngs::StdRng;
    use rand::SeedableRng;

    use crate::hdc::binary::BinaryHV;
    use crate::hdc::encoder::Encoder;
    use crate::hdc::hypervector::HyperVector;
    use crate::hdc::real::RealHV;
    use crate::language::composer::Composer;
    use crate::language::tokenizer::Tokenizer;
    use crate::language::vocabulary::Vocabulary;
    use crate::lnn::ncp::NCPConfig;
    use crate::noesis::bridge::BridgeStrategy;
    use crate::noesis::memory_dynamics::SemanticMemory;
    use crate::noesis::multiscale::{MultiScaleConfig, MultiScaleField};
    use crate::noesis::semantic_field::{SemanticField, SemanticFieldConfig};
    use crate::utils::corpus;

    // =========================================================================
    // Compositionality at given D
    // =========================================================================
    fn eval_compositionality(d: usize) -> (f32, f32, f32, f32) {
        let mut config = SemanticFieldConfig::new(d, BridgeStrategy::InputPreserving, NCPConfig::tiny());
        config.alpha_override = Some(0.5);
        let mut encoder = Encoder::new(d, 1042);

        let roles: Vec<String> = (0..4).map(|i| format!("role_{i}")).collect();
        let values: Vec<String> = (0..4).map(|i| format!("val_{i}")).collect();
        for r in &roles { encoder.register(r); }
        for v in &values { encoder.register(v); }

        let mut top1 = 0u32;
        let mut top3 = 0u32;
        let mut top5 = 0u32;
        let mut total = 0u32;
        let mut total_unbind_sim = 0.0f32;

        for (ri, role) in roles.iter().enumerate().take(3) {
            for value in &values {
                let mut sf = SemanticField::new(config.clone(), 42 + ri as u64);
                let record = encoder.encode_record_named(&[(role, value)]);
                let record_real = record.to_real();
                for _ in 0..5 { sf.step(&record_real, 0.1); }

                let role_real = encoder.encode_atom(role).unwrap().to_real();
                let query_result = sf.query(&role_real);

                // Measure unbinding similarity to correct value
                let value_real = encoder.encode_atom(value).unwrap().to_real();
                let unbind_sim = RealHV::cosine_similarity(&query_result, &value_real);
                total_unbind_sim += unbind_sim;

                let decoded = encoder.vocab().nearest_k(&query_result.to_binary(), 5);
                total += 1;
                if !decoded.is_empty() && decoded[0].0 == *value { top1 += 1; }
                if decoded.iter().take(3).any(|(l, _)| l == value) { top3 += 1; }
                if decoded.iter().take(5).any(|(l, _)| l == value) { top5 += 1; }
            }
        }

        let t = total as f32;
        (top1 as f32 / t, top3 as f32 / t, top5 as f32 / t, total_unbind_sim / t)
    }

    // =========================================================================
    // Distributional similarity at given D
    // =========================================================================
    fn eval_distributional(d: usize) -> (f32, f32, f32, f32, f32) {
        let tokenizer = Tokenizer::new();
        let mut all: Vec<String> = corpus::expanded_corpus();
        all.extend(corpus::synonym_parallel_corpus());
        let sents: Vec<Vec<String>> = all.iter().map(|s| tokenizer.tokenize(s.as_str())).collect();

        let mut vocab = Vocabulary::new(d, 42);
        for s in &sents { for w in s { vocab.get_or_create(w); } }
        for _ in 0..5 { vocab.learn_from_context_momentum(&sents, 3, 0.7); }

        (
            vocab.similarity("gatto", "micio"),
            vocab.similarity("gatto", "felino"),
            vocab.similarity("cane", "cucciolo"),
            vocab.similarity("mamma", "papà"),
            vocab.similarity("gatto", "treno"),
        )
    }

    // =========================================================================
    // Memory & dynamics at given D
    // =========================================================================
    fn eval_memory_dynamics(d: usize) -> (usize, f32, f32, f32) {
        // Memory with rehearsal
        let sf_config = SemanticFieldConfig::new(d, BridgeStrategy::InputPreserving, NCPConfig::tiny());
        let mut mem = SemanticMemory::new(sf_config, 43);
        let mut rng = StdRng::seed_from_u64(99);
        let items: Vec<RealHV> = (0..5).map(|_| RealHV::random(d, &mut rng)).collect();
        for item in &items { mem.memorize(item); }
        for _ in 0..20 { mem.consolidation_cycle(1.0, 5, 0.05); }
        let surviving = mem.len();

        // Multi-scale
        let ms_config = MultiScaleConfig::default_for_dim(d);
        let mut ms = MultiScaleField::new(ms_config, 42);
        let mut rng = StdRng::seed_from_u64(99);
        let input_a = RealHV::random(d, &mut rng);
        let input_b = RealHV::random(d, &mut rng);
        for _ in 0..10 { ms.step(&input_a); }
        for _ in 0..5 { ms.step(&input_b); }
        let slow_a = RealHV::cosine_similarity(ms.slow_state(), &input_a);
        let combined_a = ms.similarity_to(&input_a);

        // Anaphora
        let mut composer = Composer::new(d, 42);
        let turns = ["gatto nero dorme divano", "felino sveglia mangia", "animale gioca palla"];
        let hvs: Vec<RealHV> = turns.iter().filter_map(|t| composer.encode_sentence(t)).collect();
        let ms_config2 = MultiScaleConfig::default_for_dim(d);
        let mut ms2 = MultiScaleField::new(ms_config2, 42);
        for hv in &hvs { for _ in 0..10 { ms2.step(hv); } }
        let t1_trace = RealHV::cosine_similarity(ms2.slow_state(), &hvs[0]);

        (surviving, slow_a, combined_a, t1_trace)
    }

    // =========================================================================
    // Sentence similarity at given D
    // =========================================================================
    fn eval_sentence(d: usize) -> (f32, f32) {
        let tokenizer = Tokenizer::new();
        let expanded = corpus::expanded_corpus();
        let sents: Vec<Vec<String>> = expanded.iter().map(|s| tokenizer.tokenize(s.as_str())).collect();

        let mut comp = Composer::new(d, 42);
        for s in &sents { for w in s { comp.vocabulary.get_or_create(w); } }
        for _ in 0..2 { comp.vocabulary.learn_from_context_raw(&sents, 2); }

        let sim_pairs = [
            ("Il gatto mangia il pesce", "Il felino mangia il pesce"),
            ("Il cane corre nel parco", "Il cucciolo corre nel parco"),
        ];
        let dif_pairs = [
            ("Il gatto mangia il pesce", "Il treno parte dalla stazione"),
            ("Il sole splende nel cielo", "La pasta cuoce nella pentola"),
        ];

        let avg_sim: f32 = sim_pairs.iter()
            .map(|(a, b)| comp.sentence_similarity(a, b))
            .sum::<f32>() / sim_pairs.len() as f32;
        let avg_dif: f32 = dif_pairs.iter()
            .map(|(a, b)| comp.sentence_similarity(a, b))
            .sum::<f32>() / dif_pairs.len() as f32;

        (avg_sim - avg_dif, avg_dif)
    }

    // =========================================================================
    // Distributional consistency at given D
    // =========================================================================
    fn eval_consistency(d: usize) -> f32 {
        let tokenizer = Tokenizer::new();
        let mut all: Vec<String> = corpus::expanded_corpus();
        all.extend(corpus::synonym_parallel_corpus());
        let sents: Vec<Vec<String>> = all.iter().map(|s| tokenizer.tokenize(s.as_str())).collect();

        let mut vocab = Vocabulary::new(d, 42);
        for s in &sents { for w in s { vocab.get_or_create(w); } }
        for _ in 0..3 { vocab.learn_from_context(&sents, 3); }

        let tests = [
            ("gatto", "felino", "treno"),
            ("gatto", "micio", "treno"),
            ("cane", "cucciolo", "sole"),
            ("mamma", "papà", "pioggia"),
            ("cuoce", "bolle", "corre"),
        ];
        let hits: usize = tests.iter()
            .filter(|(a, b, u)| vocab.similarity(a, b) > vocab.similarity(a, u))
            .count();
        hits as f32 / tests.len() as f32
    }

    // =========================================================================
    // Performance at given D
    // =========================================================================
    fn eval_performance(d: usize) -> (f32, f32, f32) {
        let config = SemanticFieldConfig::new(d, BridgeStrategy::InputPreserving, NCPConfig::tiny());
        let mut sf = SemanticField::new(config, 42);
        let mut rng = StdRng::seed_from_u64(99);
        let input = RealHV::random(d, &mut rng);

        // Warm up
        for _ in 0..5 { sf.step(&input, 0.1); }

        // Step timing
        let n = 50;
        let start = Instant::now();
        for _ in 0..n { sf.step(&input, 0.1); }
        let step_us = start.elapsed().as_micros() as f32 / n as f32;

        // Encoding timing
        let mut comp = Composer::new(d, 42);
        let start = Instant::now();
        for _ in 0..n { let _ = comp.encode_sentence("il gatto mangia il pesce"); }
        let enc_us = start.elapsed().as_micros() as f32 / n as f32;

        // Nearest timing
        let mut encoder = Encoder::new(d, 42);
        for i in 0..100 { encoder.register(&format!("w{i}")); }
        let q = BinaryHV::random(d, &mut rng);
        let start = Instant::now();
        for _ in 0..n { let _ = encoder.vocab().nearest_k(&q, 5); }
        let near_us = start.elapsed().as_micros() as f32 / n as f32;

        (step_us, enc_us, near_us)
    }

    // =========================================================================
    // Quasi-orthogonality at given D
    // =========================================================================
    fn eval_orthogonality(d: usize) -> f32 {
        let mut rng = StdRng::seed_from_u64(42);
        let vecs: Vec<RealHV> = (0..1000).map(|_| RealHV::random(d, &mut rng)).collect();
        let mut max_sim = 0.0f32;
        for i in 0..200 {
            for j in (i + 1)..200 {
                let s = RealHV::cosine_similarity(&vecs[i], &vecs[j]).abs();
                if s > max_sim { max_sim = s; }
            }
        }
        max_sim
    }

    // =========================================================================
    // Capacity test: max vocab size before Top-1 < 50%
    // =========================================================================
    fn eval_capacity(d: usize, vocab_size: usize) -> f32 {
        let mut encoder = Encoder::new(d, 1042);
        let roles: Vec<String> = (0..3).map(|i| format!("r{i}")).collect();
        let values: Vec<String> = (0..vocab_size).map(|i| format!("v{i}")).collect();
        for r in &roles { encoder.register(r); }
        for v in &values { encoder.register(v); }

        let mut correct = 0u32;
        let mut total = 0u32;
        for role in &roles {
            for value in values.iter().take(vocab_size.min(20)) {
                let rec = encoder.encode_record_named(&[(role, value)]);
                let rec_r = rec.to_real();
                let role_r = encoder.encode_atom(role).unwrap().to_real();
                let qr = RealHV::bind(&rec_r, &role_r.inverse());
                let decoded = encoder.vocab().nearest_k(&qr.to_binary(), 1);
                total += 1;
                if !decoded.is_empty() && decoded[0].0 == *value { correct += 1; }
            }
        }
        correct as f32 / total as f32
    }

    // =========================================================================
    // MAIN TEST: Full sweep
    // =========================================================================
    #[test]
    fn test_d_scaling_sweep() {
        let dims = [1024usize, 2048, 4096, 8192];

        println!("\n{}", "=".repeat(90));
        println!("  NOESIS Dimensionality Scaling Sweep");
        println!("{}\n", "=".repeat(90));

        let mut sweep_csv = File::create("results/d_scaling/sweep.csv").unwrap();
        writeln!(sweep_csv, "D,metric,value").unwrap();

        let mut comp_csv = File::create("results/d_scaling/compositionality_curve.csv").unwrap();
        writeln!(comp_csv, "D,top1,top3,top5,mean_unbind_sim").unwrap();

        let mut ortho_csv = File::create("results/d_scaling/orthogonality.csv").unwrap();
        writeln!(ortho_csv, "D,max_sim,theoretical").unwrap();

        let mut perf_csv = File::create("results/d_scaling/performance.csv").unwrap();
        writeln!(perf_csv, "D,step_us,encode_us,nearest_us").unwrap();

        let mut cap_csv = File::create("results/d_scaling/capacity.csv").unwrap();
        writeln!(cap_csv, "D,vocab_size,top1").unwrap();

        for &d in &dims {
            println!("--- D = {} ---\n", d);

            // 1. Compositionality
            let (t1, t3, t5, unbind_sim) = eval_compositionality(d);
            println!("  Compositionality: Top1={:.1}% Top3={:.1}% Top5={:.1}% unbind_sim={:.4}",
                t1*100.0, t3*100.0, t5*100.0, unbind_sim);
            writeln!(comp_csv, "{},{:.4},{:.4},{:.4},{:.4}", d, t1, t3, t5, unbind_sim).unwrap();
            writeln!(sweep_csv, "{},comp_top1,{:.4}", d, t1).unwrap();
            writeln!(sweep_csv, "{},comp_top3,{:.4}", d, t3).unwrap();
            writeln!(sweep_csv, "{},comp_top5,{:.4}", d, t5).unwrap();
            writeln!(sweep_csv, "{},unbind_sim,{:.4}", d, unbind_sim).unwrap();

            // 2. Distributional
            let (gm, gf, cc, mp, gt) = eval_distributional(d);
            println!("  Distributional: gatto-micio={:.4} felino={:.4} cane-cucc={:.4} mamma-papà={:.4} gatto-treno={:.4}",
                gm, gf, cc, mp, gt);
            writeln!(sweep_csv, "{},sim_gatto_micio,{:.4}", d, gm).unwrap();
            writeln!(sweep_csv, "{},sim_gatto_felino,{:.4}", d, gf).unwrap();
            writeln!(sweep_csv, "{},sim_cane_cucciolo,{:.4}", d, cc).unwrap();
            writeln!(sweep_csv, "{},sim_mamma_papa,{:.4}", d, mp).unwrap();
            writeln!(sweep_csv, "{},sim_gatto_treno,{:.4}", d, gt).unwrap();

            // 3. Memory & dynamics
            let (surv, slow_a, comb_a, t1_trace) = eval_memory_dynamics(d);
            println!("  Memory: surviving={}/5 slow_A={:.4} combined_A={:.4} T1@T3={:.4}",
                surv, slow_a, comb_a, t1_trace);
            writeln!(sweep_csv, "{},mem_surviving,{}", d, surv).unwrap();
            writeln!(sweep_csv, "{},ms_slow_a,{:.4}", d, slow_a).unwrap();
            writeln!(sweep_csv, "{},ms_combined_a,{:.4}", d, comb_a).unwrap();
            writeln!(sweep_csv, "{},anaphora_t1_t3,{:.4}", d, t1_trace).unwrap();

            // 4. Sentence
            let (gap, dif) = eval_sentence(d);
            println!("  Sentence: gap={:.4} different={:.4}", gap, dif);
            writeln!(sweep_csv, "{},sentence_gap,{:.4}", d, gap).unwrap();
            writeln!(sweep_csv, "{},sentence_dif,{:.4}", d, dif).unwrap();

            // 5. Consistency
            let cons = eval_consistency(d);
            println!("  Consistency: {:.0}%", cons * 100.0);
            writeln!(sweep_csv, "{},consistency,{:.4}", d, cons).unwrap();

            // 6. Orthogonality
            let max_sim = eval_orthogonality(d);
            let theoretical = 1.0 / (d as f32).sqrt() * 3.5; // ~3.5 sigma
            println!("  Orthogonality: max_sim={:.4} theoretical={:.4}", max_sim, theoretical);
            writeln!(ortho_csv, "{},{:.4},{:.4}", d, max_sim, theoretical).unwrap();
            writeln!(sweep_csv, "{},max_random_sim,{:.4}", d, max_sim).unwrap();

            // 7. Performance
            let (step_us, enc_us, near_us) = eval_performance(d);
            println!("  Performance: step={:.0}μs encode={:.0}μs nearest={:.0}μs", step_us, enc_us, near_us);
            writeln!(perf_csv, "{},{:.1},{:.1},{:.1}", d, step_us, enc_us, near_us).unwrap();
            writeln!(sweep_csv, "{},step_us,{:.1}", d, step_us).unwrap();

            // 8. Capacity
            for &vs in &[5, 10, 20, 50] {
                let cap = eval_capacity(d, vs);
                writeln!(cap_csv, "{},{},{:.4}", d, vs, cap).unwrap();
                if vs == 10 {
                    println!("  Capacity@10: {:.1}%", cap * 100.0);
                    writeln!(sweep_csv, "{},capacity_10,{:.4}", d, cap).unwrap();
                }
            }

            println!();
        }

        println!("{}", "=".repeat(90));
        println!("  Sweep complete. Results in results/d_scaling/");
        println!("{}", "=".repeat(90));
    }

    // =========================================================================
    // DIAGNOSTIC 1: Seed sensitivity at D=2048
    // =========================================================================
    fn eval_comp_with_seed(d: usize, seed: u64) -> (f32, f32, f32) {
        let mut config = SemanticFieldConfig::new(d, BridgeStrategy::InputPreserving, NCPConfig::tiny());
        config.alpha_override = Some(0.5);
        let mut encoder = Encoder::new(d, seed);

        let roles: Vec<String> = (0..4).map(|i| format!("role_{i}")).collect();
        let values: Vec<String> = (0..4).map(|i| format!("val_{i}")).collect();
        for r in &roles { encoder.register(r); }
        for v in &values { encoder.register(v); }

        let (mut t1, mut t3, mut t5, mut total) = (0u32, 0u32, 0u32, 0u32);
        for (ri, role) in roles.iter().enumerate().take(3) {
            for value in &values {
                let mut sf = SemanticField::new(config.clone(), seed + ri as u64);
                let rec = encoder.encode_record_named(&[(role, value)]).to_real();
                for _ in 0..5 { sf.step(&rec, 0.1); }
                let qr = sf.query(&encoder.encode_atom(role).unwrap().to_real());
                let decoded = encoder.vocab().nearest_k(&qr.to_binary(), 5);
                total += 1;
                if !decoded.is_empty() && decoded[0].0 == *value { t1 += 1; }
                if decoded.iter().take(3).any(|(l, _)| l == value) { t3 += 1; }
                if decoded.iter().take(5).any(|(l, _)| l == value) { t5 += 1; }
            }
        }
        let t = total as f32;
        (t1 as f32 / t, t3 as f32 / t, t5 as f32 / t)
    }

    /// Pure HDC compositionality without SemanticField (zero steps).
    fn eval_comp_pure_hdc(d: usize, seed: u64) -> (f32, f32) {
        let mut encoder = Encoder::new(d, seed);
        let roles: Vec<String> = (0..4).map(|i| format!("role_{i}")).collect();
        let values: Vec<String> = (0..4).map(|i| format!("val_{i}")).collect();
        for r in &roles { encoder.register(r); }
        for v in &values { encoder.register(v); }

        let (mut t1, mut total) = (0u32, 0u32);
        // Also track real-space accuracy (no to_binary)
        let mut t1_real = 0u32;
        for role in roles.iter().take(3) {
            for value in &values {
                let rec = encoder.encode_record_named(&[(role, value)]);
                let rec_r = rec.to_real();
                let role_r = encoder.encode_atom(role).unwrap().to_real();
                let qr = RealHV::bind(&rec_r, &role_r.inverse());

                // Binary search
                let decoded = encoder.vocab().nearest_k(&qr.to_binary(), 1);
                total += 1;
                if !decoded.is_empty() && decoded[0].0 == *value { t1 += 1; }

                // Real search
                let real_sims = encoder.vocab().all_similarities_real(&qr);
                let best = real_sims.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                if let Some((w, _)) = best { if w == value { t1_real += 1; } }
            }
        }
        let t = total as f32;
        (t1 as f32 / t, t1_real as f32 / t)
    }

    #[test]
    fn test_d_seed_sensitivity() {
        println!("\n=== DIAGNOSTIC: Seed Sensitivity ===\n");

        let mut csv = File::create("results/d_scaling/diagnosis_seed_sensitivity.csv").unwrap();
        writeln!(csv, "D,seed,top1,top3,top5").unwrap();

        for &d in &[1024usize, 2048, 4096, 8192] {
            println!("  D={}", d);
            for seed in 1042..1052 {
                let (t1, t3, t5) = eval_comp_with_seed(d, seed);
                println!("    seed={}: Top1={:.0}% Top3={:.0}% Top5={:.0}%", seed, t1*100.0, t3*100.0, t5*100.0);
                writeln!(csv, "{},{},{:.4},{:.4},{:.4}", d, seed, t1, t3, t5).unwrap();
            }
            println!();
        }
    }

    #[test]
    fn test_d_binary_vs_real() {
        println!("\n=== DIAGNOSTIC: Binary vs Real Search (Pure HDC, no dynamics) ===\n");

        let mut csv = File::create("results/d_scaling/diagnosis_binary_vs_real.csv").unwrap();
        writeln!(csv, "D,seed,binary_top1,real_top1").unwrap();

        for &d in &[1024usize, 2048, 4096, 8192] {
            println!("  D={}", d);
            let mut avg_bin = 0.0f32;
            let mut avg_real = 0.0f32;
            let n_seeds = 10;
            for seed in 1042..(1042 + n_seeds) {
                let (bin, real) = eval_comp_pure_hdc(d, seed);
                avg_bin += bin;
                avg_real += real;
                writeln!(csv, "{},{},{:.4},{:.4}", d, seed, bin, real).unwrap();
            }
            avg_bin /= n_seeds as f32;
            avg_real /= n_seeds as f32;
            println!("    Avg binary={:.1}% real={:.1}%", avg_bin*100.0, avg_real*100.0);
        }
    }

    #[test]
    fn test_d_scaling_multi_seed() {
        println!("\n=== DIAGNOSTIC: Multi-Seed Scaling ===\n");

        let mut csv = File::create("results/d_scaling/diagnosis_multi_seed.csv").unwrap();
        writeln!(csv, "D,avg_top1,avg_top3,avg_top5,std_top1").unwrap();

        for &d in &[1024usize, 2048, 4096, 6144, 8192] {
            let n_seeds = 10u64;
            let mut t1s = Vec::new();
            let mut t3s = Vec::new();
            let mut t5s = Vec::new();
            for seed in 0..n_seeds {
                let (t1, t3, t5) = eval_comp_with_seed(d, seed * 1000 + 42);
                t1s.push(t1);
                t3s.push(t3);
                t5s.push(t5);
            }
            let avg1: f32 = t1s.iter().sum::<f32>() / n_seeds as f32;
            let avg3: f32 = t3s.iter().sum::<f32>() / n_seeds as f32;
            let avg5: f32 = t5s.iter().sum::<f32>() / n_seeds as f32;
            let std1: f32 = (t1s.iter().map(|x| (x - avg1).powi(2)).sum::<f32>() / n_seeds as f32).sqrt();

            println!("  D={:5}: Top1={:.1}%±{:.1}% Top3={:.1}% Top5={:.1}%",
                d, avg1*100.0, std1*100.0, avg3*100.0, avg5*100.0);
            writeln!(csv, "{},{:.4},{:.4},{:.4},{:.4}", d, avg1, avg3, avg5, std1).unwrap();
        }
    }
}
