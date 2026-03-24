//! Experiment 3: Memory and anaphora resolution via MultiScaleField.
//!
//! Expanded protocol:
//! 1. Process dialogues through MultiScaleField, track entity persistence in slow field
//! 2. Dialogue 1: 4 turns about a cat — measure "gatto" trace after each turn
//! 3. Dialogue 2: 4 turns about cooking — measure "cucina" trace after each turn
//! 4. Decay test: idle steps and measure entity trace degradation
//! 5. Topic switch: cat -> cooking -> query "era stanco"
//! 6. Export CSV results to results/experiments/

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

    use crate::lnn::ncp::NCPConfig;
    use crate::noesis::bridge::BridgeStrategy;
    use crate::noesis::multiscale::{MultiScaleConfig, MultiScaleField};

    const D: usize = 1024;

    #[test]
    fn test_exp3_research() {
        println!("\n{}", "=".repeat(60));
        println!("=== Experiment 3: Anaphora & Memory — Expanded Protocol ===");
        println!("{}\n", "=".repeat(60));

        // ── Setup: composer + MultiScaleField ──────────────────────────────
        let tokenizer = Tokenizer::new();
        let mut full_corpus = corpus::expanded_corpus();
        full_corpus.extend(corpus::synonym_parallel_corpus());
        let sentences: Vec<Vec<String>> = full_corpus.iter().map(|s| tokenizer.tokenize(s)).collect();

        let mut composer = Composer::new(D, 42);
        for sentence in &sentences {
            for word in sentence {
                composer.vocabulary.get_or_create(word);
            }
        }
        for _ in 0..3 {
            composer.vocabulary.learn_from_context_momentum(&sentences, 3, 0.7);
        }
        println!("  Composer trained: {} words\n", composer.vocabulary.len());

        // Encode reference word "gatto"
        let hv_gatto = composer.vocabulary.get_or_create_clone("gatto");
        let hv_cucina = composer.vocabulary.get_or_create_clone("cucina");

        // ── 1. Dialogue 1: Cat topic (4 turns) ────────────────────────────
        println!("--- Phase 1: Dialogue 1 (cat topic) ---\n");

        let cat_turns = vec![
            "il gatto dorme sul divano",
            "il felino si sveglia e si stiracchia",
            "il micio mangia il pesce nella ciotola",
            "il gatto gioca con la palla di lana",
        ];

        let ms_config = MultiScaleConfig::default_for_dim(D);
        let mut ms = MultiScaleField::new(ms_config.clone(), 42);

        struct DialogueEntry {
            turn: usize,
            text: String,
            slow_sim_entity: f32,
            fast_sim_entity: f32,
            combined_sim_entity: f32,
        }

        let mut cat_results: Vec<DialogueEntry> = Vec::new();

        for (i, turn) in cat_turns.iter().enumerate() {
            let hv_turn = composer.encode_sentence(turn)
                .unwrap_or_else(|| RealHV::zero(D));

            // Step the multi-scale field multiple times per turn
            for _ in 0..5 {
                ms.step(&hv_turn);
            }

            let slow_sim = RealHV::cosine_similarity(ms.slow_state(), &hv_gatto);
            let fast_sim = RealHV::cosine_similarity(ms.fast_state(), &hv_gatto);
            let combined_sim = RealHV::cosine_similarity(ms.combined_state(), &hv_gatto);

            println!("  Turn {}: \"{}\"", i + 1, turn);
            println!("    slow_sim(gatto)={:.4}  fast_sim(gatto)={:.4}  combined={:.4}",
                slow_sim, fast_sim, combined_sim);

            cat_results.push(DialogueEntry {
                turn: i + 1,
                text: turn.to_string(),
                slow_sim_entity: slow_sim,
                fast_sim_entity: fast_sim,
                combined_sim_entity: combined_sim,
            });
        }

        let slow_sim_after_cat = RealHV::cosine_similarity(ms.slow_state(), &hv_gatto);
        println!("\n  Entity trace in slow field after 4 cat turns: {:.4}", slow_sim_after_cat);

        // ── 2. Dialogue 2: Cooking topic (4 turns) ────────────────────────
        println!("\n--- Phase 2: Dialogue 2 (cooking topic) ---\n");

        let cooking_turns = vec![
            "il cuoco prepara la cena in cucina",
            "la pasta cuoce nella pentola grande",
            "il pane esce dal forno caldo",
            "la torta viene decorata con la crema",
        ];

        let mut ms2 = MultiScaleField::new(ms_config.clone(), 42);

        let mut cooking_results: Vec<DialogueEntry> = Vec::new();

        for (i, turn) in cooking_turns.iter().enumerate() {
            let hv_turn = composer.encode_sentence(turn)
                .unwrap_or_else(|| RealHV::zero(D));

            for _ in 0..5 {
                ms2.step(&hv_turn);
            }

            let slow_sim = RealHV::cosine_similarity(ms2.slow_state(), &hv_cucina);
            let fast_sim = RealHV::cosine_similarity(ms2.fast_state(), &hv_cucina);
            let combined_sim = RealHV::cosine_similarity(ms2.combined_state(), &hv_cucina);

            println!("  Turn {}: \"{}\"", i + 1, turn);
            println!("    slow_sim(cucina)={:.4}  fast_sim(cucina)={:.4}  combined={:.4}",
                slow_sim, fast_sim, combined_sim);

            cooking_results.push(DialogueEntry {
                turn: i + 1,
                text: turn.to_string(),
                slow_sim_entity: slow_sim,
                fast_sim_entity: fast_sim,
                combined_sim_entity: combined_sim,
            });
        }

        // ── 3. Decay test ──────────────────────────────────────────────────
        println!("\n--- Phase 3: Decay test ---\n");

        // Use the cat field from Phase 1 (ms), run idle steps
        let decay_steps = vec![5, 10, 15, 20];

        struct DecayEntry {
            idle_steps: usize,
            gatto_trace: f32,
            slow_trace: f32,
            fast_trace: f32,
        }

        let mut decay_results: Vec<DecayEntry> = Vec::new();

        // Clone state before decay to start fresh each time
        let mut ms_decay = MultiScaleField::new(ms_config.clone(), 42);
        // Replay cat dialogue
        for turn in &cat_turns {
            let hv_turn = composer.encode_sentence(turn)
                .unwrap_or_else(|| RealHV::zero(D));
            for _ in 0..5 {
                ms_decay.step(&hv_turn);
            }
        }

        println!("  {:>12} | {:>12} | {:>12} | {:>12}", "Idle Steps", "Combined", "Slow", "Fast");
        println!("  {:-<12}-+-{:-<12}-+-{:-<12}-+-{:-<12}", "", "", "", "");

        // Baseline (0 idle steps)
        let base_combined = RealHV::cosine_similarity(ms_decay.combined_state(), &hv_gatto);
        let base_slow = RealHV::cosine_similarity(ms_decay.slow_state(), &hv_gatto);
        let base_fast = RealHV::cosine_similarity(ms_decay.fast_state(), &hv_gatto);
        println!("  {:>12} | {:>12.4} | {:>12.4} | {:>12.4}", 0, base_combined, base_slow, base_fast);

        decay_results.push(DecayEntry {
            idle_steps: 0,
            gatto_trace: base_combined,
            slow_trace: base_slow,
            fast_trace: base_fast,
        });

        let mut cumulative_idle = 0usize;
        for &target_steps in &decay_steps {
            let steps_to_run = target_steps - cumulative_idle;
            for _ in 0..steps_to_run {
                ms_decay.idle_step();
            }
            cumulative_idle = target_steps;

            let combined_trace = RealHV::cosine_similarity(ms_decay.combined_state(), &hv_gatto);
            let slow_trace = RealHV::cosine_similarity(ms_decay.slow_state(), &hv_gatto);
            let fast_trace = RealHV::cosine_similarity(ms_decay.fast_state(), &hv_gatto);

            println!("  {:>12} | {:>12.4} | {:>12.4} | {:>12.4}",
                target_steps, combined_trace, slow_trace, fast_trace);

            decay_results.push(DecayEntry {
                idle_steps: target_steps,
                gatto_trace: combined_trace,
                slow_trace,
                fast_trace,
            });
        }

        // ── 4. Topic switch test ───────────────────────────────────────────
        println!("\n--- Phase 4: Topic switch test ---\n");

        let mut ms_switch = MultiScaleField::new(ms_config.clone(), 42);

        struct SwitchEntry {
            phase: String,
            turn: usize,
            text: String,
            gatto_trace: f32,
            cucina_trace: f32,
        }

        let mut switch_results: Vec<SwitchEntry> = Vec::new();

        // 3 turns about cat
        let cat_switch_turns = vec![
            "il gatto dorme sul divano",
            "il felino mangia il pesce",
            "il micio era stanco dopo aver giocato",
        ];

        println!("  [Cat phase]");
        for (i, turn) in cat_switch_turns.iter().enumerate() {
            let hv_turn = composer.encode_sentence(turn)
                .unwrap_or_else(|| RealHV::zero(D));
            for _ in 0..5 {
                ms_switch.step(&hv_turn);
            }

            let gt = RealHV::cosine_similarity(ms_switch.slow_state(), &hv_gatto);
            let ct = RealHV::cosine_similarity(ms_switch.slow_state(), &hv_cucina);
            println!("    Turn {}: gatto_trace={:.4}  cucina_trace={:.4}  \"{}\"",
                i + 1, gt, ct, turn);

            switch_results.push(SwitchEntry {
                phase: "cat".to_string(),
                turn: i + 1,
                text: turn.to_string(),
                gatto_trace: gt,
                cucina_trace: ct,
            });
        }

        // 3 turns about cooking
        let cooking_switch_turns = vec![
            "il cuoco prepara la cena",
            "la pasta cuoce in cucina",
            "il pane esce dal forno",
        ];

        println!("  [Cooking phase]");
        for (i, turn) in cooking_switch_turns.iter().enumerate() {
            let hv_turn = composer.encode_sentence(turn)
                .unwrap_or_else(|| RealHV::zero(D));
            for _ in 0..5 {
                ms_switch.step(&hv_turn);
            }

            let gt = RealHV::cosine_similarity(ms_switch.slow_state(), &hv_gatto);
            let ct = RealHV::cosine_similarity(ms_switch.slow_state(), &hv_cucina);
            println!("    Turn {}: gatto_trace={:.4}  cucina_trace={:.4}  \"{}\"",
                i + 4, gt, ct, turn);

            switch_results.push(SwitchEntry {
                phase: "cooking".to_string(),
                turn: i + 4,
                text: turn.to_string(),
                gatto_trace: gt,
                cucina_trace: ct,
            });
        }

        // Query: "era stanco" — should still have some gatto trace
        let query = "era stanco";
        let hv_query = composer.encode_sentence(query)
            .unwrap_or_else(|| RealHV::zero(D));

        let gatto_trace_at_query = RealHV::cosine_similarity(ms_switch.slow_state(), &hv_gatto);
        let query_sim_combined = RealHV::cosine_similarity(ms_switch.combined_state(), &hv_query);

        println!("\n  Query: \"{}\"", query);
        println!("    gatto trace in slow field: {:.4}", gatto_trace_at_query);
        println!("    query sim to combined state: {:.4}", query_sim_combined);

        switch_results.push(SwitchEntry {
            phase: "query".to_string(),
            turn: 7,
            text: query.to_string(),
            gatto_trace: gatto_trace_at_query,
            cucina_trace: RealHV::cosine_similarity(ms_switch.slow_state(), &hv_cucina),
        });

        // ── 5. Save CSVs ──────────────────────────────────────────────────
        println!("\n--- Phase 5: Saving CSV results ---");

        // exp3_dialogue.csv (both dialogues combined)
        {
            let mut f = File::create("results/experiments/exp3_dialogue.csv")
                .expect("Failed to create exp3_dialogue.csv");
            writeln!(f, "dialogue,turn,text,slow_sim_entity,fast_sim_entity,combined_sim_entity").unwrap();
            for r in &cat_results {
                writeln!(f, "cat,{},\"{}\",{:.6},{:.6},{:.6}",
                    r.turn, r.text, r.slow_sim_entity, r.fast_sim_entity, r.combined_sim_entity).unwrap();
            }
            for r in &cooking_results {
                writeln!(f, "cooking,{},\"{}\",{:.6},{:.6},{:.6}",
                    r.turn, r.text, r.slow_sim_entity, r.fast_sim_entity, r.combined_sim_entity).unwrap();
            }
            println!("  Saved results/experiments/exp3_dialogue.csv");
        }

        // exp3_decay.csv
        {
            let mut f = File::create("results/experiments/exp3_decay.csv")
                .expect("Failed to create exp3_decay.csv");
            writeln!(f, "idle_steps,gatto_trace_combined,gatto_trace_slow,gatto_trace_fast").unwrap();
            for r in &decay_results {
                writeln!(f, "{},{:.6},{:.6},{:.6}", r.idle_steps, r.gatto_trace, r.slow_trace, r.fast_trace).unwrap();
            }
            println!("  Saved results/experiments/exp3_decay.csv");
        }

        // exp3_topic_switch.csv
        {
            let mut f = File::create("results/experiments/exp3_topic_switch.csv")
                .expect("Failed to create exp3_topic_switch.csv");
            writeln!(f, "phase,turn,text,gatto_trace,cucina_trace").unwrap();
            for r in &switch_results {
                writeln!(f, "{},{},\"{}\",{:.6},{:.6}",
                    r.phase, r.turn, r.text, r.gatto_trace, r.cucina_trace).unwrap();
            }
            println!("  Saved results/experiments/exp3_topic_switch.csv");
        }

        // ── 6. Assertions ─────────────────────────────────────────────────
        println!("\n--- Phase 6: Assertions ---");

        // Entity trace in slow field after 4 turns should be > 0.5
        // Use the maximum slow_sim from cat dialogue turns
        let max_slow_sim = cat_results.iter()
            .map(|r| r.slow_sim_entity)
            .fold(f32::NEG_INFINITY, f32::max);

        println!("  Max slow_sim(gatto) during cat dialogue: {:.4}", max_slow_sim);
        println!("  Final slow_sim(gatto) after 4 turns: {:.4}", slow_sim_after_cat);

        // The slow field should accumulate entity trace
        // We check the final value after all 4 turns
        assert!(
            slow_sim_after_cat > 0.5,
            "Entity trace in slow field after 4 cat turns should be > 0.5, got {:.4}",
            slow_sim_after_cat
        );
        println!("  [PASS] Slow field entity trace = {:.4} > 0.5", slow_sim_after_cat);

        println!("\n=== Experiment 3 Complete ===\n");
    }
}
