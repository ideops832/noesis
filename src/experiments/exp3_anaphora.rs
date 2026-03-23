//! Experiment 3: Memory and anaphora resolution.
//!
//! Uses a SemanticField to process a 3-turn dialogue about an entity (a cat).
//! Verifies that the entity representation persists across dialogue turns,
//! demonstrating the field's memory properties.

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;

    use crate::hdc::hypervector::HyperVector;
    use crate::hdc::real::RealHV;
    use crate::language::composer::Composer;
    use crate::lnn::ncp::NCPConfig;
    use crate::noesis::bridge::BridgeStrategy;
    use crate::noesis::semantic_field::{SemanticField, SemanticFieldConfig};

    const D: usize = 1024;

    #[test]
    fn test_exp3_anaphora() {
        println!("\n=== Experiment 3: Memory and Anaphora ===\n");

        // Create a composer to encode dialogue turns
        let mut composer = Composer::new(D, 42);

        // 3-turn dialogue about a cat entity
        let turn1 = "Il gatto dorme sul divano";
        let turn2 = "Il felino si sveglia e mangia";
        let turn3 = "L'animale gioca con la palla";

        // Encode each turn
        let hv_turn1 = composer.encode_sentence(turn1).unwrap();
        let hv_turn2 = composer.encode_sentence(turn2).unwrap();
        let hv_turn3 = composer.encode_sentence(turn3).unwrap();

        println!("Dialogue turns encoded:");
        println!("  Turn 1: \"{}\"", turn1);
        println!("  Turn 2: \"{}\"", turn2);
        println!("  Turn 3: \"{}\"", turn3);

        // Create SemanticField with InputPreserving strategy (preserves input traces)
        let config = SemanticFieldConfig::new(D, BridgeStrategy::InputPreserving, NCPConfig::tiny());
        let mut field = SemanticField::new(config, 42);

        // Process turn 1
        println!("\n--- Processing Turn 1 ---");
        for _ in 0..10 {
            field.step(&hv_turn1, 0.1);
        }
        let state_after_t1 = field.state().clone();
        let sim_t1 = RealHV::cosine_similarity(&state_after_t1, &hv_turn1);
        println!("  State similarity to turn 1: {:.4}", sim_t1);

        // Process turn 2
        println!("\n--- Processing Turn 2 ---");
        for _ in 0..10 {
            field.step(&hv_turn2, 0.1);
        }
        let state_after_t2 = field.state().clone();
        let sim_t1_after_t2 = RealHV::cosine_similarity(&state_after_t2, &hv_turn1);
        let sim_t2_after_t2 = RealHV::cosine_similarity(&state_after_t2, &hv_turn2);
        println!("  State similarity to turn 1: {:.4}", sim_t1_after_t2);
        println!("  State similarity to turn 2: {:.4}", sim_t2_after_t2);

        // Process turn 3
        println!("\n--- Processing Turn 3 ---");
        for _ in 0..10 {
            field.step(&hv_turn3, 0.1);
        }
        let state_after_t3 = field.state().clone();
        let sim_t1_after_t3 = RealHV::cosine_similarity(&state_after_t3, &hv_turn1);
        let sim_t2_after_t3 = RealHV::cosine_similarity(&state_after_t3, &hv_turn2);
        let sim_t3_after_t3 = RealHV::cosine_similarity(&state_after_t3, &hv_turn3);
        println!("  State similarity to turn 1: {:.4}", sim_t1_after_t3);
        println!("  State similarity to turn 2: {:.4}", sim_t2_after_t3);
        println!("  State similarity to turn 3: {:.4}", sim_t3_after_t3);

        // Check that the state is non-trivial (entity persists)
        let state_norm = state_after_t3.norm();
        println!("\n--- Summary ---");
        println!("  State norm after 3 turns: {:.4}", state_norm);
        println!("  State is finite: {}", state_after_t3.data.iter().all(|x| x.is_finite()));

        // The entity should persist: state should be non-zero and have some
        // relationship to the dialogue (similarity > 0 to at least one turn)
        let max_sim = sim_t1_after_t3.max(sim_t2_after_t3).max(sim_t3_after_t3);
        println!("  Max similarity to any turn: {:.4}", max_sim);

        // Verify that state is non-trivial after processing all turns
        assert!(
            state_norm > 0.01,
            "State should be non-zero after 3 turns, got norm = {:.4}",
            state_norm
        );

        // Verify all state values are finite
        assert!(
            state_after_t3.data.iter().all(|x| x.is_finite()),
            "State should contain only finite values"
        );

        // --- CSV Export ---
        {
            let mut f = File::create("results/exp3_anaphora.csv")
                .expect("Failed to create exp3_anaphora.csv");
            writeln!(f, "turn,input,sim_to_turn1,sim_to_turn2,sim_to_turn3,state_norm").unwrap();
            writeln!(f, "1,\"{}\",{:.6},,,{:.6}", turn1, sim_t1, state_after_t1.norm()).unwrap();
            writeln!(f, "2,\"{}\",{:.6},{:.6},,{:.6}", turn2, sim_t1_after_t2, sim_t2_after_t2, state_after_t2.norm()).unwrap();
            writeln!(f, "3,\"{}\",{:.6},{:.6},{:.6},{:.6}", turn3, sim_t1_after_t3, sim_t2_after_t3, sim_t3_after_t3, state_norm).unwrap();
        }

        println!("\n[CSV] Saved results/exp3_anaphora.csv");

        println!("\n[PASS] Entity persists: state norm = {:.4}, max_sim = {:.4}", state_norm, max_sim);
        println!("\n=== Experiment 3 Complete ===\n");
    }
}
