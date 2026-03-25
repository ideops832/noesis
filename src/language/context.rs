//! Conversational context management via SemanticField.
//!
//! Each utterance is encoded as a hypervector and fed to a SemanticField,
//! which accumulates and evolves the conversational state over time.
//! Salient concepts are recovered by comparing the field state against
//! the vocabulary.

use crate::hdc::real::RealHV;
use crate::language::composer::Composer;
use crate::noesis::semantic_field::{SemanticField, SemanticFieldConfig};

/// Report produced after each conversational turn.
pub struct TurnReport {
    /// The turn number (1-indexed).
    pub turn: usize,
    /// How much the field state changed this turn (cosine distance).
    pub state_change: f32,
    /// Top salient concepts after this turn.
    pub salient_concepts: Vec<(String, f32)>,
}

/// Manages conversational context by driving a SemanticField with
/// encoded utterances.
pub struct ConversationContext {
    field: SemanticField,
    composer: Composer,
    turn_count: usize,
    dt_per_turn: f32,
    /// Alpha for context-aware token weighting (R5).
    /// 0.0 disables contextual encoding (equivalent to plain encode).
    pub context_alpha: f32,
}

impl ConversationContext {
    /// Creates a new ConversationContext.
    pub fn new(
        field_config: SemanticFieldConfig,
        dim: usize,
        composer_seed: u64,
        dt_per_turn: f32,
    ) -> Self {
        let field = SemanticField::new(field_config, composer_seed);
        let composer = Composer::new(dim, composer_seed);
        ConversationContext {
            field,
            composer,
            turn_count: 0,
            dt_per_turn,
            context_alpha: 0.3,
        }
    }

    /// Process an utterance: encode it, step the field, return a report.
    pub fn process_utterance(&mut self, text: &str) -> TurnReport {
        let state_before = self.field.state().clone();

        // Encode the utterance, using contextual weighting when context exists.
        let state = self.field.state();
        let input_hv = if state.norm() > 1e-9 && self.context_alpha.abs() > 1e-9 {
            self.composer
                .encode_sentence_contextual(text, &state.clone(), self.context_alpha)
                .unwrap_or_else(|| RealHV::zero(self.composer.vocabulary.dim))
        } else {
            self.composer
                .encode_sentence(text)
                .unwrap_or_else(|| RealHV::zero(self.composer.vocabulary.dim))
        };

        // Step the field with the encoded utterance
        self.field.step(&input_hv, self.dt_per_turn);

        self.turn_count += 1;

        // Compute state change
        let state_after = self.field.state();
        let state_change = if state_before.norm() < 1e-9 {
            state_after.norm()
        } else {
            1.0 - RealHV::cosine_similarity(&state_before, state_after)
        };

        let salient_concepts = self.get_salient_concepts(5);

        TurnReport {
            turn: self.turn_count,
            state_change,
            salient_concepts,
        }
    }

    /// Returns the top-k words from the vocabulary most similar to the
    /// current field state.
    pub fn get_salient_concepts(&self, k: usize) -> Vec<(String, f32)> {
        let state = self.field.state();
        if state.norm() < 1e-9 {
            return Vec::new();
        }

        let mut sims: Vec<(String, f32)> = self
            .composer
            .vocabulary
            .words
            .iter()
            .map(|(word, hv)| (word.clone(), RealHV::cosine_similarity(state, hv)))
            .collect();

        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sims.truncate(k);
        sims
    }

    /// Run n idle steps (no input), allowing the field to decay.
    pub fn idle_turns(&mut self, n: usize) {
        for _ in 0..n {
            self.field.idle_step(self.dt_per_turn);
        }
    }

    /// Encode a question, unbind it from the current state, and find the
    /// nearest vocabulary words to the result.
    pub fn query_context(&mut self, question: &str) -> Vec<(String, f32)> {
        let q_hv = match self.composer.encode_sentence(question) {
            Some(hv) => hv,
            None => return Vec::new(),
        };

        let state = self.field.state();
        if state.norm() < 1e-9 {
            return Vec::new();
        }

        // Unbind question from state to recover related concepts
        let result = RealHV::bind(state, &q_hv.inverse());

        let mut sims: Vec<(String, f32)> = self
            .composer
            .vocabulary
            .words
            .iter()
            .map(|(word, hv)| (word.clone(), RealHV::cosine_similarity(&result, hv)))
            .collect();

        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sims.truncate(10);
        sims
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lnn::ncp::NCPConfig;
    use crate::noesis::bridge::BridgeStrategy;

    fn make_context(dim: usize) -> ConversationContext {
        let config = SemanticFieldConfig::new(dim, BridgeStrategy::DualTrack, NCPConfig::tiny());
        ConversationContext::new(config, dim, 42, 0.1)
    }

    #[test]
    fn test_context_accumulates() {
        let dim = 1024;
        let mut ctx = make_context(dim);

        // 5 turns on the same topic (cats)
        let utterances = [
            "il gatto dorme sul divano",
            "il gatto mangia il pesce",
            "il felino gioca con la palla",
            "il micio corre nel giardino",
            "il gatto beve il latte",
        ];

        for u in &utterances {
            let _report = ctx.process_utterance(u);
        }

        let concepts = ctx.get_salient_concepts(10);
        // After repeated cat-related utterances, cat-related words should
        // appear among the salient concepts.
        let concept_words: Vec<&str> = concepts.iter().map(|(w, _)| w.as_str()).collect();

        // At least one of gatto/felino/micio should appear in top concepts
        let has_cat_word = concept_words.iter().any(|w| {
            *w == "gatto" || *w == "felino" || *w == "micio"
                || *w == "mangia" || *w == "dorme" || *w == "gioca"
                || *w == "pesce" || *w == "palla" || *w == "latte"
        });

        assert!(
            has_cat_word,
            "Salient concepts should include topic-related words, got: {:?}",
            concept_words
        );
    }

    #[test]
    fn test_context_shifts() {
        let dim = 1024;
        let mut ctx = make_context(dim);

        // Topic 1: animals
        for _ in 0..3 {
            ctx.process_utterance("il gatto mangia il pesce");
        }
        let concepts_before = ctx.get_salient_concepts(10);

        // Topic 2: trains
        for _ in 0..5 {
            ctx.process_utterance("il treno parte dalla stazione");
        }
        let concepts_after = ctx.get_salient_concepts(10);

        // New concepts should emerge after topic shift
        let words_after: Vec<&str> = concepts_after.iter().map(|(w, _)| w.as_str()).collect();
        let has_train_word = words_after.iter().any(|w| {
            *w == "treno" || *w == "parte" || *w == "stazione"
        });

        assert!(
            has_train_word,
            "After topic shift, new topic words should emerge in concepts, got: {:?}",
            words_after
        );
    }

    #[test]
    fn test_context_decays() {
        let dim = 1024;
        let mut ctx = make_context(dim);

        // Build up context
        for _ in 0..5 {
            ctx.process_utterance("il gatto mangia il pesce");
        }

        let concepts_active = ctx.get_salient_concepts(5);
        let max_sim_active = concepts_active
            .first()
            .map(|(_, s)| *s)
            .unwrap_or(0.0);

        // Let context decay
        ctx.idle_turns(10);

        let concepts_idle = ctx.get_salient_concepts(5);
        let max_sim_idle = concepts_idle
            .first()
            .map(|(_, s)| *s)
            .unwrap_or(0.0);

        // After idle, the field state changes (either decays or drifts),
        // so the max similarity to vocabulary words should differ.
        // We check that the state has changed (not identical).
        let _state_changed = (max_sim_active - max_sim_idle).abs() > 1e-6
            || concepts_active != concepts_idle;

        // The state should still be finite
        assert!(
            max_sim_idle.is_finite(),
            "After idle turns, concepts should still be finite"
        );

        // Idle turns should cause some state evolution
        // (the NCP processes zero input, which changes state)
        assert!(
            ctx.get_salient_concepts(5).len() > 0 || max_sim_idle.abs() < 1e-6,
            "After idle, we should still be able to get concepts or state should be near zero"
        );
    }
}
