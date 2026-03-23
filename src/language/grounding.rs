//! Symbol grounding: words to contextual hypervectors with structural roles.
//!
//! The GroundingSystem assigns semantic roles (agente, azione, paziente, etc.)
//! to tokens in a sentence using simple heuristic rules, then encodes the
//! grounded meaning as a single hypervector via bind+bundle.

use std::collections::HashMap;

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::hdc::real::RealHV;
use crate::language::vocabulary::Vocabulary;

/// Assigns semantic roles to sentence tokens and encodes them as a
/// single grounded hypervector.
pub struct GroundingSystem {
    /// Role name -> role hypervector.
    roles: HashMap<String, RealHV>,
    /// HDC dimensionality.
    dim: usize,
}

impl GroundingSystem {
    /// Creates a new GroundingSystem with predefined Italian semantic roles.
    pub fn new(dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let role_names = ["agente", "azione", "paziente", "proprietà", "luogo", "tempo"];

        let roles: HashMap<String, RealHV> = role_names
            .iter()
            .map(|&name| (name.to_string(), RealHV::random(dim, &mut rng)))
            .collect();

        GroundingSystem { roles, dim }
    }

    /// Ground a tokenized sentence into a single hypervector.
    ///
    /// Uses a simple heuristic for Italian SVO order:
    /// - First token -> agente (subject/agent)
    /// - Second token -> azione (verb/action)
    /// - Third token onwards -> paziente (object/patient)
    ///
    /// The grounded representation is: bundle(bind(role_i, token_i) for each i).
    pub fn ground_sentence(&self, tokens: &[String], vocab: &mut Vocabulary) -> RealHV {
        if tokens.is_empty() {
            return RealHV::zero(self.dim);
        }

        let mut bindings: Vec<RealHV> = Vec::new();

        for (i, token) in tokens.iter().enumerate() {
            let role_name = match i {
                0 => "agente",
                1 => "azione",
                _ => "paziente",
            };

            if let Some(role_hv) = self.roles.get(role_name) {
                let token_hv = vocab.get_or_create_clone(token);
                bindings.push(RealHV::bind(role_hv, &token_hv));
            }
        }

        if bindings.is_empty() {
            return RealHV::zero(self.dim);
        }

        let refs: Vec<&RealHV> = bindings.iter().collect();
        RealHV::bundle_normalized(&refs)
    }

    /// Query a grounded hypervector for a specific role.
    ///
    /// Unbinds the role from the grounded HV and finds the nearest
    /// vocabulary words to the result.
    pub fn query_role(
        &self,
        grounded_hv: &RealHV,
        role: &str,
        vocab: &Vocabulary,
    ) -> Vec<(String, f32)> {
        let role_hv = match self.roles.get(role) {
            Some(hv) => hv,
            None => return Vec::new(),
        };

        // Unbind: grounded * role_inverse recovers the token bound to that role
        let recovered = RealHV::bind(grounded_hv, &role_hv.inverse());

        let mut sims: Vec<(String, f32)> = vocab
            .words
            .iter()
            .map(|(word, hv)| (word.clone(), RealHV::cosine_similarity(&recovered, hv)))
            .collect();

        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sims.truncate(10);
        sims
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::language::tokenizer::Tokenizer;

    #[test]
    fn test_grounding_roles() {
        let dim = 2000;
        let gs = GroundingSystem::new(dim, 42);
        let mut vocab = Vocabulary::new(dim, 99);
        let tokenizer = Tokenizer::new();

        // "gatto mangia topo" -> tokens: ["gatto", "mangia", "topo"]
        let tokens = tokenizer.tokenize("gatto mangia topo");
        assert_eq!(tokens, vec!["gatto", "mangia", "topo"]);

        let grounded = gs.ground_sentence(&tokens, &mut vocab);

        // Query the agente role -> should recover "gatto"
        let agente_results = gs.query_role(&grounded, "agente", &vocab);
        let agente_words: Vec<&str> = agente_results.iter().map(|(w, _)| w.as_str()).collect();

        assert!(
            agente_words.contains(&"gatto"),
            "query_role('agente') should contain 'gatto', got: {:?}",
            agente_results
        );

        // Query the azione role -> should recover "mangia"
        let azione_results = gs.query_role(&grounded, "azione", &vocab);
        let azione_words: Vec<&str> = azione_results.iter().map(|(w, _)| w.as_str()).collect();

        assert!(
            azione_words.contains(&"mangia"),
            "query_role('azione') should contain 'mangia', got: {:?}",
            azione_results
        );

        // Query the paziente role -> should recover "topo"
        let paziente_results = gs.query_role(&grounded, "paziente", &vocab);
        let paziente_words: Vec<&str> = paziente_results.iter().map(|(w, _)| w.as_str()).collect();

        assert!(
            paziente_words.contains(&"topo"),
            "query_role('paziente') should contain 'topo', got: {:?}",
            paziente_results
        );
    }
}
