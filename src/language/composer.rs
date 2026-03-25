//! Sentence composition via permutation-based encoding.
//!
//! Each sentence is encoded as a single hypervector by:
//! 1. Tokenizing the sentence.
//! 2. Looking up (or creating) the HV for each token.
//! 3. Permuting each token's HV by its positional index.
//! 4. Bundling and normalizing the permuted vectors.
//!
//! This gives an order-sensitive sentence representation.

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::hdc::hypervector::HyperVector;
use crate::hdc::real::RealHV;
use crate::language::tokenizer::Tokenizer;
use crate::language::vocabulary::Vocabulary;

/// Words treated as negation markers. When present in a sentence, the entire
/// sentence vector is bound with the negation operator to flip its meaning.
const NEGATION_WORDS: &[&str] = &["non", "mai", "nessuno"];

/// Sentence composer that converts text into HDC sentence vectors.
pub struct Composer {
    /// Tokenizer used for text preprocessing.
    pub tokenizer: Tokenizer,
    /// Vocabulary mapping words to hypervectors.
    pub vocabulary: Vocabulary,
    /// Whether to include bigram (word-pair) components in sentence encoding.
    pub use_bigrams: bool,
    /// Fixed negation operator hypervector.
    neg_hv: RealHV,
}

impl Composer {
    /// Creates a new Composer with the given HDC dimensionality and RNG seed.
    pub fn new(dim: usize, seed: u64) -> Self {
        // Use a dedicated seed for the negation operator so it doesn't
        // interfere with the vocabulary's RNG stream.
        let mut neg_rng = StdRng::seed_from_u64(seed.wrapping_add(0xDEAD_BEEF));
        let neg_hv = RealHV::random(dim, &mut neg_rng);

        Composer {
            tokenizer: Tokenizer::new(),
            vocabulary: Vocabulary::new(dim, seed),
            use_bigrams: true,
            neg_hv,
        }
    }

    /// Encodes a sentence into a single hypervector.
    ///
    /// Each token at position `i` is encoded as `permute(hv_i, i)`, and
    /// all permuted vectors are bundled and normalized.
    /// Returns `None` if the sentence produces no tokens.
    pub fn encode_sentence(&mut self, text: &str) -> Option<RealHV> {
        let tokens = self.tokenizer.tokenize(text);
        if tokens.is_empty() {
            return None;
        }

        // Detect negation and strip negation tokens from the list.
        let mut has_negation = false;
        let mut content_tokens: Vec<String> = Vec::new();

        for token in &tokens {
            if NEGATION_WORDS.contains(&token.as_str()) {
                has_negation = true;
                continue; // skip the negation marker itself
            }
            content_tokens.push(token.clone());
        }

        if content_tokens.is_empty() {
            return None;
        }

        // Positional unigrams: permute(word_i, i)
        let token_hvs: Vec<RealHV> = content_tokens
            .iter()
            .map(|token| self.vocabulary.get_or_create_clone(token))
            .collect();

        let mut components: Vec<RealHV> = token_hvs
            .iter()
            .enumerate()
            .map(|(i, hv)| hv.permute(i))
            .collect();

        // Bigrams: bind(word_i, word_{i+1}) for consecutive pairs
        if self.use_bigrams && token_hvs.len() >= 2 {
            for i in 0..token_hvs.len() - 1 {
                let bigram = RealHV::bind(&token_hvs[i], &token_hvs[i + 1]);
                components.push(bigram);
            }
        }

        let refs: Vec<&RealHV> = components.iter().collect();
        let sentence_hv = RealHV::bundle_normalized(&refs);

        // Apply global negation: bind the entire sentence vector with NEG.
        if has_negation {
            Some(RealHV::bind(&self.neg_hv, &sentence_hv))
        } else {
            Some(sentence_hv)
        }
    }

    /// Encodes a sentence with context-aware token weighting.
    ///
    /// Each token's weight is modulated by its similarity to the context state:
    ///   weight_i = 1.0 + alpha * cosine_similarity(vocab[token_i], context)
    /// This amplifies tokens relevant to the current context.
    ///
    /// If alpha=0, this is identical to `encode_sentence()`.
    pub fn encode_sentence_contextual(
        &mut self,
        text: &str,
        context: &RealHV,
        alpha: f32,
    ) -> Option<RealHV> {
        let tokens = self.tokenizer.tokenize(text);
        if tokens.is_empty() {
            return None;
        }

        // Detect negation and strip negation tokens (same as encode_sentence).
        let mut has_negation = false;
        let mut content_tokens: Vec<String> = Vec::new();

        for token in &tokens {
            if NEGATION_WORDS.contains(&token.as_str()) {
                has_negation = true;
                continue;
            }
            content_tokens.push(token.clone());
        }

        if content_tokens.is_empty() {
            return None;
        }

        // Build token HVs (no per-token negation).
        let token_hvs: Vec<RealHV> = content_tokens
            .iter()
            .map(|token| self.vocabulary.get_or_create_clone(token))
            .collect();

        // Compute context-weighted positional components.
        let context_is_nonzero = context.norm() > 1e-9;
        let mut components: Vec<RealHV> = token_hvs
            .iter()
            .enumerate()
            .map(|(pos, token_hv)| {
                if context_is_nonzero && alpha.abs() > 1e-9 {
                    let weight =
                        1.0 + alpha * RealHV::cosine_similarity(token_hv, context);
                    token_hv.scale(weight).permute(pos)
                } else {
                    token_hv.permute(pos)
                }
            })
            .collect();

        // Bigrams (same as encode_sentence).
        if self.use_bigrams && token_hvs.len() >= 2 {
            for i in 0..token_hvs.len() - 1 {
                let bigram = RealHV::bind(&token_hvs[i], &token_hvs[i + 1]);
                components.push(bigram);
            }
        }

        let refs: Vec<&RealHV> = components.iter().collect();
        let sentence_hv = RealHV::bundle_normalized(&refs);

        // Apply global negation: bind the entire sentence vector with NEG.
        if has_negation {
            Some(RealHV::bind(&self.neg_hv, &sentence_hv))
        } else {
            Some(sentence_hv)
        }
    }

    /// Computes cosine similarity between two sentence encodings.
    /// Returns 0.0 if either sentence produces no tokens.
    pub fn sentence_similarity(&mut self, text_a: &str, text_b: &str) -> f32 {
        let hv_a = match self.encode_sentence(text_a) {
            Some(hv) => hv,
            None => return 0.0,
        };
        let hv_b = match self.encode_sentence(text_b) {
            Some(hv) => hv,
            None => return 0.0,
        };
        hv_a.similarity(&hv_b)
    }

    /// Encodes a batch of sentences, returning their hypervectors.
    /// Sentences that produce no tokens are skipped (not included in output).
    pub fn batch_encode(&mut self, texts: &[&str]) -> Vec<RealHV> {
        texts
            .iter()
            .filter_map(|text| self.encode_sentence(text))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_negation_handling() {
        use std::fs;
        use std::io::Write;

        let mut composer = Composer::new(2000, 42);

        // --- Core negation pairs ---
        let pairs: Vec<(&str, &str, &str)> = vec![
            ("gatto dorme", "gatto non dorme", "negation"),
            ("cane mangia", "cane non mangia", "negation"),
            ("bambino gioca", "bambino non gioca", "negation"),
            ("sole splende", "sole non splende", "negation"),
            ("pioggia cade", "pioggia non cade", "negation"),
            // Baseline: different verbs (no negation)
            ("gatto dorme", "gatto mangia", "different_verb"),
            ("cane mangia", "cane corre", "different_verb"),
            // Both negated: should mirror positive relationship
            ("gatto non dorme", "gatto non mangia", "both_negated"),
            ("cane non mangia", "cane non corre", "both_negated"),
            // Backward compatibility: no negation words at all
            ("gatto dorme", "gatto dorme", "identity"),
            ("gatto mangia pesce", "pesce mangia gatto", "word_order"),
        ];

        println!("\n--- R1: Negation handling results ---\n");
        println!("  {:25} | {:25} | {:>8} | {}", "Sentence A", "Sentence B", "Sim", "Type");
        println!("  {:-<25}-+-{:-<25}-+-{:-<8}-+-{:-<15}", "", "", "", "");

        let mut csv_rows: Vec<String> = vec!["sentence_a,sentence_b,similarity,pair_type".to_string()];

        for (a, b, ptype) in &pairs {
            let sim = composer.sentence_similarity(a, b);
            println!("  {:25} | {:25} | {:>8.4} | {}", a, b, sim, ptype);
            csv_rows.push(format!("\"{}\",\"{}\",{:.6},{}", a, b, sim, ptype));
        }

        // Save CSV
        fs::create_dir_all("results/improvements/r1_negation").expect("create dir");
        let mut f = fs::File::create("results/improvements/r1_negation/negation_pairs.csv")
            .expect("create csv");
        for row in &csv_rows {
            writeln!(f, "{}", row).unwrap();
        }
        println!("\n  Saved results/improvements/r1_negation/negation_pairs.csv");

        // --- Assertions ---
        // Global negation should make vectors quasi-orthogonal (sim < 0.2)
        let neg_sim = composer.sentence_similarity("gatto dorme", "gatto non dorme");
        assert!(
            neg_sim < 0.2,
            "Negation should reduce similarity below 0.2: got {neg_sim}"
        );

        // Identity should be 1.0
        let id_sim = composer.sentence_similarity("gatto dorme", "gatto dorme");
        assert!(
            (id_sim - 1.0).abs() < 1e-5,
            "Identity should be ~1.0, got {id_sim}"
        );

        // Sentences without negation words should be backward-compatible:
        // Verify that non-negation sentences produce deterministic results
        let hv1 = composer.encode_sentence("gatto mangia pesce").unwrap();
        let hv2 = composer.encode_sentence("gatto mangia pesce").unwrap();
        assert!(
            (RealHV::cosine_similarity(&hv1, &hv2) - 1.0).abs() < 1e-5,
            "Non-negation sentences should be identical across calls"
        );
    }

    #[test]
    fn test_sentence_order_matters() {
        let mut composer = Composer::new(2000, 42);

        let hv_ab = composer.encode_sentence("gatto mangia pesce").unwrap();
        let hv_ba = composer.encode_sentence("pesce mangia gatto").unwrap();

        let sim = RealHV::cosine_similarity(&hv_ab, &hv_ba);
        // Same words in different order should produce different (not identical) vectors
        assert!(
            sim < 0.95,
            "Different word order should produce different vectors, got sim={sim}"
        );
    }

    #[test]
    fn test_identical_sentences_same_vector() {
        let mut composer = Composer::new(2000, 42);
        let hv1 = composer.encode_sentence("il gatto dorme").unwrap();
        let hv2 = composer.encode_sentence("il gatto dorme").unwrap();
        let sim = RealHV::cosine_similarity(&hv1, &hv2);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "Identical sentences should have similarity ~1.0, got {sim}"
        );
    }

    #[test]
    fn test_batch_encode() {
        let mut composer = Composer::new(1000, 42);
        let texts = vec!["il gatto mangia", "il cane corre", "di e o ma"];
        let hvs = composer.batch_encode(&texts);
        // Third sentence is all stopwords, so only 2 results
        assert_eq!(hvs.len(), 2, "Should have 2 valid encodings (third is all stopwords)");
    }

    #[test]
    fn test_contextual_encoding_alpha_zero_matches_plain() {
        let mut composer = Composer::new(2000, 42);
        let context = RealHV::random(2000, &mut rand::rngs::StdRng::seed_from_u64(99));

        let hv_plain = composer.encode_sentence("gatto dorme divano").unwrap();
        let hv_ctx = composer
            .encode_sentence_contextual("gatto dorme divano", &context, 0.0)
            .unwrap();

        let sim = RealHV::cosine_similarity(&hv_plain, &hv_ctx);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "alpha=0 contextual encoding should match plain encoding, got sim={sim}"
        );
    }

    #[test]
    fn test_contextual_encoding_amplifies_relevant_tokens() {
        let mut composer = Composer::new(2000, 42);

        // Build a context that is close to "gatto"
        let cat_context = composer.vocabulary.get_or_create_clone("gatto");

        // Encode "gatto dorme" with and without context
        let hv_plain = composer.encode_sentence("gatto dorme").unwrap();
        let hv_ctx = composer
            .encode_sentence_contextual("gatto dorme", &cat_context, 0.5)
            .unwrap();

        // The contextual version should differ from the plain version
        let sim = RealHV::cosine_similarity(&hv_plain, &hv_ctx);
        assert!(
            sim < 0.999,
            "Contextual encoding with nonzero alpha should differ from plain, got sim={sim}"
        );

        // The contextual encoding should be more similar to the cat context
        // than the plain encoding is (because "gatto" gets amplified).
        let plain_to_ctx = RealHV::cosine_similarity(&hv_plain, &cat_context);
        let contextual_to_ctx = RealHV::cosine_similarity(&hv_ctx, &cat_context);
        println!(
            "  plain->cat_ctx={:.4}, contextual->cat_ctx={:.4}",
            plain_to_ctx, contextual_to_ctx
        );
        assert!(
            contextual_to_ctx > plain_to_ctx,
            "Contextual encoding should be closer to context: contextual={contextual_to_ctx}, plain={plain_to_ctx}"
        );
    }

    #[test]
    fn test_contextual_encoding_with_zero_context_matches_plain() {
        let mut composer = Composer::new(2000, 42);
        let zero_ctx = RealHV::zero(2000);

        let hv_plain = composer.encode_sentence("gatto dorme divano").unwrap();
        let hv_ctx = composer
            .encode_sentence_contextual("gatto dorme divano", &zero_ctx, 0.5)
            .unwrap();

        let sim = RealHV::cosine_similarity(&hv_plain, &hv_ctx);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "Zero context should produce same result as plain encoding, got sim={sim}"
        );
    }

    #[test]
    fn test_similar_sentences_closer() {
        let mut composer = Composer::new(2000, 42);
        let hv_a = composer.encode_sentence("gatto mangia pesce").unwrap();
        let hv_b = composer.encode_sentence("gatto mangia carne").unwrap();
        let hv_c = composer.encode_sentence("treno parte stazione").unwrap();

        let sim_ab = RealHV::cosine_similarity(&hv_a, &hv_b);
        let sim_ac = RealHV::cosine_similarity(&hv_a, &hv_c);

        // Sentences sharing more words should be more similar
        assert!(
            sim_ab > sim_ac,
            "Sentences sharing words should be more similar: ab={sim_ab}, ac={sim_ac}"
        );
    }
}
