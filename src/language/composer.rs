//! Sentence composition via permutation-based encoding.
//!
//! Each sentence is encoded as a single hypervector by:
//! 1. Tokenizing the sentence.
//! 2. Looking up (or creating) the HV for each token.
//! 3. Permuting each token's HV by its positional index.
//! 4. Bundling and normalizing the permuted vectors.
//!
//! This gives an order-sensitive sentence representation.

use crate::hdc::hypervector::HyperVector;
use crate::hdc::real::RealHV;
use crate::language::tokenizer::Tokenizer;
use crate::language::vocabulary::Vocabulary;

/// Sentence composer that converts text into HDC sentence vectors.
pub struct Composer {
    /// Tokenizer used for text preprocessing.
    pub tokenizer: Tokenizer,
    /// Vocabulary mapping words to hypervectors.
    pub vocabulary: Vocabulary,
    /// Whether to include bigram (word-pair) components in sentence encoding.
    pub use_bigrams: bool,
}

impl Composer {
    /// Creates a new Composer with the given HDC dimensionality and RNG seed.
    pub fn new(dim: usize, seed: u64) -> Self {
        Composer {
            tokenizer: Tokenizer::new(),
            vocabulary: Vocabulary::new(dim, seed),
            use_bigrams: true,
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

        // Positional unigrams: permute(word_i, i)
        let token_hvs: Vec<RealHV> = tokens
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
        Some(RealHV::bundle_normalized(&refs))
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
