//! HDC-based vocabulary with distributional learning.
//!
//! Each word is represented as a real-valued hypervector. Words are lazily
//! initialised on first access. Distributional learning updates word vectors
//! by bundling context windows from a corpus of sentences.

use std::collections::HashMap;

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::hdc::hypervector::HyperVector;
use crate::hdc::real::RealHV;

/// HDC vocabulary that maps words to real-valued hypervectors.
pub struct Vocabulary {
    /// Hypervector dimensionality.
    pub dim: usize,
    /// Word -> hypervector mapping.
    pub words: HashMap<String, RealHV>,
    /// Deterministic RNG seeded for reproducibility.
    rng: StdRng,
}

impl Vocabulary {
    /// Creates a new empty vocabulary with the given dimensionality and RNG seed.
    pub fn new(dim: usize, seed: u64) -> Self {
        Vocabulary {
            dim,
            words: HashMap::new(),
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Returns the hypervector for `word`, creating a random bipolar vector if
    /// the word is not yet in the vocabulary.
    pub fn get_or_create(&mut self, word: &str) -> &RealHV {
        if !self.words.contains_key(word) {
            let hv = RealHV::random(self.dim, &mut self.rng);
            self.words.insert(word.to_string(), hv);
        }
        self.words.get(word).unwrap()
    }

    /// Returns a clone of the hypervector for `word` (creating it if needed).
    pub fn get_or_create_clone(&mut self, word: &str) -> RealHV {
        self.get_or_create(word).clone()
    }

    /// Distributional learning from a set of tokenized sentences.
    ///
    /// For each word `w` at position `i` in a sentence, gather the hypervectors
    /// of all words within positions `[i - window_size, i + window_size]`
    /// (excluding `w` itself). Bundle those context vectors and update `w`:
    ///
    ///   w_new = bundle_normalized([w_old, context_bundle])
    ///
    /// This causes words that appear in similar contexts to drift closer in
    /// hypervector space.
    /// Distributional learning with IDF weighting.
    pub fn learn_from_context(&mut self, sentences: &[Vec<String>], window_size: usize) {
        self.learn_from_context_idf(sentences, window_size, true);
    }

    /// Distributional learning without IDF weighting (uniform context).
    pub fn learn_from_context_raw(&mut self, sentences: &[Vec<String>], window_size: usize) {
        self.learn_from_context_idf(sentences, window_size, false);
    }

    /// Internal: distributional learning with optional IDF.
    fn learn_from_context_idf(
        &mut self,
        sentences: &[Vec<String>],
        window_size: usize,
        use_idf: bool,
    ) {
        use rayon::prelude::*;

        // Ensure all words exist (must be sequential — mutates self)
        for sentence in sentences {
            for word in sentence {
                self.get_or_create(word);
            }
        }

        // Compute document frequency for IDF weighting
        let n_docs = sentences.len() as f32;
        let doc_freq: std::collections::HashMap<String, usize> = if use_idf {
            let mut df = std::collections::HashMap::new();
            for sentence in sentences {
                let unique: std::collections::HashSet<&String> = sentence.iter().collect();
                for word in unique {
                    *df.entry(word.clone()).or_insert(0) += 1;
                }
            }
            df
        } else {
            std::collections::HashMap::new()
        };

        // Compute all updates in parallel (read-only access to self.words)
        let words_snapshot = &self.words;
        let updates: Vec<(String, RealHV)> = sentences
            .par_iter()
            .flat_map(|sentence| {
                if sentence.is_empty() {
                    return Vec::new();
                }
                let mut local_updates = Vec::new();
                for (i, word) in sentence.iter().enumerate() {
                    let start = i.saturating_sub(window_size);
                    let end = (i + window_size + 1).min(sentence.len());

                    let context_hvs: Vec<RealHV> = (start..end)
                        .filter(|&j| j != i)
                        .filter_map(|j| {
                            let ctx_word = &sentence[j];
                            let hv = words_snapshot.get(ctx_word)?.clone();
                            if use_idf {
                                let df = *doc_freq.get(ctx_word).unwrap_or(&1) as f32;
                                let idf = (n_docs / (1.0 + df)).ln() + 1.0;
                                Some(hv.scale(idf))
                            } else {
                                Some(hv)
                            }
                        })
                        .collect();

                    if context_hvs.is_empty() {
                        continue;
                    }

                    let ctx_refs: Vec<&RealHV> = context_hvs.iter().collect();
                    let context_bundle = RealHV::bundle_normalized(&ctx_refs);

                    if let Some(old_hv) = words_snapshot.get(word) {
                        let new_hv = RealHV::bundle_normalized(&[old_hv, &context_bundle]);
                        local_updates.push((word.clone(), new_hv));
                    }
                }
                local_updates
            })
            .collect();

        // Apply updates sequentially (last write wins per word)
        for (word, hv) in updates {
            self.words.insert(word, hv);
        }
    }

    /// Distributional learning with momentum blending.
    ///
    /// Same as `learn_from_context_raw` but blends old and new vectors using:
    ///   new_hv = bundle_normalized([old_hv.scale(momentum), context_bundle.scale(1.0 - momentum)])
    ///
    /// A momentum of 0.7 means 70% old vector, 30% new context per update,
    /// allowing gradual convergence without catastrophic forgetting.
    pub fn learn_from_context_momentum(
        &mut self,
        sentences: &[Vec<String>],
        window_size: usize,
        momentum: f32,
    ) {
        use rayon::prelude::*;

        // Ensure all words exist
        for sentence in sentences {
            for word in sentence {
                self.get_or_create(word);
            }
        }

        // Parallel update computation
        let words_snapshot = &self.words;
        let updates: Vec<(String, RealHV)> = sentences
            .par_iter()
            .flat_map(|sentence| {
                if sentence.is_empty() {
                    return Vec::new();
                }
                let mut local_updates = Vec::new();
                for (i, word) in sentence.iter().enumerate() {
                    let start = i.saturating_sub(window_size);
                    let end = (i + window_size + 1).min(sentence.len());

                    let context_hvs: Vec<RealHV> = (start..end)
                        .filter(|&j| j != i)
                        .filter_map(|j| words_snapshot.get(&sentence[j]).cloned())
                        .collect();

                    if context_hvs.is_empty() {
                        continue;
                    }

                    let ctx_refs: Vec<&RealHV> = context_hvs.iter().collect();
                    let context_bundle = RealHV::bundle_normalized(&ctx_refs);

                    if let Some(old_hv) = words_snapshot.get(word) {
                        let weighted_old = old_hv.scale(momentum);
                        let weighted_ctx = context_bundle.scale(1.0 - momentum);
                        let new_hv = RealHV::bundle_normalized(&[&weighted_old, &weighted_ctx]);
                        local_updates.push((word.clone(), new_hv));
                    }
                }
                local_updates
            })
            .collect();

        for (word, hv) in updates {
            self.words.insert(word, hv);
        }
    }

    /// Distributional learning with momentum and directional bigrams.
    ///
    /// Same as `learn_from_context_momentum` but also includes directional bigrams
    /// in the context. For each adjacent pair (w_j, w_{j+1}) in the context window,
    /// a bigram hypervector is computed as:
    ///   bigram_hv = bind(permute(w_j, 0), permute(w_{j+1}, 1))
    ///
    /// The permute ensures directionality: bind(perm(A,0), perm(B,1)) ≠ bind(perm(B,0), perm(A,1)).
    /// Bigrams are weighted by `bigram_weight` relative to unigrams (weight 1.0).
    pub fn learn_from_context_with_bigrams(
        &mut self,
        sentences: &[Vec<String>],
        window_size: usize,
        momentum: f32,
        bigram_weight: f32,
    ) {
        use rayon::prelude::*;

        // Ensure all words exist
        for sentence in sentences {
            for word in sentence {
                self.get_or_create(word);
            }
        }

        // Parallel update computation
        let words_snapshot = &self.words;
        let updates: Vec<(String, RealHV)> = sentences
            .par_iter()
            .flat_map(|sentence| {
                if sentence.is_empty() {
                    return Vec::new();
                }
                let mut local_updates = Vec::new();
                for (i, word) in sentence.iter().enumerate() {
                    let start = i.saturating_sub(window_size);
                    let end = (i + window_size + 1).min(sentence.len());

                    // Collect unigram context vectors (weight 1.0)
                    let unigram_hvs: Vec<RealHV> = (start..end)
                        .filter(|&j| j != i)
                        .filter_map(|j| words_snapshot.get(&sentence[j]).cloned())
                        .collect();

                    // Collect directional bigram vectors from adjacent pairs in the window
                    let mut bigram_hvs: Vec<RealHV> = Vec::new();
                    for j in start..end.saturating_sub(1) {
                        // Skip bigrams that include the target word itself
                        if j == i || j + 1 == i {
                            continue;
                        }
                        if let (Some(hv_left), Some(hv_right)) = (
                            words_snapshot.get(&sentence[j]),
                            words_snapshot.get(&sentence[j + 1]),
                        ) {
                            let bigram_hv = RealHV::bind(
                                &hv_left.permute(0),
                                &hv_right.permute(1),
                            );
                            bigram_hvs.push(bigram_hv);
                        }
                    }

                    if unigram_hvs.is_empty() && bigram_hvs.is_empty() {
                        continue;
                    }

                    // Bundle all context: unigrams at weight 1.0, bigrams at bigram_weight
                    let mut all_ctx: Vec<RealHV> = unigram_hvs;
                    for bg in bigram_hvs {
                        all_ctx.push(bg.scale(bigram_weight));
                    }

                    let ctx_refs: Vec<&RealHV> = all_ctx.iter().collect();
                    let context_bundle = RealHV::bundle_normalized(&ctx_refs);

                    if let Some(old_hv) = words_snapshot.get(word) {
                        let weighted_old = old_hv.scale(momentum);
                        let weighted_ctx = context_bundle.scale(1.0 - momentum);
                        let new_hv = RealHV::bundle_normalized(&[&weighted_old, &weighted_ctx]);
                        local_updates.push((word.clone(), new_hv));
                    }
                }
                local_updates
            })
            .collect();

        for (word, hv) in updates {
            self.words.insert(word, hv);
        }
    }

    /// Cosine similarity between two words. Returns 0.0 if either word is unknown.
    pub fn similarity(&self, word_a: &str, word_b: &str) -> f32 {
        match (self.words.get(word_a), self.words.get(word_b)) {
            (Some(a), Some(b)) => a.similarity(b),
            _ => 0.0,
        }
    }

    /// Returns the `k` most similar words to `word` (excluding itself).
    pub fn most_similar(&self, word: &str, k: usize) -> Vec<(String, f32)> {
        let target = match self.words.get(word) {
            Some(hv) => hv,
            None => return Vec::new(),
        };

        let mut sims: Vec<(String, f32)> = self
            .words
            .iter()
            .filter(|(w, _)| w.as_str() != word)
            .map(|(w, hv)| (w.clone(), target.similarity(hv)))
            .collect();

        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sims.truncate(k);
        sims
    }

    /// Analogy (vector arithmetic): a is to b as c is to ?
    ///
    /// Computes: target = b - a + c, then finds the `k` nearest words.
    pub fn analogy(&self, a: &str, b: &str, c: &str, k: usize) -> Vec<(String, f32)> {
        let (hv_a, hv_b, hv_c) = match (
            self.words.get(a),
            self.words.get(b),
            self.words.get(c),
        ) {
            (Some(ha), Some(hb), Some(hc)) => (ha, hb, hc),
            _ => return Vec::new(),
        };

        let neg_a = hv_a.scale(-1.0);
        let target = RealHV::add(&RealHV::add(hv_b, &neg_a), hv_c).normalized();

        let exclude: std::collections::HashSet<&str> = [a, b, c].into_iter().collect();

        let mut sims: Vec<(String, f32)> = self
            .words
            .iter()
            .filter(|(w, _)| !exclude.contains(w.as_str()))
            .map(|(w, hv)| (w.clone(), RealHV::cosine_similarity(&target, hv)))
            .collect();

        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sims.truncate(k);
        sims
    }

    /// HDC-native analogy using binding to capture relations.
    ///
    /// The relation between a and b is encoded as bind(a, b).
    /// To find what has the same relation to c: unbind(relation, c) → nearest.
    /// This is the natural HDC approach: relations ARE hypervectors.
    pub fn analogy_hdc(&self, a: &str, b: &str, c: &str, k: usize) -> Vec<(String, f32)> {
        let (hv_a, hv_b, hv_c) = match (
            self.words.get(a),
            self.words.get(b),
            self.words.get(c),
        ) {
            (Some(ha), Some(hb), Some(hc)) => (ha, hb, hc),
            _ => return Vec::new(),
        };

        // The relation a→b is captured by bind(inverse(a), b)
        // Applying this relation to c: bind(relation, c) = bind(bind(inv(a), b), c)
        let relation = RealHV::bind(&hv_a.inverse(), hv_b);
        let target = RealHV::bind(&relation, hv_c).normalized();

        let exclude: std::collections::HashSet<&str> = [a, b, c].into_iter().collect();

        let mut sims: Vec<(String, f32)> = self
            .words
            .iter()
            .filter(|(w, _)| !exclude.contains(w.as_str()))
            .map(|(w, hv)| (w.clone(), RealHV::cosine_similarity(&target, hv)))
            .collect();

        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sims.truncate(k);
        sims
    }

    /// Combined analogy: tries both vector arithmetic and HDC binding,
    /// returns the best results from either method.
    pub fn analogy_combined(&self, a: &str, b: &str, c: &str, k: usize) -> Vec<(String, f32)> {
        let arith = self.analogy(a, b, c, k);
        let hdc = self.analogy_hdc(a, b, c, k);

        // Merge: take the union, sort by max similarity
        let mut merged: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
        for (w, s) in arith.iter().chain(hdc.iter()) {
            let entry = merged.entry(w.clone()).or_insert(0.0);
            if *s > *entry { *entry = *s; }
        }

        let mut result: Vec<(String, f32)> = merged.into_iter().collect();
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result.truncate(k);
        result
    }

    /// Number of words in the vocabulary.
    pub fn len(&self) -> usize {
        self.words.len()
    }

    /// Whether the vocabulary is empty.
    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    /// Heuristic: considered "trained on a real corpus" if at least 1000
    /// words have been learned (the hardcoded Italian corpus produces ~300).
    pub fn is_trained(&self) -> bool {
        self.words.len() >= 1000
    }

    /// Serialize vocabulary (dim + word→HV map) to a JSON file.
    /// The RNG is not serialized — a fresh one is seeded on load.
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        let snapshot = VocabularySnapshot {
            dim: self.dim,
            words: self.words.clone(),
        };
        let json = serde_json::to_string(&snapshot).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
        })?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, json)
    }

    /// Deserialize vocabulary from a JSON file written by `save()`.
    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let snapshot: VocabularySnapshot = serde_json::from_slice(&bytes).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
        })?;
        Ok(Vocabulary {
            dim: snapshot.dim,
            words: snapshot.words,
            // RNG seed is arbitrary here: post-load the vocabulary is
            // generally used read-only.
            rng: StdRng::seed_from_u64(0xDEAD_BEEF),
        })
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct VocabularySnapshot {
    dim: usize,
    words: HashMap<String, RealHV>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::language::tokenizer::Tokenizer;

    #[test]
    fn test_vocabulary_consistency() {
        let mut vocab = Vocabulary::new(1000, 42);
        let hv1 = vocab.get_or_create("gatto").clone();
        let hv2 = vocab.get_or_create("gatto").clone();
        // Same word should always return the same vector
        assert_eq!(hv1.data, hv2.data, "Same word should return identical HV");
    }

    #[test]
    fn test_vocabulary_lazy() {
        let mut vocab = Vocabulary::new(1000, 42);
        assert!(vocab.is_empty());
        vocab.get_or_create("gatto");
        assert_eq!(vocab.len(), 1);
        vocab.get_or_create("cane");
        assert_eq!(vocab.len(), 2);
        // Accessing existing word should not increase count
        vocab.get_or_create("gatto");
        assert_eq!(vocab.len(), 2);
    }

    #[test]
    fn test_vocabulary_random_words_dissimilar() {
        let mut vocab = Vocabulary::new(2000, 42);
        vocab.get_or_create("gatto");
        vocab.get_or_create("treno");
        let sim = vocab.similarity("gatto", "treno");
        assert!(
            sim.abs() < 0.15,
            "Random words should be nearly orthogonal, got sim={sim}"
        );
    }

    #[test]
    fn test_contextual_learning() {
        let tokenizer = Tokenizer::new();
        let mut vocab = Vocabulary::new(2000, 42);

        // Build sentences where gatto/felino/micio appear in similar contexts
        // and treno appears in very different contexts
        let raw_sentences = vec![
            "il gatto dorme sul divano",
            "il felino dorme sul divano",
            "il micio dorme sul divano",
            "il gatto mangia il pesce",
            "il felino mangia il pesce",
            "il micio mangia il pesce",
            "il gatto gioca con la palla",
            "il felino gioca con la palla",
            "il micio gioca con la palla",
            "il gatto corre nel giardino",
            "il felino corre nel giardino",
            "il micio corre nel giardino",
            "il treno parte dalla stazione",
            "il treno arriva alla stazione",
            "il treno viaggia veloce",
            "il treno trasporta passeggeri",
        ];

        let sentences: Vec<Vec<String>> = raw_sentences
            .iter()
            .map(|s| tokenizer.tokenize(s))
            .collect();

        // Multiple learning passes to strengthen distributional signal
        for _ in 0..5 {
            vocab.learn_from_context(&sentences, 3);
        }

        let sim_gatto_felino = vocab.similarity("gatto", "felino");
        let sim_gatto_treno = vocab.similarity("gatto", "treno");

        assert!(
            sim_gatto_felino > sim_gatto_treno,
            "After learning, gatto-felino ({sim_gatto_felino:.4}) should be more similar \
             than gatto-treno ({sim_gatto_treno:.4})"
        );
    }
}
