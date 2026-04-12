//! Corpus loader and English tokenizer for multi-domain text files.
//!
//! Reads `.txt` files from one or more directories, tokenizes them in
//! English with stopword removal, and drives distributional learning on
//! a [`Vocabulary`].

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::language::vocabulary::Vocabulary;

// ---------------------------------------------------------------------------
// CorpusConfig / CorpusStats
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CorpusConfig {
    pub paths: Vec<PathBuf>,
    pub max_words: usize,
    pub min_frequency: usize,
    pub context_window: usize,
    pub learning_passes: usize,
}

impl Default for CorpusConfig {
    fn default() -> Self {
        CorpusConfig {
            paths: Vec::new(),
            max_words: 50_000,
            min_frequency: 5,
            context_window: 3,
            learning_passes: 3,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CorpusStats {
    pub total_tokens: usize,
    pub unique_words: usize,
    pub words_learned: usize,
    pub words_discarded: usize,
    pub avg_context_richness: f32,
    pub training_time_secs: f32,
}

// ---------------------------------------------------------------------------
// EnglishTokenizer
// ---------------------------------------------------------------------------

pub struct EnglishTokenizer {
    stopwords: HashSet<String>,
}

impl EnglishTokenizer {
    pub fn new() -> Self {
        let sw = [
            "the","a","an","and","or","but","in","on","at","to","for","of","with",
            "by","from","is","are","was","were","be","been","being","have","has",
            "had","do","does","did","will","would","could","should","may","might",
            "this","that","these","those","it","its","he","she","they","we","you",
            "i","me","him","her","us","them","my","your","his","our","their",
            "not","no","nor","so","yet","both","either","neither","as","if",
            "then","than","because","while","although","though","since",
            "which","who","whom","what","when","where","how","all","each",
            "every","any","some","more","most","other","such","same",
            "can","about","also","just","only","very","too","up","out",
            "there","here","now","even","still","back","over","down",
            "after","before","into","during","between","under","through",
            "new","like","make","many","much","own","well","way",
            "said","one","two","three","first","last","time","been","made",
            "dont","didnt","im","youre","thats","hes","shes","theyre",
        ];
        EnglishTokenizer {
            stopwords: sw.iter().map(|s| s.to_string()).collect(),
        }
    }

    pub fn tokenize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|w| w.len() >= 2 && !self.stopwords.contains(*w))
            .map(|w| w.to_string())
            .collect()
    }

    pub fn tokenize_into_sentences(&self, text: &str) -> Vec<Vec<String>> {
        text.split(|c: char| c == '.' || c == '!' || c == '?' || c == '\n')
            .map(|sent| self.tokenize(sent))
            .filter(|tokens| tokens.len() >= 3)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// CorpusLoader
// ---------------------------------------------------------------------------

pub struct CorpusLoader;

impl CorpusLoader {
    /// Load text from all `.txt` files under `config.paths`, tokenize,
    /// filter by frequency, and train the vocabulary distribuzionally.
    pub fn load_and_train(
        config: &CorpusConfig,
        vocabulary: &mut Vocabulary,
    ) -> CorpusStats {
        let start = Instant::now();
        let tokenizer = EnglishTokenizer::new();

        // 1. Collect all .txt file paths.
        let mut files: Vec<PathBuf> = Vec::new();
        for p in &config.paths {
            if p.is_file() {
                files.push(p.clone());
            } else if p.is_dir() {
                if let Ok(entries) = std::fs::read_dir(p) {
                    for e in entries.flatten() {
                        let path = e.path();
                        if path.extension().map(|e| e == "txt").unwrap_or(false) {
                            files.push(path);
                        }
                    }
                }
            }
        }

        // 2. Read + tokenize (sentence-level). Limit per-file to keep
        //    memory tractable (~2 MB text per file).
        let mut all_sentences: Vec<Vec<String>> = Vec::new();
        let mut total_tokens = 0usize;
        let mut word_freq: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

        for file in &files {
            let text = match std::fs::read_to_string(file) {
                Ok(t) => t,
                Err(_) => continue,
            };
            // Cap per file.
            let capped: String = text.chars().take(2_000_000).collect();
            let sents = tokenizer.tokenize_into_sentences(&capped);
            for sent in &sents {
                for w in sent {
                    *word_freq.entry(w.clone()).or_insert(0) += 1;
                    total_tokens += 1;
                }
            }
            all_sentences.extend(sents);
        }

        let unique_words = word_freq.len();

        // 3. Filter by frequency: discard words below min_frequency.
        let allowed: HashSet<String> = word_freq
            .iter()
            .filter(|(_, &f)| f >= config.min_frequency)
            .map(|(w, _)| w.clone())
            .collect();

        let words_learned = allowed.len().min(config.max_words);
        let words_discarded = unique_words - allowed.len();

        // Filter sentences to only allowed tokens.
        let filtered_sentences: Vec<Vec<String>> = all_sentences
            .iter()
            .map(|sent| {
                sent.iter()
                    .filter(|w| allowed.contains(w.as_str()))
                    .cloned()
                    .collect()
            })
            .filter(|sent: &Vec<String>| sent.len() >= 2)
            .collect();

        // 4. Ensure all allowed words exist in the vocabulary.
        for w in &allowed {
            vocabulary.get_or_create(w);
        }

        // 5. Distributional learning passes using momentum for faster convergence.
        for _pass in 0..config.learning_passes {
            vocabulary.learn_from_context_momentum(
                &filtered_sentences,
                config.context_window,
                0.7, // momentum factor
            );
        }

        // 6. Trim vocabulary to max_words by frequency if needed.
        if vocabulary.len() > config.max_words {
            // Keep the top-N by frequency.
            let mut by_freq: Vec<(String, usize)> = word_freq
                .iter()
                .filter(|(w, _)| vocabulary.words.contains_key(w.as_str()))
                .map(|(w, &f)| (w.clone(), f))
                .collect();
            by_freq.sort_by(|a, b| b.1.cmp(&a.1));
            let keep: HashSet<String> = by_freq
                .into_iter()
                .take(config.max_words)
                .map(|(w, _)| w)
                .collect();
            vocabulary.words.retain(|w, _| keep.contains(w));
        }

        // 7. Context richness (mean tokens per sentence).
        let avg_context_richness = if filtered_sentences.is_empty() {
            0.0
        } else {
            filtered_sentences.iter().map(|s| s.len() as f32).sum::<f32>()
                / filtered_sentences.len() as f32
        };

        CorpusStats {
            total_tokens,
            unique_words,
            words_learned,
            words_discarded,
            avg_context_richness,
            training_time_secs: start.elapsed().as_secs_f32(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Test 1: English tokenizer
    #[test]
    fn test_english_tokenizer() {
        let tok = EnglishTokenizer::new();
        let tokens = tok.tokenize("The quick brown fox jumps over the lazy dog.");
        assert!(!tokens.contains(&"the".to_string()), "stopword 'the' should be removed");
        assert!(!tokens.contains(&"over".to_string()), "stopword 'over' should be removed");
        assert!(tokens.contains(&"quick".to_string()));
        assert!(tokens.contains(&"brown".to_string()));
        assert!(tokens.contains(&"fox".to_string()));
        assert!(tokens.contains(&"jumps".to_string()));
        assert!(tokens.contains(&"lazy".to_string()));
        assert!(tokens.contains(&"dog".to_string()));
        // All lowercase.
        for t in &tokens { assert_eq!(*t, t.to_lowercase()); }
    }

    // Test 2: corpus loader on small inline corpus
    #[test]
    fn test_corpus_loader_small() {
        use std::io::Write;

        let dir = std::path::PathBuf::from("results/vocabulary");
        let _ = std::fs::create_dir_all(&dir);
        let tmpfile = dir.join("_test_corpus.txt");
        {
            let mut f = std::fs::File::create(&tmpfile).unwrap();
            for _ in 0..50 {
                writeln!(f, "The cat sat on the mat near the dog.").unwrap();
                writeln!(f, "Dogs and cats are common household pets.").unwrap();
                writeln!(f, "Water flows from the river to the sea.").unwrap();
                writeln!(f, "The quick brown fox jumps over the lazy dog.").unwrap();
            }
        }

        let mut vocab = Vocabulary::new(1024, 42);
        let cfg = CorpusConfig {
            paths: vec![tmpfile.clone()],
            max_words: 10_000,
            min_frequency: 2,
            context_window: 2,
            learning_passes: 2,
        };
        let stats = CorpusLoader::load_and_train(&cfg, &mut vocab);

        assert!(stats.words_learned > 0, "should learn some words");
        assert!(vocab.words.contains_key("cat"), "cat should be in vocab");
        assert!(vocab.words.contains_key("dog"), "dog should be in vocab");

        let _ = std::fs::remove_file(&tmpfile);
    }

    // Test 3: vocabulary serialization round-trip
    #[test]
    fn test_vocabulary_serialization() {
        use crate::hdc::real::RealHV;

        let mut vocab = Vocabulary::new(1024, 42);
        for w in &["hello", "world", "test", "example"] {
            vocab.get_or_create(w);
        }

        let dir = std::path::PathBuf::from("results/vocabulary");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("_test_roundtrip.vocab");

        vocab.save(&path).expect("save should work");
        let loaded = Vocabulary::load(&path).expect("load should work");

        assert_eq!(loaded.len(), vocab.len());
        assert_eq!(loaded.dim, vocab.dim);
        for (w, hv) in &vocab.words {
            let loaded_hv = loaded.words.get(w).expect("word should exist after load");
            let sim = RealHV::cosine_similarity(hv, loaded_hv);
            assert!(sim > 0.999, "round-trip similarity for '{}': {}", w, sim);
        }

        let _ = std::fs::remove_file(&path);
    }
}
