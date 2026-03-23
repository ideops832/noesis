//! Minimal Italian tokenization.
//!
//! Handles lowercasing, stopword removal, punctuation stripping, and
//! Italian apostrophe elision (e.g. "l'uomo" -> ["uomo"]).

use std::collections::HashSet;

/// A minimal tokenizer for Italian text.
pub struct Tokenizer {
    /// Set of stopwords to remove.
    pub stopwords: HashSet<String>,
    /// Whether to apply basic stemming (currently unused, reserved).
    pub use_stemming: bool,
}

/// Italian article prefixes that appear before apostrophes (elision).
const ELISION_PREFIXES: &[&str] = &[
    "l", "d", "un", "all", "dall", "dell", "nell", "sull", "coll",
    "quest", "quell", "c", "s",
];

impl Tokenizer {
    /// Creates a new Tokenizer with default Italian stopwords.
    pub fn new() -> Self {
        let stopwords: HashSet<String> = [
            "il", "lo", "la", "i", "gli", "le", "un", "uno", "una",
            "di", "a", "da", "in", "con", "su", "per", "tra", "fra",
            "e", "o", "ma", "che", "non", "si", "ci", "ne", "se",
            "è", "sono", "ha", "hanno", "era", "essere", "avere",
            "come", "anche", "più", "molto", "questo", "quello",
            "suo", "loro", "mio", "tuo",
            "al", "del", "nel", "dal", "sul", "col",
            "allo", "dello", "nello", "dallo", "sullo",
            "alla", "della", "nella", "dalla", "sulla",
            "alle", "delle", "nelle", "dalle", "sulle",
            "ai", "dei", "nei", "dai", "sui",
            "agli", "degli", "negli", "dagli", "sugli",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        Tokenizer {
            stopwords,
            use_stemming: false,
        }
    }

    /// Tokenizes input text into a list of cleaned tokens.
    ///
    /// Steps:
    /// 1. Lowercase the input.
    /// 2. Split on whitespace and punctuation (keeping apostrophes for elision handling).
    /// 3. Handle Italian apostrophe elision: "l'uomo" -> "uomo".
    /// 4. Remove stopwords and empty tokens.
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let lower = text.to_lowercase();

        // Split on whitespace first, then process each chunk
        let mut tokens = Vec::new();
        for chunk in lower.split_whitespace() {
            // Strip leading/trailing punctuation except apostrophes (handled separately)
            let chunk = chunk.trim_matches(|c: char| c.is_ascii_punctuation() && c != '\'');
            if chunk.is_empty() {
                continue;
            }

            // Handle apostrophe elision
            if let Some(pos) = chunk.find('\'') {
                let prefix = &chunk[..pos];
                let suffix = &chunk[pos + 1..];

                let is_elision = ELISION_PREFIXES.contains(&prefix);

                if is_elision && !suffix.is_empty() {
                    // Drop the article prefix, keep the content word
                    let clean = suffix.trim_matches(|c: char| c.is_ascii_punctuation());
                    if !clean.is_empty() && !self.stopwords.contains(clean) {
                        tokens.push(clean.to_string());
                    }
                } else {
                    // Not an elision pattern: keep both parts as separate tokens
                    for part in [prefix, suffix] {
                        let clean = part.trim_matches(|c: char| c.is_ascii_punctuation());
                        if !clean.is_empty() && !self.stopwords.contains(clean) {
                            tokens.push(clean.to_string());
                        }
                    }
                }
            } else {
                // No apostrophe: strip punctuation and check stopword
                let clean = chunk.trim_matches(|c: char| c.is_ascii_punctuation());
                if !clean.is_empty() && !self.stopwords.contains(clean) {
                    tokens.push(clean.to_string());
                }
            }
        }

        tokens
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_basic() {
        let tok = Tokenizer::new();
        let tokens = tok.tokenize("Il gatto mangia il pesce");
        // "il" is a stopword, so it should be removed
        assert!(tokens.contains(&"gatto".to_string()));
        assert!(tokens.contains(&"mangia".to_string()));
        assert!(tokens.contains(&"pesce".to_string()));
        assert!(!tokens.contains(&"il".to_string()));
    }

    #[test]
    fn test_tokenizer_stopwords() {
        let tok = Tokenizer::new();
        let tokens = tok.tokenize("di la con per tra fra e o ma che non");
        assert!(tokens.is_empty(), "All stopwords should be removed, got: {:?}", tokens);
    }

    #[test]
    fn test_tokenizer_apostrophe() {
        let tok = Tokenizer::new();
        // "l'" is an elision prefix -> drop it, keep "uomo"
        let tokens = tok.tokenize("l'uomo cammina");
        assert!(tokens.contains(&"uomo".to_string()));
        assert!(tokens.contains(&"cammina".to_string()));
        assert!(!tokens.contains(&"l".to_string()));
    }

    #[test]
    fn test_tokenizer_punctuation() {
        let tok = Tokenizer::new();
        let tokens = tok.tokenize("Ciao, mondo! Come stai?");
        assert!(tokens.contains(&"ciao".to_string()));
        assert!(tokens.contains(&"mondo".to_string()));
        assert!(tokens.contains(&"stai".to_string()));
    }

    #[test]
    fn test_tokenizer_case_insensitive() {
        let tok = Tokenizer::new();
        let tokens = tok.tokenize("GATTO Mangia PESCE");
        assert!(tokens.contains(&"gatto".to_string()));
        assert!(tokens.contains(&"mangia".to_string()));
        assert!(tokens.contains(&"pesce".to_string()));
    }
}
