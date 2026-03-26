use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use crate::hdc::real::RealHV;
use crate::language::vocabulary::Vocabulary;

#[derive(Serialize, Clone)]
pub struct DashboardFrame {
    pub timestamp_ms: u64,
    pub field_state: FieldState,
    pub salient_concepts: Vec<ConceptScore>,
    pub metrics: Metrics,
    pub last_event: Option<Event>,
    pub narrative: Vec<NarrativePoint>,
}

#[derive(Serialize, Clone)]
pub struct FieldState {
    pub dominant_concept: String,
    pub dominant_similarity: f32,
    pub category: String,
    pub activation: f32,
    pub fast_activation: f32,
    pub medium_activation: f32,
    pub slow_activation: f32,
    pub velocity: f32,
}

#[derive(Serialize, Clone)]
pub struct ConceptScore {
    pub word: String,
    pub similarity: f32,
    pub category: String,
}

#[derive(Serialize, Clone)]
pub struct Metrics {
    pub coherence: f32,
    pub novelty: f32,
    pub total_steps: u64,
    pub turn_count: u32,
    pub idle_time: f32,
}

#[derive(Serialize, Clone)]
pub struct Event {
    pub event_type: String,
    pub text: String,
    pub result: Option<String>,
}

#[derive(Serialize, Clone)]
pub struct NarrativePoint {
    pub timestamp_ms: u64,
    pub dominant_concept: String,
    pub category: String,
    pub coherence: f32,
    pub event_type: String,
}

/// Category prototype words for semantic categorization.
const CATEGORY_PROTOTYPES: &[(&str, &[&str])] = &[
    ("animali", &["gatto", "cane", "pesce", "uccello", "cavallo", "topo", "leone", "tigre", "felino", "animale"]),
    ("cibo", &["mangia", "cucina", "pane", "pasta", "pizza", "carne", "pesce", "frutta", "verdura", "cibo"]),
    ("lavoro", &["lavora", "ufficio", "progetto", "scrivania", "computer", "riunione", "collega", "lavoro"]),
    ("natura", &["sole", "pioggia", "vento", "mare", "montagna", "fiume", "albero", "fiore", "cielo", "natura"]),
    ("famiglia", &["mamma", "papa", "figlio", "figlia", "fratello", "sorella", "nonna", "nonno", "famiglia"]),
];

/// Categorize a word by checking its similarity to category prototype bundles.
///
/// Returns the category name with the highest average similarity, or "altro"
/// if no category scores above a minimal threshold.
pub fn categorize_word(word: &str, vocab: &Vocabulary) -> String {
    let word_hv = match vocab.words.get(word) {
        Some(hv) => hv,
        None => return "altro".to_string(),
    };

    let mut best_category = "altro";
    let mut best_score: f32 = 0.05; // minimum threshold

    for &(category, prototypes) in CATEGORY_PROTOTYPES {
        let mut total_sim: f32 = 0.0;
        let mut count: usize = 0;

        for &proto_word in prototypes {
            if let Some(proto_hv) = vocab.words.get(proto_word) {
                let sim = RealHV::cosine_similarity(word_hv, proto_hv);
                total_sim += sim;
                count += 1;
            }
        }

        if count > 0 {
            let avg_sim = total_sim / count as f32;
            if avg_sim > best_score {
                best_score = avg_sim;
                best_category = category;
            }
        }
    }

    best_category.to_string()
}

/// Build a dashboard frame from the current field state and salient concepts.
pub fn build_frame(
    salient: &[(String, f32)],
    vocab: &Vocabulary,
    fast_norm: f32,
    medium_norm: f32,
    slow_norm: f32,
    velocity: f32,
    novelty: f32,
    total_steps: u64,
    turn_count: u32,
    idle_time: f32,
    narrative: &[NarrativePoint],
    last_event: Option<Event>,
) -> DashboardFrame {
    let timestamp_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;

    let (dominant_concept, dominant_similarity) = salient
        .first()
        .map(|(w, s)| (w.clone(), *s))
        .unwrap_or_else(|| ("(vuoto)".to_string(), 0.0));

    let category = categorize_word(&dominant_concept, vocab);

    let activation = fast_norm * 0.2 + medium_norm * 0.4 + slow_norm * 0.4;

    let field_state = FieldState {
        dominant_concept: dominant_concept.clone(),
        dominant_similarity,
        category: category.clone(),
        activation,
        fast_activation: fast_norm,
        medium_activation: medium_norm,
        slow_activation: slow_norm,
        velocity,
    };

    let salient_concepts: Vec<ConceptScore> = salient
        .iter()
        .map(|(word, sim)| ConceptScore {
            word: word.clone(),
            similarity: *sim,
            category: categorize_word(word, vocab),
        })
        .collect();

    let coherence = if salient.len() >= 2 {
        // Coherence: how similar are the top concepts to each other (via their scores)
        let top_sim = salient[0].1;
        let second_sim = salient[1].1;
        if top_sim > 1e-8 { second_sim / top_sim } else { 0.0 }
    } else {
        0.0
    };

    let metrics = Metrics {
        coherence,
        novelty,
        total_steps,
        turn_count,
        idle_time,
    };

    DashboardFrame {
        timestamp_ms,
        field_state,
        salient_concepts,
        metrics,
        last_event,
        narrative: narrative.to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_state_serialization() {
        let frame = DashboardFrame {
            timestamp_ms: 1234567890,
            field_state: FieldState {
                dominant_concept: "gatto".to_string(),
                dominant_similarity: 0.95,
                category: "animali".to_string(),
                activation: 0.8,
                fast_activation: 0.9,
                medium_activation: 0.7,
                slow_activation: 0.6,
                velocity: 0.1,
            },
            salient_concepts: vec![
                ConceptScore {
                    word: "gatto".to_string(),
                    similarity: 0.95,
                    category: "animali".to_string(),
                },
                ConceptScore {
                    word: "cane".to_string(),
                    similarity: 0.80,
                    category: "animali".to_string(),
                },
            ],
            metrics: Metrics {
                coherence: 0.84,
                novelty: 0.1,
                total_steps: 42,
                turn_count: 3,
                idle_time: 1.5,
            },
            last_event: Some(Event {
                event_type: "feed".to_string(),
                text: "il gatto dorme".to_string(),
                result: None,
            }),
            narrative: vec![
                NarrativePoint {
                    timestamp_ms: 1234567800,
                    dominant_concept: "gatto".to_string(),
                    category: "animali".to_string(),
                    coherence: 0.9,
                    event_type: "feed".to_string(),
                },
            ],
        };

        let json = serde_json::to_string(&frame).expect("serialization should succeed");
        assert!(json.contains("\"gatto\""), "JSON should contain dominant concept");
        assert!(json.contains("\"animali\""), "JSON should contain category");
        assert!(json.contains("1234567890"), "JSON should contain timestamp");
        assert!(json.contains("\"feed\""), "JSON should contain event type");
        assert!(json.contains("\"coherence\""), "JSON should contain metrics");

        // Verify it round-trips through serde_json::Value
        let value: serde_json::Value = serde_json::from_str(&json).expect("should parse as JSON");
        assert_eq!(value["field_state"]["dominant_concept"], "gatto");
        assert_eq!(value["metrics"]["total_steps"], 42);
        assert_eq!(value["salient_concepts"].as_array().unwrap().len(), 2);
        assert_eq!(value["narrative"].as_array().unwrap().len(), 1);
    }
}
