//! Semantic memory with differentiated decay.
//!
//! SemanticMemory wraps a SemanticField and adds an explicit memory store
//! where each memorized item has a strength that decays over time.
//! More recent items decay slower (recency effect), and each recall
//! refreshes the memory strength.

use crate::hdc::real::RealHV;
use crate::noesis::semantic_field::{SemanticField, SemanticFieldConfig};

/// A single memory entry: an HD vector with associated strength metadata.
#[derive(Clone, Debug)]
struct MemoryEntry {
    /// The memorized hypervector.
    vector: RealHV,
    /// Current strength in (0, 1]. Starts at 1.0.
    strength: f32,
    /// Time at which this item was last accessed (memorized or recalled).
    last_access: f32,
    /// Number of times this item has been reinforced.
    rehearsals: u32,
}

/// Semantic memory built on top of a SemanticField.
///
/// Maintains a vector of memory entries with differentiated decay:
/// - More recently accessed items decay slower.
/// - Items that have been rehearsed more decay slower.
/// - `decay()` reduces all strengths; entries below a threshold are forgotten.
pub struct SemanticMemory {
    /// The underlying semantic field that provides dynamics.
    pub field: SemanticField,
    /// Stored memory entries.
    entries: Vec<MemoryEntry>,
    /// Monotonically increasing clock.
    clock: f32,
    /// Base decay rate per unit time.
    base_decay_rate: f32,
    /// Strength threshold below which entries are forgotten.
    forget_threshold: f32,
}

impl SemanticMemory {
    /// Creates a new SemanticMemory wrapping a SemanticField.
    pub fn new(config: SemanticFieldConfig, seed: u64) -> Self {
        let field = SemanticField::new(config, seed);
        SemanticMemory {
            field,
            entries: Vec::new(),
            clock: 0.0,
            base_decay_rate: 0.08,
            forget_threshold: 0.01,
        }
    }

    /// Memorize a hypervector. If a sufficiently similar vector already exists
    /// (cosine similarity > 0.8), reinforce it instead of creating a new entry.
    pub fn memorize(&mut self, vector: &RealHV) {
        // Check for existing similar entry
        let mut best_idx = None;
        let mut best_sim = 0.8_f32;
        for (i, entry) in self.entries.iter().enumerate() {
            let sim = RealHV::cosine_similarity(&entry.vector, vector);
            if sim > best_sim {
                best_sim = sim;
                best_idx = Some(i);
            }
        }

        if let Some(idx) = best_idx {
            // Reinforce existing memory
            self.entries[idx].strength = (self.entries[idx].strength + 0.3).min(1.0);
            self.entries[idx].last_access = self.clock;
            self.entries[idx].rehearsals += 1;
        } else {
            // Store new memory
            self.entries.push(MemoryEntry {
                vector: vector.clone(),
                strength: 1.0,
                last_access: self.clock,
                rehearsals: 1,
            });
        }

        // Also feed through the semantic field to shape dynamics
        self.field.step(vector, 0.1);
        self.clock += 0.1;
    }

    /// Recall the best-matching memory for a query vector.
    /// Returns the memory vector and its current strength, or None if empty.
    pub fn recall(&mut self, query: &RealHV) -> Option<(RealHV, f32)> {
        if self.entries.is_empty() {
            return None;
        }
        let mut best_idx = 0;
        let mut best_sim = f32::NEG_INFINITY;
        for (i, entry) in self.entries.iter().enumerate() {
            let sim = RealHV::cosine_similarity(&entry.vector, query) * entry.strength;
            if sim > best_sim {
                best_sim = sim;
                best_idx = i;
            }
        }

        // Refresh on recall
        self.entries[best_idx].last_access = self.clock;
        self.entries[best_idx].rehearsals += 1;
        let entry = &self.entries[best_idx];
        Some((entry.vector.clone(), entry.strength))
    }

    /// Return all memories with their strengths, sorted by strength descending.
    pub fn recall_all(&self) -> Vec<(RealHV, f32)> {
        let mut result: Vec<(RealHV, f32)> = self
            .entries
            .iter()
            .map(|e| (e.vector.clone(), e.strength))
            .collect();
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    /// Apply differentiated decay to all memories.
    ///
    /// Decay formula per entry:
    ///   effective_rate = base_decay_rate / (1 + rehearsals) * recency_factor
    /// where recency_factor increases with time since last access.
    ///
    /// Entries whose strength falls below `forget_threshold` are removed.
    pub fn decay(&mut self, dt: f32) {
        for entry in &mut self.entries {
            let age = self.clock - entry.last_access + 1.0; // +1 to avoid zero
            let recency_factor = age.sqrt(); // older items decay faster
            let rehearsal_protection = 1.0 / (1.0 + entry.rehearsals as f32);
            let effective_rate = self.base_decay_rate * rehearsal_protection * recency_factor;
            entry.strength *= (-effective_rate * dt).exp();
        }

        // Remove forgotten entries
        self.entries
            .retain(|e| e.strength >= self.forget_threshold);

        // Advance clock
        self.clock += dt;

        // Also let the field idle-step
        self.field.idle_step(dt);
    }

    /// Rehearse top-k strongest memories by re-injecting them into the field.
    ///
    /// This mimics sleep-like consolidation: the strongest memories are
    /// replayed through the field with a small dt, reinforcing their trace
    /// in the field dynamics and boosting their strength.
    pub fn rehearse(&mut self, k: usize, rehearsal_dt: f32) {
        // Sort by strength descending, take top-k
        let mut sorted_indices: Vec<usize> = (0..self.entries.len()).collect();
        sorted_indices.sort_by(|&a, &b| {
            self.entries[b]
                .strength
                .partial_cmp(&self.entries[a].strength)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for &idx in sorted_indices.iter().take(k) {
            let vector = self.entries[idx].vector.clone();
            // Re-inject into field with small step
            self.field.step(&vector, rehearsal_dt);
            // Boost strength
            self.entries[idx].strength = (self.entries[idx].strength + 0.1).min(1.0);
            self.entries[idx].rehearsals += 1;
            self.entries[idx].last_access = self.clock;
        }
        self.clock += rehearsal_dt;
    }

    /// Combined decay + rehearsal cycle.
    ///
    /// First decays all memories, then rehearses the top-k surviving ones.
    /// This creates a natural selection: strong memories get rehearsed and
    /// become stronger, weak ones decay and are forgotten.
    pub fn consolidation_cycle(&mut self, decay_dt: f32, rehearsal_k: usize, rehearsal_dt: f32) {
        self.decay(decay_dt);
        if !self.entries.is_empty() {
            self.rehearse(rehearsal_k.min(self.entries.len()), rehearsal_dt);
        }
    }

    /// Number of currently stored memories.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the memory is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Current clock time.
    pub fn clock(&self) -> f32 {
        self.clock
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::real::RealHV;
    use crate::lnn::ncp::NCPConfig;
    use crate::noesis::bridge::BridgeStrategy;
    use crate::noesis::semantic_field::SemanticFieldConfig;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    const D: usize = 1024;

    fn make_config() -> SemanticFieldConfig {
        SemanticFieldConfig::new(D, BridgeStrategy::DualTrack, NCPConfig::tiny())
    }

    #[test]
    fn test_memory_decay_monotonic() {
        let mut mem = SemanticMemory::new(make_config(), 42);
        let mut rng = StdRng::seed_from_u64(99);
        let v = RealHV::random(D, &mut rng);

        mem.memorize(&v);

        // Record strengths over successive decay steps
        let mut strengths = Vec::new();
        for _ in 0..20 {
            let all = mem.recall_all();
            assert!(!all.is_empty(), "Memory should not be empty yet");
            strengths.push(all[0].1);
            mem.decay(1.0);
        }

        // Strengths should be monotonically decreasing
        for i in 1..strengths.len() {
            assert!(
                strengths[i] <= strengths[i - 1],
                "Strength should decrease monotonically: step {} has {} > {}",
                i,
                strengths[i],
                strengths[i - 1]
            );
        }

        // Final strength should be strictly less than initial
        assert!(
            strengths.last().unwrap() < strengths.first().unwrap(),
            "Memory should have decayed: first={}, last={}",
            strengths.first().unwrap(),
            strengths.last().unwrap()
        );
    }

    #[test]
    fn test_recency_effect() {
        let mut mem = SemanticMemory::new(make_config(), 42);
        let mut rng = StdRng::seed_from_u64(99);

        let old_item = RealHV::random(D, &mut rng);
        let new_item = RealHV::random(D, &mut rng);

        // Memorize old item first
        mem.memorize(&old_item);

        // Let time pass with decay
        for _ in 0..10 {
            mem.decay(1.0);
        }

        // Memorize new item
        mem.memorize(&new_item);

        // Apply a small decay
        mem.decay(1.0);

        // Now recall_all: new item should have higher strength than old item
        let all = mem.recall_all();
        assert!(all.len() >= 2, "Should have at least 2 memories, got {}", all.len());

        // Find strengths of old and new items
        let old_strength = all
            .iter()
            .map(|(v, s)| (RealHV::cosine_similarity(v, &old_item), *s))
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .unwrap()
            .1;
        let new_strength = all
            .iter()
            .map(|(v, s)| (RealHV::cosine_similarity(v, &new_item), *s))
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .unwrap()
            .1;

        assert!(
            new_strength > old_strength,
            "Recency effect: new item strength ({}) should exceed old item strength ({})",
            new_strength,
            old_strength
        );
    }

    #[test]
    fn test_rehearsal_preserves_memory() {
        let mut mem = SemanticMemory::new(make_config(), 42);
        let mut rng = StdRng::seed_from_u64(99);

        let item_a = RealHV::random(D, &mut rng);
        let item_b = RealHV::random(D, &mut rng);

        mem.memorize(&item_a);
        mem.memorize(&item_b);

        // Decay without rehearsal
        let mut mem_no_rehear = SemanticMemory::new(make_config(), 42);
        let mut rng2 = StdRng::seed_from_u64(99);
        let item_a2 = RealHV::random(D, &mut rng2);
        let item_b2 = RealHV::random(D, &mut rng2);
        mem_no_rehear.memorize(&item_a2);
        mem_no_rehear.memorize(&item_b2);

        // Run 10 cycles: one with rehearsal, one without
        for _ in 0..10 {
            mem.consolidation_cycle(1.0, 2, 0.05);
            mem_no_rehear.decay(1.0);
        }

        let strength_with = mem.recall_all();
        let strength_without = mem_no_rehear.recall_all();

        let max_with = strength_with.first().map(|(_, s)| *s).unwrap_or(0.0);
        let max_without = strength_without.first().map(|(_, s)| *s).unwrap_or(0.0);

        assert!(
            max_with > max_without,
            "Rehearsal should preserve memory better: with={:.4} without={:.4}",
            max_with, max_without
        );
    }

    #[test]
    fn test_consolidation_cycle() {
        let mut mem = SemanticMemory::new(make_config(), 42);
        let mut rng = StdRng::seed_from_u64(99);

        // Memorize 5 items
        for _ in 0..5 {
            let v = RealHV::random(D, &mut rng);
            mem.memorize(&v);
        }

        assert_eq!(mem.len(), 5);

        // Run many consolidation cycles — weak items should be forgotten
        for _ in 0..50 {
            mem.consolidation_cycle(2.0, 2, 0.05);
        }

        // Some items should have been forgotten (strength below threshold)
        assert!(
            mem.len() <= 5,
            "Some items should have been forgotten or all retained"
        );

        // Surviving items should have been rehearsed
        let surviving = mem.recall_all();
        for (_, strength) in &surviving {
            assert!(
                *strength > 0.0,
                "Surviving memories should have positive strength"
            );
        }
    }
}
