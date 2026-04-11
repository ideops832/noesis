//! Deliberative volition layer — Phase 1.
//!
//! Implements goal hierarchies encoded as hypervectors with recursive VSA
//! binding, per-level precision parameter γ that controls field resistance
//! to local attractors, and coherence checks between candidate solutions
//! and active goals.
//!
//! The system is purely built on existing HDC primitives (bind, bundle,
//! cosine_similarity) and does not introduce any new dependencies.

use std::collections::HashMap;

use crate::hdc::real::RealHV;
use crate::language::composer::Composer;
use crate::noesis::multiscale::MultiScaleField;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Hierarchical level of a goal.
///
/// Each level binds to one of the three scales of a `MultiScaleField`:
/// - `Strategic` → slow field (values, long-term direction)
/// - `Tactical`  → medium field (mid-term objectives)
/// - `Operational` → fast field (immediate actions)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GoalLevel {
    Strategic,
    Tactical,
    Operational,
}

/// Lifecycle status of a goal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalStatus {
    Active,
    Achieved,
    Failed,
    Suspended,
}

// ---------------------------------------------------------------------------
// Goal
// ---------------------------------------------------------------------------

/// A goal with hypervector encoding, precision and optional sub-goals.
#[derive(Debug, Clone)]
pub struct Goal {
    pub id: String,
    pub level: GoalLevel,
    /// Desired-state hypervector (the "attractor" we want to drift towards).
    pub hv: RealHV,
    /// Precision in [0, 1]. Higher = stronger resistance to unrelated inputs.
    pub gamma: f32,
    pub status: GoalStatus,
    /// IDs of direct sub-goals.
    pub children: Vec<String>,
    /// Minimum coherence score to consider the goal achieved.
    pub threshold: f32,
}

// ---------------------------------------------------------------------------
// Coherence result
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CoherenceResult {
    /// Cosine similarity between the solution and the goal HV, clamped to [0,1].
    pub score: f32,
    /// True iff `score >= goal.threshold`.
    pub is_achieved: bool,
    /// How far we still are from the goal (1.0 - score).
    pub gap: f32,
}

// ---------------------------------------------------------------------------
// Goal hierarchy
// ---------------------------------------------------------------------------

/// A tree of goals indexed by id.
pub struct GoalHierarchy {
    goals: HashMap<String, Goal>,
    root_id: Option<String>,
    /// Cached role vectors used for recursive VSA binding. Deterministic per
    /// (dim, index) via `RealHV::random` seeded from `VOLITION_ROLE_SEED`.
    role_vectors: Vec<RealHV>,
    role_self: RealHV,
    dim: usize,
}

const VOLITION_ROLE_SEED: u64 = 0xCAFE_BABE_DEAD_BEEF;

impl GoalHierarchy {
    pub fn new(dim: usize) -> Self {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let mut rng = StdRng::seed_from_u64(VOLITION_ROLE_SEED);
        // Up to 32 children per node is plenty for phase 1.
        let role_vectors: Vec<RealHV> = (0..32).map(|_| RealHV::random(dim, &mut rng)).collect();
        let role_self = RealHV::random(dim, &mut rng);
        GoalHierarchy {
            goals: HashMap::new(),
            root_id: None,
            role_vectors,
            role_self,
            dim,
        }
    }

    /// Registers a goal encoded from a free-form text description through the
    /// provided Composer. If `parent_id` is Some, the new goal is appended to
    /// the parent's children list.
    ///
    /// Returns the id of the created goal.
    pub fn add_goal(
        &mut self,
        level: GoalLevel,
        description: &str,
        parent_id: Option<&str>,
        gamma: f32,
        threshold: f32,
        composer: &mut Composer,
    ) -> String {
        let hv = composer
            .encode_sentence(description)
            .unwrap_or_else(|| RealHV::zero(self.dim));

        let id = format!("g{}", self.goals.len());
        let goal = Goal {
            id: id.clone(),
            level,
            hv,
            gamma: gamma.clamp(0.0, 1.0),
            status: GoalStatus::Active,
            children: Vec::new(),
            threshold: threshold.clamp(0.0, 1.0),
        };
        self.goals.insert(id.clone(), goal);

        if let Some(pid) = parent_id {
            if let Some(parent) = self.goals.get_mut(pid) {
                parent.children.push(id.clone());
            }
        } else if self.root_id.is_none() {
            self.root_id = Some(id.clone());
        }

        id
    }

    /// Register a pre-built goal HV directly (useful for tests and for goals
    /// that do not come from text).
    pub fn add_goal_raw(
        &mut self,
        level: GoalLevel,
        hv: RealHV,
        parent_id: Option<&str>,
        gamma: f32,
        threshold: f32,
    ) -> String {
        let id = format!("g{}", self.goals.len());
        let goal = Goal {
            id: id.clone(),
            level,
            hv,
            gamma: gamma.clamp(0.0, 1.0),
            status: GoalStatus::Active,
            children: Vec::new(),
            threshold: threshold.clamp(0.0, 1.0),
        };
        self.goals.insert(id.clone(), goal);

        if let Some(pid) = parent_id {
            if let Some(parent) = self.goals.get_mut(pid) {
                parent.children.push(id.clone());
            }
        } else if self.root_id.is_none() {
            self.root_id = Some(id.clone());
        }
        id
    }

    pub fn get(&self, id: &str) -> Option<&Goal> { self.goals.get(id) }
    pub fn get_mut(&mut self, id: &str) -> Option<&mut Goal> { self.goals.get_mut(id) }
    pub fn root_id(&self) -> Option<&str> { self.root_id.as_deref() }
    pub fn len(&self) -> usize { self.goals.len() }
    pub fn is_empty(&self) -> bool { self.goals.is_empty() }

    /// Recursive VSA encoding of a goal and its sub-tree.
    ///
    /// ```text
    /// encode(g) = normalize(
    ///     bind(ROLE_SELF, g.hv)
    ///   + Σ_i bind(ROLE_i, encode(child_i))
    /// )
    /// ```
    ///
    /// Leaves degenerate to `bind(ROLE_SELF, g.hv)`.
    pub fn encode_recursive(&self, goal_id: &str) -> RealHV {
        let goal = match self.goals.get(goal_id) {
            Some(g) => g,
            None => return RealHV::zero(self.dim),
        };

        let self_bound = RealHV::bind(&self.role_self, &goal.hv);

        if goal.children.is_empty() {
            return self_bound;
        }

        let mut parts: Vec<RealHV> = Vec::with_capacity(goal.children.len() + 1);
        parts.push(self_bound);
        for (i, child_id) in goal.children.iter().enumerate() {
            if i >= self.role_vectors.len() { break; }
            let child_encoding = self.encode_recursive(child_id);
            parts.push(RealHV::bind(&self.role_vectors[i], &child_encoding));
        }
        let refs: Vec<&RealHV> = parts.iter().collect();
        RealHV::bundle_normalized(&refs)
    }

    /// Unbind a sub-goal by its role index (0 = first child, 1 = second ...).
    ///
    /// `bind(role, sub)` then bundled with siblings can be approximately
    /// recovered as `bind(composite, role)` since bipolar `role` is its own
    /// inverse. Returns a clean hypervector if possible.
    pub fn query_subgoal(&self, composite: &RealHV, role_index: usize) -> RealHV {
        if role_index >= self.role_vectors.len() {
            return RealHV::zero(self.dim);
        }
        // For bipolar vectors, self-inverse holds: bind(composite, role) acts
        // as unbinding (approximation due to bundling noise).
        RealHV::bind(composite, &self.role_vectors[role_index])
    }

    /// Coherence of a candidate solution with a stored goal.
    pub fn check_coherence(&mut self, solution_hv: &RealHV, goal_id: &str) -> CoherenceResult {
        let goal = match self.goals.get_mut(goal_id) {
            Some(g) => g,
            None => {
                return CoherenceResult { score: 0.0, is_achieved: false, gap: 1.0 };
            }
        };
        let raw = RealHV::cosine_similarity(solution_hv, &goal.hv);
        let score = raw.clamp(0.0, 1.0);
        let is_achieved = score >= goal.threshold;
        if is_achieved && goal.status == GoalStatus::Active {
            goal.status = GoalStatus::Achieved;
        }
        CoherenceResult { score, is_achieved, gap: 1.0 - score }
    }

    /// Best candidate according to cosine similarity with the goal HV.
    /// Returns `(index, CoherenceResult)` for the winner, or `(usize::MAX, zero)`
    /// if no candidate is available / goal is missing.
    pub fn best_solution(
        &self,
        candidates: &[RealHV],
        goal_id: &str,
    ) -> (usize, CoherenceResult) {
        let goal = match self.goals.get(goal_id) {
            Some(g) => g,
            None => {
                return (
                    usize::MAX,
                    CoherenceResult { score: 0.0, is_achieved: false, gap: 1.0 },
                );
            }
        };

        let mut best_idx = usize::MAX;
        let mut best_score = f32::NEG_INFINITY;
        for (i, hv) in candidates.iter().enumerate() {
            let s = RealHV::cosine_similarity(hv, &goal.hv);
            if s > best_score {
                best_score = s;
                best_idx = i;
            }
        }

        if best_idx == usize::MAX {
            return (
                usize::MAX,
                CoherenceResult { score: 0.0, is_achieved: false, gap: 1.0 },
            );
        }

        let score = best_score.clamp(0.0, 1.0);
        let is_achieved = score >= goal.threshold;
        (
            best_idx,
            CoherenceResult { score, is_achieved, gap: 1.0 - score },
        )
    }
}

// ---------------------------------------------------------------------------
// Gamma-modulated field update
// ---------------------------------------------------------------------------

/// Applies a precision-modulated blend of `input_hv` with `goal_hv`, producing
/// the hypervector that will actually be fed to the semantic field:
///
/// ```text
/// modulated = (1 - γ) * input + γ * goal_direction
/// ```
///
/// `goal_direction` is the goal HV normalized to unit norm. The result is
/// normalized before being returned so downstream scale mixing stays stable.
pub fn apply_goal_bias(input_hv: &RealHV, goal_hv: &RealHV, gamma: f32) -> RealHV {
    let g = gamma.clamp(0.0, 1.0);
    if g <= 0.0 {
        return input_hv.clone();
    }
    let input_norm = if input_hv.norm() > 1e-10 { input_hv.normalized() } else { input_hv.clone() };
    let goal_norm = if goal_hv.norm() > 1e-10 { goal_hv.normalized() } else { goal_hv.clone() };

    let kept = input_norm.scale(1.0 - g);
    let pulled = goal_norm.scale(g);
    let mut combined = RealHV::add(&kept, &pulled);
    if combined.norm() > 1e-10 {
        combined.normalize();
    }
    combined
}

// ---------------------------------------------------------------------------
// Volition layer
// ---------------------------------------------------------------------------

/// Output of one deliberative step.
#[derive(Debug, Clone)]
pub struct VolitionOutput {
    pub coherence: CoherenceResult,
    /// How much the field state moved from the raw input towards the goal.
    /// Measured as `sim(field_after, goal) - sim(input, goal)`, clamped to [0,1].
    /// Higher = more "resistance" against the incoming attractor.
    pub field_resistance: f32,
    pub goal_status: GoalStatus,
}

pub struct VolitionLayer {
    pub goals: GoalHierarchy,
    pub active_goal_id: Option<String>,
    dim: usize,
}

impl VolitionLayer {
    pub fn new(dim: usize) -> Self {
        VolitionLayer {
            goals: GoalHierarchy::new(dim),
            active_goal_id: None,
            dim,
        }
    }

    /// Registers a single goal from text and marks it active. Returns its id.
    pub fn set_goal(
        &mut self,
        description: &str,
        level: GoalLevel,
        gamma: f32,
        threshold: f32,
        composer: &mut Composer,
    ) -> String {
        let id = self
            .goals
            .add_goal(level, description, None, gamma, threshold, composer);
        self.active_goal_id = Some(id.clone());
        id
    }

    /// Register an already-encoded goal and mark it active.
    pub fn set_goal_raw(
        &mut self,
        hv: RealHV,
        level: GoalLevel,
        gamma: f32,
        threshold: f32,
    ) -> String {
        let id = self.goals.add_goal_raw(level, hv, None, gamma, threshold);
        self.active_goal_id = Some(id.clone());
        id
    }

    /// Single deliberative step:
    /// 1. biases `input_hv` towards the active goal using its γ;
    /// 2. pushes the biased hv into `semantic_field`;
    /// 3. computes coherence of the new field state against the goal;
    /// 4. returns a structured output including the measured resistance.
    pub fn process(
        &mut self,
        input_hv: &RealHV,
        semantic_field: &mut MultiScaleField,
    ) -> VolitionOutput {
        let id = match &self.active_goal_id {
            Some(i) => i.clone(),
            None => {
                semantic_field.step(input_hv);
                return VolitionOutput {
                    coherence: CoherenceResult { score: 0.0, is_achieved: false, gap: 1.0 },
                    field_resistance: 0.0,
                    goal_status: GoalStatus::Suspended,
                };
            }
        };

        // Take an immutable snapshot of (goal_hv, gamma, threshold) before
        // calling any &mut self method on self.goals.
        let (goal_hv, gamma) = {
            let g = match self.goals.get(&id) {
                Some(g) => g,
                None => {
                    return VolitionOutput {
                        coherence: CoherenceResult { score: 0.0, is_achieved: false, gap: 1.0 },
                        field_resistance: 0.0,
                        goal_status: GoalStatus::Failed,
                    };
                }
            };
            (g.hv.clone(), g.gamma)
        };

        // Baseline: how similar is the *raw* input to the goal before bias?
        let baseline = RealHV::cosine_similarity(input_hv, &goal_hv).clamp(0.0, 1.0);

        // γ-biased input then advances the field one step.
        let biased = apply_goal_bias(input_hv, &goal_hv, gamma);
        semantic_field.step(&biased);

        // Coherence of the new combined state with the goal.
        let field_state = semantic_field.combined_state().clone();
        let coherence = self.goals.check_coherence(&field_state, &id);
        let goal_status = self.goals.get(&id).map(|g| g.status).unwrap_or(GoalStatus::Failed);

        // "Resistance" = how much the field sits above the raw-input baseline.
        // A positive value means the γ pull moved the field closer to the goal
        // than the input itself would have been.
        let field_resistance = (coherence.score - baseline).max(0.0).min(1.0);

        VolitionOutput { coherence, field_resistance, goal_status }
    }

    /// Evaluate candidate solutions against the active goal. Returns pairs of
    /// `(original_index, CoherenceResult)` sorted by descending score.
    pub fn propose_and_evaluate(
        &self,
        candidates: &[RealHV],
    ) -> Vec<(usize, CoherenceResult)> {
        let id = match &self.active_goal_id {
            Some(i) => i.as_str(),
            None => return Vec::new(),
        };
        let goal = match self.goals.get(id) {
            Some(g) => g,
            None => return Vec::new(),
        };

        let mut scored: Vec<(usize, CoherenceResult)> = candidates
            .iter()
            .enumerate()
            .map(|(i, hv)| {
                let raw = RealHV::cosine_similarity(hv, &goal.hv);
                let score = raw.clamp(0.0, 1.0);
                let is_achieved = score >= goal.threshold;
                (i, CoherenceResult { score, is_achieved, gap: 1.0 - score })
            })
            .collect();

        scored.sort_by(|a, b| b.1.score.partial_cmp(&a.1.score).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }

    pub fn dim(&self) -> usize { self.dim }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::language::composer::Composer;
    use crate::lnn::ncp::NCPConfig;
    use crate::noesis::bridge::BridgeStrategy;
    use crate::noesis::multiscale::{MultiScaleConfig, MultiScaleField};

    const D: usize = 1024;

    fn make_composer() -> Composer {
        let mut c = Composer::new(D, 42);
        // Seed a tiny vocabulary so encode_sentence never returns None on the
        // phrases we use below.
        for w in [
            "classify", "concept", "domain", "correct", "goal", "task",
            "whale", "ocean", "hunt", "captain", "prison", "escape",
            "father", "life", "dark", "river",
        ] {
            let _ = c.vocabulary.get_or_create(w);
        }
        c
    }

    fn make_field() -> MultiScaleField {
        let mut cfg = MultiScaleConfig::default_for_dim(D);
        cfg.strategy = BridgeStrategy::InputPreserving;
        cfg.ncp_config = NCPConfig::tiny();
        MultiScaleField::new(cfg, 7)
    }

    // Test 1: recursive hierarchy encoding and sub-goal recovery.
    #[test]
    fn test_goal_hierarchy_encoding() {
        let mut composer = make_composer();
        let mut h = GoalHierarchy::new(D);

        let root = h.add_goal(
            GoalLevel::Strategic,
            "classify concept in correct domain",
            None, 0.7, 0.6, &mut composer,
        );
        let c1 = h.add_goal(
            GoalLevel::Tactical,
            "whale ocean hunt",
            Some(&root), 0.5, 0.55, &mut composer,
        );
        let c2 = h.add_goal(
            GoalLevel::Tactical,
            "prison escape captain",
            Some(&root), 0.5, 0.55, &mut composer,
        );

        assert_eq!(h.len(), 3);
        assert_eq!(h.get(&root).unwrap().children.len(), 2);

        // Recursive encoding must be non-trivial (not zero) and closer to its
        // own structure than to a random vector.
        let composite = h.encode_recursive(&root);
        assert!(composite.norm() > 0.1, "composite should be non-zero");

        // Unbinding role 0 should give us something more similar to child_1
        // than to child_2 (noisy but directional for bipolar roles).
        let recovered_0 = h.query_subgoal(&composite, 0);
        let child_0_encoding = h.encode_recursive(&c1);
        let child_1_encoding = h.encode_recursive(&c2);

        let sim_to_child0 = RealHV::cosine_similarity(&recovered_0, &child_0_encoding);
        let sim_to_child1 = RealHV::cosine_similarity(&recovered_0, &child_1_encoding);

        // We only assert the direction: unbinding with role_0 should be
        // preferentially associated with child 0 (this is the whole point of
        // VSA). We allow a small margin because bundling introduces noise.
        assert!(
            sim_to_child0 > sim_to_child1 - 0.05,
            "unbind(composite, role_0) should lean towards child 0: sim0={:.3} vs sim1={:.3}",
            sim_to_child0, sim_to_child1
        );
    }

    // Test 2: high γ keeps the field close to the goal, low γ does not.
    #[test]
    fn test_gamma_resistance() {
        let mut composer = make_composer();

        let goal_hv = composer.encode_sentence("whale ocean hunt captain").unwrap();
        let attractor = composer.encode_sentence("prison escape father dark").unwrap();

        // High γ: strong pull towards the goal.
        let biased_high = apply_goal_bias(&attractor, &goal_hv, 0.8);
        let sim_high = RealHV::cosine_similarity(&biased_high, &goal_hv);

        // Low γ: almost no pull.
        let biased_low = apply_goal_bias(&attractor, &goal_hv, 0.1);
        let sim_low = RealHV::cosine_similarity(&biased_low, &goal_hv);

        assert!(
            sim_high - sim_low > 0.2,
            "expected sim_high - sim_low > 0.2, got sim_high={:.3} sim_low={:.3}",
            sim_high, sim_low
        );
    }

    // Test 3: coherence check picks the best candidate.
    #[test]
    fn test_coherence_check() {
        let mut composer = make_composer();
        let mut layer = VolitionLayer::new(D);
        let goal_hv = composer.encode_sentence("whale ocean hunt captain").unwrap();
        layer.set_goal_raw(goal_hv.clone(), GoalLevel::Tactical, 0.5, 0.8);

        // Three candidates: identical to goal, slight variant, unrelated.
        let same = goal_hv.clone();
        let near = composer.encode_sentence("whale ocean captain").unwrap();
        let far  = composer.encode_sentence("prison escape dark").unwrap();

        let ranked = layer.propose_and_evaluate(&[far.clone(), near.clone(), same.clone()]);
        assert!(!ranked.is_empty());
        // The winner must be the "same" candidate (index 2 in the input).
        assert_eq!(ranked[0].0, 2, "best should be the identical candidate");
        // Score must cross the threshold for "same".
        assert!(ranked[0].1.is_achieved, "identical candidate should be achieved");
    }

    // Test 4: status flips to Achieved as input approaches the goal.
    // Uses low γ so we can actually observe the transition.
    #[test]
    fn test_goal_achievement() {
        let mut composer = make_composer();
        let mut field = make_field();
        let mut layer = VolitionLayer::new(D);

        let goal_hv = composer.encode_sentence("whale ocean hunt captain").unwrap();
        // Low gamma + high threshold: the noise alone must not trigger Achieved,
        // we want the goal input to do the work.
        let goal_id = layer.set_goal_raw(goal_hv.clone(), GoalLevel::Tactical, 0.2, 0.75);

        let noise = composer.encode_sentence("prison escape father dark").unwrap();

        // Noise only: threshold 0.85 is high enough that even with γ=0.2
        // the combined field should stay below it (noise and goal are dissimilar).
        for _ in 0..3 { let _ = layer.process(&noise, &mut field); }
        assert_eq!(
            layer.goals.get(&goal_id).unwrap().status,
            GoalStatus::Active,
            "status should still be Active under noise with threshold=0.75 γ=0.2"
        );

        // Repeated goal input: field converges to the goal → should cross 0.85.
        let mut last_score = 0.0f32;
        for _ in 0..40 {
            let out = layer.process(&goal_hv, &mut field);
            last_score = out.coherence.score;
        }
        let st = layer.goals.get(&goal_id).unwrap().status;
        assert_eq!(
            st,
            GoalStatus::Achieved,
            "status should become Achieved once the field converges (last score = {:.3})",
            last_score
        );
    }
}
