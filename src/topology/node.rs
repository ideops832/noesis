use crate::hdc::real::RealHV;
use crate::noesis::multiscale::{MultiScaleConfig, MultiScaleField};
use crate::server::state::{NarrativePoint, Event};

pub struct Node {
    pub id: usize,
    pub field: MultiScaleField,
    pub narrative: Vec<NarrativePoint>,
    pub total_steps: u64,
    pub turn_count: u32,
    pub last_event: Option<Event>,
    prev_state: Option<RealHV>,
}

impl Node {
    pub fn new(id: usize, config: MultiScaleConfig, seed: u64) -> Self {
        Node {
            id,
            field: MultiScaleField::new(config, seed),
            narrative: Vec::new(),
            total_steps: 0,
            turn_count: 0,
            last_event: None,
            prev_state: None,
        }
    }

    pub fn step(&mut self, input: &RealHV) {
        self.field.step(input);
        self.total_steps += 1;
    }

    pub fn idle_step(&mut self) {
        self.field.idle_step();
        self.total_steps += 1;
    }

    pub fn reset(&mut self) {
        self.field.reset();
        self.narrative.clear();
        self.total_steps = 0;
        self.turn_count = 0;
        self.last_event = None;
        self.prev_state = None;
    }

    /// Velocity: 1 - cosine_similarity(prev_state, current combined_state).
    /// Returns 0.0 if no previous state is available.
    pub fn velocity(&self) -> f32 {
        if let Some(prev) = &self.prev_state {
            let curr = self.field.combined_state();
            if prev.norm() > 1e-8 && curr.norm() > 1e-8 {
                1.0 - RealHV::cosine_similarity(prev, curr)
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    /// Snapshot the current combined state as the previous state (call after each tick).
    pub fn update_prev_state(&mut self) {
        self.prev_state = Some(self.field.combined_state().clone());
    }
}
