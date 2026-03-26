use crate::hdc::real::RealHV;
use crate::noesis::multiscale::MultiScaleConfig;
use super::node::Node;

pub struct Ring {
    pub nodes: Vec<Node>,
    pub coupling_lr: f32,
    pub tick: u64,
}

impl Ring {
    pub fn new(config: MultiScaleConfig, n_nodes: usize, coupling_lr: f32, base_seed: u64) -> Self {
        let nodes = (0..n_nodes)
            .map(|i| Node::new(i, config.clone(), base_seed + i as u64 * 100))
            .collect();
        Ring { nodes, coupling_lr, tick: 0 }
    }

    /// Propagate state around the ring: node i's combined_state bundles into node (i+1)%N's slow field.
    pub fn step_ring(&mut self) {
        let n = self.nodes.len();
        // Collect all states first (can't borrow mutably while reading)
        let states: Vec<RealHV> = self.nodes.iter()
            .map(|node| node.field.combined_state().clone())
            .collect();

        // Bundle each state into the next node's slow field
        for i in 0..n {
            let next = (i + 1) % n;
            let incoming = &states[i];
            if incoming.norm() < 1e-8 { continue; }

            let old_slow = self.nodes[next].field.slow_state().clone();
            let incoming_norm = incoming.normalized();
            if old_slow.norm() < 1e-8 {
                // Empty slow field: seed with a fraction of the incoming signal
                self.nodes[next].field.set_slow_state(incoming_norm.scale(self.coupling_lr * 5.0));
            } else {
                // Gradual bundling: mix incoming into existing slow field
                let kept = old_slow.scale(1.0 - self.coupling_lr);
                let added = incoming_norm.scale(self.coupling_lr);
                let mut new_slow = RealHV::add(&kept, &added);
                new_slow.normalize();
                self.nodes[next].field.set_slow_state(new_slow);
            }
        }

        self.tick += 1;
    }

    pub fn feed_node(&mut self, node_id: usize, input: &RealHV) {
        if node_id < self.nodes.len() {
            self.nodes[node_id].step(input);
            self.nodes[node_id].turn_count += 1;
        }
    }

    pub fn idle_all(&mut self) {
        for node in &mut self.nodes {
            node.idle_step();
        }
    }

    pub fn reset_all(&mut self) {
        for node in &mut self.nodes {
            node.reset();
        }
        self.tick = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::real::RealHV;
    use crate::noesis::multiscale::MultiScaleConfig;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn make_ring(n_nodes: usize) -> Ring {
        let config = MultiScaleConfig::default_for_dim(256);
        Ring::new(config, n_nodes, 0.1, 42)
    }

    #[test]
    fn test_ring_propagation() {
        let mut ring = make_ring(4);
        let mut rng = StdRng::seed_from_u64(99);

        // Create an input signal (simulating "gatto" encoding)
        let input = RealHV::random(256, &mut rng);

        // Feed node 0 several times to build up state
        for _ in 0..10 {
            ring.feed_node(0, &input);
        }

        // Node 0 should have non-zero combined state
        assert!(
            ring.nodes[0].field.combined_state().norm() > 0.1,
            "Node 0 should have built up state after feeding"
        );

        // Perform several ring steps with idle steps so nodes recompute combined_state
        // (ring_step propagates combined_state -> slow field, idle_step recomputes combined_state)
        for _ in 0..50 {
            ring.step_ring();
            ring.idle_all(); // recomputes combined_state from the updated slow fields
        }

        // Node 1's slow field should now show some similarity to the input
        let node1_slow = ring.nodes[1].field.slow_state();
        let sim = RealHV::cosine_similarity(node1_slow, &input.normalized());
        println!("Node 1 slow field similarity to input: {:.4}", sim);

        // The similarity should be non-trivially positive (signal has propagated)
        assert!(
            sim > 0.01,
            "Node 1 slow field should show some similarity to input after ring propagation, got {}",
            sim
        );

        // Node 2 should also have received propagated signal (1 -> 2 path)
        let node2_slow_norm = ring.nodes[2].field.slow_state().norm();
        let node3_slow_norm = ring.nodes[3].field.slow_state().norm();
        println!("Node 2 slow norm: {:.4}, Node 3 slow norm: {:.4}", node2_slow_norm, node3_slow_norm);
        assert!(
            node2_slow_norm > 1e-8,
            "Node 2 should have received propagated signal"
        );
    }

    #[test]
    fn test_ring_reset() {
        let mut ring = make_ring(4);
        let mut rng = StdRng::seed_from_u64(42);

        // Feed some data
        let input = RealHV::random(256, &mut rng);
        for _ in 0..5 {
            ring.feed_node(0, &input);
            ring.feed_node(1, &input);
        }
        ring.step_ring();

        // Verify non-zero states
        assert!(
            ring.nodes[0].field.combined_state().norm() > 0.1,
            "Node 0 should have non-zero state before reset"
        );
        assert!(ring.tick > 0, "Tick should be > 0 before reset");

        // Reset
        ring.reset_all();

        // Verify everything is zeroed
        for (i, node) in ring.nodes.iter().enumerate() {
            assert!(
                node.field.combined_state().norm() < 1e-8,
                "Node {} combined state should be zero after reset",
                i
            );
            assert_eq!(node.total_steps, 0, "Node {} total_steps should be 0 after reset", i);
            assert_eq!(node.turn_count, 0, "Node {} turn_count should be 0 after reset", i);
        }
        assert_eq!(ring.tick, 0, "Ring tick should be 0 after reset");
    }
}
