use crate::hdc::real::RealHV;
#[allow(unused_imports)]
use crate::hdc::hypervector::HyperVector;
use crate::noesis::multiscale::MultiScaleConfig;
use super::node::Node;

pub struct Torus {
    pub nodes: Vec<Node>,
    pub rows: usize,
    pub cols: usize,
    pub coupling_lr: f32,
    pub tick: u64,
    neighbors: Vec<[usize; 4]>, // [north, south, west, east] per node
    pub spatial_coherence: f32,
    pub num_zones: usize,
    pub topographic_quality: f32,
}

impl Torus {
    pub fn new(rows: usize, cols: usize, config: MultiScaleConfig, coupling_lr: f32, base_seed: u64) -> Self {
        let n = rows * cols;
        let nodes: Vec<Node> = (0..n)
            .map(|i| Node::new(i, config.clone(), base_seed + i as u64 * 100))
            .collect();

        // Precompute neighbor indices with toroidal wrap-around
        let mut neighbors = Vec::with_capacity(n);
        for i in 0..n {
            let (x, y) = (i % cols, i / cols); // x=col, y=row
            let north = ((y + rows - 1) % rows) * cols + x;
            let south = ((y + 1) % rows) * cols + x;
            let west = y * cols + (x + cols - 1) % cols;
            let east = y * cols + (x + 1) % cols;
            neighbors.push([north, south, west, east]);
        }

        Torus {
            nodes,
            rows,
            cols,
            coupling_lr,
            tick: 0,
            neighbors,
            spatial_coherence: 0.0,
            num_zones: 0,
            topographic_quality: 0.0,
        }
    }

    /// Index from grid position (x=col, y=row).
    pub fn node_at(&self, x: usize, y: usize) -> usize {
        y * self.cols + x
    }

    /// Grid position from index: returns (x=col, y=row).
    pub fn position_of(&self, i: usize) -> (usize, usize) {
        (i % self.cols, i / self.cols)
    }

    /// Manhattan distance on the torus (minimum with wrap-around).
    pub fn torus_distance(&self, a: usize, b: usize) -> usize {
        let (ax, ay) = self.position_of(a);
        let (bx, by) = self.position_of(b);
        let dx = {
            let d = if ax > bx { ax - bx } else { bx - ax };
            d.min(self.cols - d)
        };
        let dy = {
            let d = if ay > by { ay - by } else { by - ay };
            d.min(self.rows - d)
        };
        dx + dy
    }

    /// Get the 4 neighbor indices for node i: [north, south, west, east].
    pub fn neighbors_of(&self, i: usize) -> [usize; 4] {
        self.neighbors[i]
    }

    /// One step of torus coupling: each node's slow field is mixed with its 4 neighbors' combined states.
    pub fn step_torus(&mut self) {
        let n = self.nodes.len();

        // 1. Snapshot all combined_states
        let states: Vec<RealHV> = self.nodes.iter()
            .map(|node| node.field.combined_state().clone())
            .collect();

        // 2. For each node, bundle its 4 neighbors' states and mix into slow field
        for i in 0..n {
            let [north, south, west, east] = self.neighbors[i];
            let neighbor_refs: Vec<&RealHV> = vec![
                &states[north], &states[south], &states[west], &states[east],
            ];
            let neighbor_bundle = RealHV::bundle_normalized(&neighbor_refs);

            if neighbor_bundle.norm() < 1e-8 {
                continue;
            }

            let old_slow = self.nodes[i].field.slow_state().clone();
            if old_slow.norm() < 1e-8 {
                // Empty slow field: seed with a fraction of the neighbor bundle
                self.nodes[i].field.set_slow_state(neighbor_bundle.scale(self.coupling_lr * 5.0));
            } else {
                // Gradual mixing: blend old slow with neighbor bundle
                let kept = old_slow.scale(1.0 - self.coupling_lr);
                let added = neighbor_bundle.scale(self.coupling_lr);
                let mut new_slow = RealHV::add(&kept, &added);
                new_slow.normalize();
                self.nodes[i].field.set_slow_state(new_slow);
            }
        }

        self.tick += 1;
        self.update_metrics();
    }

    pub fn feed_node(&mut self, node_id: usize, input: &RealHV) {
        if node_id < self.nodes.len() {
            self.nodes[node_id].step(input);
            self.nodes[node_id].turn_count += 1;
        }
    }

    pub fn feed_all(&mut self, input: &RealHV) {
        for node in &mut self.nodes {
            node.step(input);
            node.turn_count += 1;
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
        self.spatial_coherence = 0.0;
        self.num_zones = 0;
        self.topographic_quality = 0.0;
    }

    pub fn set_coupling(&mut self, lr: f32) {
        self.coupling_lr = lr;
    }

    /// Compute spatial_coherence, num_zones, topographic_quality.
    pub fn update_metrics(&mut self) {
        let n = self.nodes.len();
        if n == 0 { return; }

        let states: Vec<&RealHV> = self.nodes.iter()
            .map(|node| node.field.combined_state())
            .collect();

        // --- Spatial coherence: avg neighbor similarity / avg all similarity ---
        let mut neighbor_sim_sum = 0.0f32;
        let mut neighbor_count = 0u32;
        let mut all_sim_sum = 0.0f32;
        let mut all_count = 0u32;

        for i in 0..n {
            if states[i].norm() < 1e-8 { continue; }

            // Neighbor similarities
            for &nb in &self.neighbors[i] {
                if states[nb].norm() < 1e-8 { continue; }
                neighbor_sim_sum += RealHV::cosine_similarity(states[i], states[nb]);
                neighbor_count += 1;
            }

            // All-pairs similarities (only i < j to avoid double counting)
            for j in (i + 1)..n {
                if states[j].norm() < 1e-8 { continue; }
                let sim = RealHV::cosine_similarity(states[i], states[j]);
                all_sim_sum += sim;
                all_count += 1;
            }
        }

        let avg_neighbor = if neighbor_count > 0 { neighbor_sim_sum / neighbor_count as f32 } else { 0.0 };
        let avg_all = if all_count > 0 { all_sim_sum / all_count as f32 } else { 0.0 };
        self.spatial_coherence = if avg_all.abs() > 1e-8 { avg_neighbor / avg_all } else { 0.0 };

        // --- Num zones: flood fill with threshold 0.3 ---
        self.num_zones = self.compute_zones(0.3);

        // --- Topographic quality: Pearson correlation(distance, dissimilarity) ---
        self.topographic_quality = self.compute_topographic_quality(&states);
    }

    fn compute_zones(&self, threshold: f32) -> usize {
        let n = self.nodes.len();
        let states: Vec<&RealHV> = self.nodes.iter()
            .map(|node| node.field.combined_state())
            .collect();

        let mut visited = vec![false; n];
        let mut zones = 0;

        for start in 0..n {
            if visited[start] { continue; }
            if states[start].norm() < 1e-8 { visited[start] = true; continue; }

            // BFS flood fill
            zones += 1;
            let mut queue = vec![start];
            visited[start] = true;

            while let Some(current) = queue.pop() {
                for &nb in &self.neighbors[current] {
                    if visited[nb] { continue; }
                    if states[nb].norm() < 1e-8 { visited[nb] = true; continue; }
                    let sim = RealHV::cosine_similarity(states[current], states[nb]);
                    if sim > threshold {
                        visited[nb] = true;
                        queue.push(nb);
                    }
                }
            }
        }

        zones
    }

    fn compute_topographic_quality(&self, states: &[&RealHV]) -> f32 {
        let n = states.len();
        if n < 2 { return 0.0; }

        let mut distances = Vec::new();
        let mut dissimilarities = Vec::new();

        for i in 0..n {
            if states[i].norm() < 1e-8 { continue; }
            for j in (i + 1)..n {
                if states[j].norm() < 1e-8 { continue; }
                distances.push(self.torus_distance(i, j) as f32);
                dissimilarities.push(1.0 - RealHV::cosine_similarity(states[i], states[j]));
            }
        }

        if distances.len() < 2 { return 0.0; }

        // Pearson correlation
        let n_pairs = distances.len() as f32;
        let mean_d = distances.iter().sum::<f32>() / n_pairs;
        let mean_s = dissimilarities.iter().sum::<f32>() / n_pairs;

        let mut cov = 0.0f32;
        let mut var_d = 0.0f32;
        let mut var_s = 0.0f32;

        for k in 0..distances.len() {
            let dd = distances[k] - mean_d;
            let ds = dissimilarities[k] - mean_s;
            cov += dd * ds;
            var_d += dd * dd;
            var_s += ds * ds;
        }

        let denom = (var_d * var_s).sqrt();
        if denom < 1e-12 { 0.0 } else { cov / denom }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::real::RealHV;
    use crate::noesis::multiscale::MultiScaleConfig;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn make_torus(rows: usize, cols: usize) -> Torus {
        let config = MultiScaleConfig::default_for_dim(256);
        Torus::new(rows, cols, config, 0.3, 1042)
    }

    #[test]
    fn test_torus_creation() {
        let torus = make_torus(4, 4);
        assert_eq!(torus.nodes.len(), 16);
        assert_eq!(torus.rows, 4);
        assert_eq!(torus.cols, 4);

        // Each node has 4 neighbors
        for i in 0..16 {
            let nb = torus.neighbors_of(i);
            assert_eq!(nb.len(), 4);
            // All neighbors should be valid indices
            for &n in &nb {
                assert!(n < 16, "Neighbor {} of node {} out of range", n, i);
            }
        }

        // Check wrap-around for corner node (0,0) = index 0
        // x=0, y=0: north=(0,3)=12, south=(0,1)=4, west=(3,0)=3, east=(1,0)=1
        let nb0 = torus.neighbors_of(0);
        assert_eq!(nb0[0], 12, "north of (0,0) should be (0,3)=12");
        assert_eq!(nb0[1], 4, "south of (0,0) should be (0,1)=4");
        assert_eq!(nb0[2], 3, "west of (0,0) should be (3,0)=3");
        assert_eq!(nb0[3], 1, "east of (0,0) should be (1,0)=1");
    }

    #[test]
    fn test_torus_distance() {
        let torus = make_torus(4, 4);

        // (0,0) to (2,2): dx=2, dy=2 → 4
        let a = torus.node_at(0, 0);
        let b = torus.node_at(2, 2);
        assert_eq!(torus.torus_distance(a, b), 4);

        // (0,0) to (3,3): dx=min(3,1)=1, dy=min(3,1)=1 → 2 (wrap shorter)
        let c = torus.node_at(3, 3);
        assert_eq!(torus.torus_distance(a, c), 2);
    }

    #[test]
    fn test_torus_propagation() {
        let mut torus = make_torus(4, 4);
        let mut rng = StdRng::seed_from_u64(99);

        // Create input signal
        let input = RealHV::random(256, &mut rng);

        // Feed node 0 several times to build up state
        for _ in 0..10 {
            torus.feed_node(0, &input);
        }

        assert!(
            torus.nodes[0].field.combined_state().norm() > 0.1,
            "Node 0 should have state after feeding"
        );

        // Find node at maximum distance from 0
        let max_dist_node = (0..16)
            .max_by_key(|&i| torus.torus_distance(0, i))
            .unwrap();

        // Perform torus steps with idle to recompute combined_state
        for _ in 0..10 {
            torus.step_torus();
            torus.idle_all();
        }

        // The farthest node should have received some signal
        let far_slow_norm = torus.nodes[max_dist_node].field.slow_state().norm();
        println!(
            "Max distance node {} (dist={}) slow norm: {:.6}",
            max_dist_node,
            torus.torus_distance(0, max_dist_node),
            far_slow_norm
        );
        assert!(
            far_slow_norm > 1e-8,
            "Node at max distance should have some signal after propagation, got {}",
            far_slow_norm
        );
    }

    #[test]
    fn test_torus_two_zones() {
        let mut torus = make_torus(4, 4);
        let mut rng = StdRng::seed_from_u64(42);

        // Create two distinct signals
        let signal_a = RealHV::random(256, &mut rng);
        let signal_b = RealHV::random(256, &mut rng);

        // Feed node 0 with signal_a ("gatto"-like)
        for _ in 0..10 {
            torus.feed_node(0, &signal_a);
        }

        // Feed node 10 with signal_b ("pasta"-like)
        for _ in 0..10 {
            torus.feed_node(10, &signal_b);
        }

        // Step a few times
        for _ in 0..5 {
            torus.step_torus();
            torus.idle_all();
        }

        torus.update_metrics();

        println!(
            "spatial_coherence={:.4}, num_zones={}, topographic_quality={:.4}",
            torus.spatial_coherence, torus.num_zones, torus.topographic_quality
        );

        // Spatial coherence should be positive (neighbors more similar than random pairs)
        assert!(
            torus.spatial_coherence > 0.0,
            "Spatial coherence should be > 0, got {}",
            torus.spatial_coherence
        );
    }
}
