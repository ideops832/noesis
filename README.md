# NOESIS

**Compositional Semantics as Continuous Dynamics in Hyperdimensional Space**

NOESIS is a research system that unifies Hyperdimensional Computing (HDC) and Liquid Neural Networks (LNN) into a single computational substrate where:

- **Meaning** has compositional algebraic structure (bind, bundle, permutation)
- **Dynamics** are continuous in time, governed by CfC neurons with adaptive time constants
- **Concepts** emerge as attractors in hyperdimensional space
- **Memory** is state inertia with differentiated decay
- **Reasoning** is trajectory through meaning space

## Architecture

```
HDC Space (D=1024..10048)          Bridge            LNN (NCP)
┌─────────────────────┐     ┌──────────────┐    ┌─────────────────┐
│  BinaryHV / RealHV  │────▶│  DualTrack   │───▶│ Sensory→Inter→  │
│  bind ⊗ bundle ⊕    │◀────│  (probes)    │◀───│ Command→Motor   │
│  permute π          │     └──────────────┘    │ CfC neurons     │
└─────────────────────┘                         └─────────────────┘
```

## Quick Start

```bash
# Build
cargo build

# Run all tests (90 tests)
cargo test

# Run experiments with output
cargo test experiments -- --nocapture

# Run a specific experiment
cargo test exp1_similarity -- --nocapture

# Run benchmarks
cargo bench
```

## Project Structure

```
src/
├── hdc/           # Hyperdimensional Computing
│   ├── binary.rs      # BinaryHV: Vec<u64>, XOR-based operations
│   ├── real.rs        # RealHV: Vec<f32>, multiply-based operations
│   ├── memory.rs      # Item Memory (associative store)
│   ├── encoder.rs     # Compositional encoding (sequences, records, sets)
│   └── algebra.rs     # Algebraic property verification
├── lnn/           # Liquid Neural Networks
│   ├── neuron.rs      # CfC (Closed-form Continuous-depth) neuron
│   ├── synapse.rs     # Synaptic dynamics
│   ├── ncp.rs         # Neural Circuit Policy (tiny/standard presets)
│   └── integrator.rs  # ODE integrators (Euler, RK4)
├── noesis/        # Core integration
│   ├── bridge.rs      # 4 bridge strategies (A/B/C/D)
│   ├── semantic_field.rs  # SemanticField: HDC state + LNN dynamics
│   ├── attractor.rs   # Attractor detection and analysis
│   ├── memory_dynamics.rs # Semantic memory with decay
│   └── temporal_binding.rs # Temporal binding
├── language/      # Language processing
│   ├── tokenizer.rs   # Italian tokenization
│   ├── vocabulary.rs  # HDC vocabulary with distributional learning
│   ├── composer.rs    # Sentence composition
│   ├── context.rs     # Conversational context
│   └── grounding.rs   # Symbol grounding with semantic roles
├── experiments/   # 7 research experiments
└── utils/
    └── corpus.rs      # 325-sentence Italian corpus
```

## Key Results

- **DualTrack bridge** best preserves compositionality (~19% top-1, ~42% top-3)
- **Distributional learning** produces meaningful semantic similarities (gatto-felino > gatto-treno)
- **CfC neurons** develop differentiated time constants under varying input
- **Permutation-based encoding** captures word order in sentences

See `results/analysis.md` for the full analysis report.

## Dependencies

Pure Rust, no ML frameworks:
- `rand` — random number generation
- `rand_distr` — normal distribution
- `rayon` — parallelism
- `serde` / `serde_json` — serialization
- `thiserror` — error types

## Language

The system processes Italian text. The corpus covers 5 themes: animals, food, work, weather, family (325 sentences with shared vocabulary for distributional learning).
