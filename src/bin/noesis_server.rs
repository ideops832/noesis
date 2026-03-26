use std::sync::mpsc;
use std::time::{Duration, Instant};
use noesis::language::tokenizer::Tokenizer;
use noesis::language::vocabulary::Vocabulary;
use noesis::language::composer::Composer;
use noesis::noesis::multiscale::{MultiScaleConfig, MultiScaleField};
use noesis::hdc::real::RealHV;
#[allow(unused_imports)]
use noesis::hdc::hypervector::HyperVector;
use noesis::utils::corpus;
use noesis::server::{state, websocket, api};

fn main() {
    println!("NOESIS Proto 1 — Initializing...");

    // 1. Build trained vocabulary
    let tokenizer = Tokenizer::new();
    let mut all = corpus::expanded_corpus();
    all.extend(corpus::synonym_parallel_corpus());
    // Don't include relational_corpus to keep init fast
    let sentences: Vec<Vec<String>> = all.iter().map(|s| tokenizer.tokenize(s)).collect();

    let mut composer = Composer::new(1024, 42);
    for s in &sentences { for w in s { composer.vocabulary.get_or_create(w); } }
    for _ in 0..3 { composer.vocabulary.learn_from_context_momentum(&sentences, 3, 0.7); }
    println!("  Vocabulary: {} words", composer.vocabulary.len());

    // 2. Create MultiScaleField
    let ms_config = MultiScaleConfig::default_for_dim(1024);
    let mut field = MultiScaleField::new(ms_config, 42);
    println!("  MultiScaleField: 3 scales (fast, medium, slow)");

    // 3. Start servers
    let (_ws_handle, ws_clients) = websocket::start_ws_server(8080);
    let (cmd_tx, cmd_rx) = mpsc::channel();
    let _api_handle = std::thread::spawn(move || api::start_api_server(8081, cmd_tx));
    println!("  WebSocket: ws://localhost:8080");
    println!("  REST API:  http://localhost:8081");
    println!("  Dashboard: http://localhost:8081/");
    println!("\nNOESIS Proto 1 ready.\n");

    // 4. Main loop
    let mut narrative: Vec<state::NarrativePoint> = Vec::new();
    let mut last_event: Option<state::Event> = None;
    let mut total_steps: u64 = 0;
    let mut turn_count: u32 = 0;
    let mut prev_state: Option<RealHV> = None;
    let mut last_input_time = Instant::now();
    let start_time = Instant::now();

    loop {
        let loop_start = Instant::now();

        // Process commands from REST API (non-blocking)
        while let Ok(cmd) = cmd_rx.try_recv() {
            match cmd {
                api::ApiCommand::Feed { text, response_tx } => {
                    let hv = composer.encode_sentence(&text);
                    if let Some(hv) = hv {
                        field.step(&hv);
                        total_steps += 1;
                        turn_count += 1;
                        last_input_time = Instant::now();
                        last_event = Some(state::Event {
                            event_type: "feed".into(),
                            text: text.clone(),
                            result: None,
                        });
                        let tokens: Vec<String> = tokenizer.tokenize(&text);
                        let _ = response_tx.send(format!(
                            r#"{{"status":"ok","tokens":{:?},"turn":{}}}"#,
                            tokens, turn_count
                        ));
                    } else {
                        let _ = response_tx.send(
                            r#"{"status":"error","message":"empty encoding"}"#.into(),
                        );
                    }
                }
                api::ApiCommand::Query { text, response_tx } => {
                    // Query doesn't change state
                    let salient = get_salient(&field, &composer.vocabulary, 5);
                    let results: Vec<String> = salient
                        .iter()
                        .map(|(w, s)| {
                            format!(r#"{{"concept":"{}","similarity":{:.4}}}"#, w, s)
                        })
                        .collect();
                    last_event = Some(state::Event {
                        event_type: "query".into(),
                        text: text.clone(),
                        result: Some(format!("[{}]", results.join(","))),
                    });
                    let _ = response_tx.send(format!(
                        r#"{{"status":"ok","results":[{}]}}"#,
                        results.join(",")
                    ));
                }
                api::ApiCommand::Reset { response_tx } => {
                    field.reset();
                    total_steps = 0;
                    turn_count = 0;
                    narrative.clear();
                    last_event = Some(state::Event {
                        event_type: "reset".into(),
                        text: "reset".into(),
                        result: None,
                    });
                    let _ =
                        response_tx.send(r#"{"status":"ok","state":"reset"}"#.into());
                }
                api::ApiCommand::Status { response_tx } => {
                    let uptime = start_time.elapsed().as_secs();
                    let _ = response_tx.send(format!(
                        r#"{{"engine":"noesis","version":"0.1.0","dim":1024,"neurons":20,"scales":3,"vocab_size":{},"total_steps":{},"uptime_seconds":{}}}"#,
                        composer.vocabulary.len(),
                        total_steps,
                        uptime
                    ));
                }
                api::ApiCommand::Idle { steps, response_tx } => {
                    for _ in 0..steps {
                        field.idle_step();
                        total_steps += 1;
                    }
                    last_event = Some(state::Event {
                        event_type: "idle".into(),
                        text: format!("sleep {}", steps),
                        result: None,
                    });
                    let _ = response_tx.send(format!(
                        r#"{{"status":"ok","idle_steps":{}}}"#,
                        steps
                    ));
                }
            }
        }

        // Build frame
        let salient = get_salient(&field, &composer.vocabulary, 10);
        let velocity = if let Some(prev) = &prev_state {
            let curr = field.combined_state();
            if prev.norm() > 1e-8 && curr.norm() > 1e-8 {
                1.0 - RealHV::cosine_similarity(prev, curr)
            } else {
                0.0
            }
        } else {
            0.0
        };

        let novelty = velocity; // simplified: velocity IS novelty
        let idle_time = last_input_time.elapsed().as_secs_f32();

        let frame = state::build_frame(
            &salient,
            &composer.vocabulary,
            field.fast_state().norm(),
            field.medium_state().norm(),
            field.slow_state().norm(),
            velocity,
            novelty,
            total_steps,
            turn_count,
            idle_time,
            &narrative,
            last_event.clone(),
        );

        // Add to narrative (keep last 500)
        if let Some(top) = salient.first() {
            let cat = state::categorize_word(&top.0, &composer.vocabulary);
            narrative.push(state::NarrativePoint {
                timestamp_ms: start_time.elapsed().as_millis() as u64,
                dominant_concept: top.0.clone(),
                category: cat,
                coherence: top.1,
                event_type: if last_event.is_some() {
                    last_event.as_ref().unwrap().event_type.clone()
                } else {
                    "idle".into()
                },
            });
            if narrative.len() > 500 {
                narrative.remove(0);
            }
        }

        // Broadcast via WebSocket
        let json = serde_json::to_string(&frame).unwrap_or_default();
        websocket::broadcast(&ws_clients, &json);

        // Save previous state
        prev_state = Some(field.combined_state().clone());
        last_event = None;

        // Sleep to maintain ~10fps
        let elapsed = loop_start.elapsed();
        if elapsed < Duration::from_millis(100) {
            std::thread::sleep(Duration::from_millis(100) - elapsed);
        }
    }
}

fn get_salient(
    field: &MultiScaleField,
    vocab: &Vocabulary,
    k: usize,
) -> Vec<(String, f32)> {
    let state = field.combined_state();
    if state.norm() < 1e-8 {
        return Vec::new();
    }
    let mut sims: Vec<(String, f32)> = vocab
        .words
        .iter()
        .map(|(word, hv)| (word.clone(), RealHV::cosine_similarity(state, hv)))
        .collect();
    sims.sort_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    sims.truncate(k);
    sims
}
