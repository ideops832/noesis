//! REST API server using std::net only. No external crates.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::Sender;

/// An API command sent from the REST handler to the main loop.
pub enum ApiCommand {
    Feed { text: String, response_tx: Sender<String> },
    Query { text: String, response_tx: Sender<String> },
    Reset { response_tx: Sender<String> },
    Status { response_tx: Sender<String> },
    Idle { steps: usize, response_tx: Sender<String> },
    // Proto 2 ring commands
    FeedNode { node_id: usize, text: String, response_tx: Sender<String> },
    QueryNode { node_id: usize, text: String, response_tx: Sender<String> },
    RingStep { response_tx: Sender<String> },
    IdleAll { steps: usize, response_tx: Sender<String> },
    // Proto 3 torus commands
    FeedTorus { node_id: usize, text: String, response_tx: Sender<String> },
    FeedTorusAll { text: String, response_tx: Sender<String> },
    QueryTorus { node_id: usize, text: String, response_tx: Sender<String> },
    TorusStep { response_tx: Sender<String> },
    TorusCoupling { value: f32, response_tx: Sender<String> },
    TorusIdleAll { steps: usize, response_tx: Sender<String> },
}

/// A parsed HTTP request.
pub struct HttpRequest {
    pub method: String,
    pub path: String,
    pub body: String,
}

/// Parse an HTTP request from a TcpStream.
pub fn parse_request(stream: &mut TcpStream) -> Option<HttpRequest> {
    // Read initial data (headers + possibly partial body)
    let mut data = Vec::with_capacity(8192);
    let mut buf = [0u8; 8192];
    let n = match stream.read(&mut buf) {
        Ok(n) if n > 0 => n,
        _ => return None,
    };
    data.extend_from_slice(&buf[..n]);

    let raw = match std::str::from_utf8(&data) {
        Ok(s) => s.to_string(),
        Err(_) => return None,
    };

    // Parse request line
    let request_line = raw.lines().next()?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next()?.to_string();
    let path = parts.next()?.to_string();

    // Find Content-Length
    let mut content_length: usize = 0;
    for line in raw.lines() {
        if line.is_empty() { break; }
        if line.to_lowercase().starts_with("content-length:") {
            if let Some(len_str) = line.splitn(2, ':').nth(1) {
                content_length = len_str.trim().parse().unwrap_or(0);
            }
        }
    }

    // Find header-body boundary
    let header_end_pos = raw.find("\r\n\r\n").map(|p| p + 4)
        .or_else(|| raw.find("\n\n").map(|p| p + 2))
        .unwrap_or(raw.len());

    // Read remaining body if Content-Length says we're missing data
    let body_so_far = if header_end_pos < raw.len() {
        raw[header_end_pos..].to_string()
    } else {
        String::new()
    };

    let body = if content_length > 0 && body_so_far.len() < content_length {
        // Need to read more data
        let mut full_body = body_so_far.into_bytes();
        let remaining = content_length - full_body.len();
        let mut extra = vec![0u8; remaining];
        let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(2)));
        match stream.read_exact(&mut extra) {
            Ok(_) => {
                full_body.extend_from_slice(&extra);
                String::from_utf8(full_body).unwrap_or_default()
            }
            Err(_) => String::from_utf8(full_body).unwrap_or_default(),
        }
    } else {
        body_so_far
    };

    // Truncate body to content_length if specified
    let body = if content_length > 0 && body.len() > content_length {
        body[..content_length].to_string()
    } else {
        body
    };

    Some(HttpRequest { method, path, body })
}

/// Send an HTTP response with the given status code and body.
pub fn send_response(stream: &mut TcpStream, status: u16, body: &str) {
    let status_text = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        500 => "Internal Server Error",
        _ => "Unknown",
    };

    let response = format!(
        "HTTP/1.1 {} {}\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\
         Access-Control-Allow-Headers: Content-Type\r\n\
         Connection: close\r\n\
         \r\n\
         {}",
        status,
        status_text,
        body.len(),
        body
    );

    let _ = stream.write_all(response.as_bytes());
}

/// Send a static file as an HTTP response.
pub fn send_file(stream: &mut TcpStream, path: &str) {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => {
            send_response(stream, 404, r#"{"error":"file not found"}"#);
            return;
        }
    };

    let content_type = if path.ends_with(".html") {
        "text/html; charset=utf-8"
    } else if path.ends_with(".js") {
        "application/javascript"
    } else if path.ends_with(".css") {
        "text/css"
    } else if path.ends_with(".json") {
        "application/json"
    } else {
        "text/plain"
    };

    let response = format!(
        "HTTP/1.1 200 OK\r\n\
         Content-Type: {}\r\n\
         Content-Length: {}\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Connection: close\r\n\
         \r\n\
         {}",
        content_type,
        content.len(),
        content
    );

    let _ = stream.write_all(response.as_bytes());
}

/// Extract the "text" field from a JSON body (minimal parsing, no serde needed).
fn extract_json_field(body: &str, field: &str) -> Option<String> {
    // Try serde_json first for robustness
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(body) {
        return v.get(field)?.as_str().map(|s| s.to_string());
    }
    None
}

/// Extract an integer field from a JSON body.
fn extract_json_int(body: &str, field: &str) -> Option<usize> {
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(body) {
        return v.get(field)?.as_u64().map(|n| n as usize);
    }
    None
}

/// Extract a float field from a JSON body.
fn extract_json_float(body: &str, field: &str) -> Option<f32> {
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(body) {
        return v.get(field)?.as_f64().map(|n| n as f32);
    }
    None
}

/// Start the REST API server.
///
/// Listens on the given port and dispatches commands to the main loop
/// via the provided channel.
pub fn start_api_server(port: u16, cmd_tx: Sender<ApiCommand>) {
    let addr = format!("0.0.0.0:{}", port);
    let listener = match TcpListener::bind(&addr) {
        Ok(l) => {
            println!("[API] Listening on {}", addr);
            l
        }
        Err(e) => {
            eprintln!("\n[ERROR] Cannot bind API port {}: {}", port, e);
            eprintln!("  Hint: kill previous instance with: lsof -ti:{} | xargs kill -9\n", port);
            std::process::exit(1);
        }
    };

    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                let cmd_tx = cmd_tx.clone();
                std::thread::spawn(move || {
                    handle_request(&mut stream, &cmd_tx);
                });
            }
            Err(e) => {
                eprintln!("[API] Accept error: {}", e);
            }
        }
    }
}

fn handle_request(stream: &mut TcpStream, cmd_tx: &Sender<ApiCommand>) {
    let request = match parse_request(stream) {
        Some(r) => r,
        None => return,
    };

    // Handle CORS preflight
    if request.method == "OPTIONS" {
        send_response(stream, 200, "");
        return;
    }

    match (request.method.as_str(), request.path.as_str()) {
        // Serve dashboard
        ("GET", "/") => {
            send_file(stream, "static/dashboard.html");
        }

        // Serve static files
        ("GET", path) if path.starts_with("/static/") => {
            let file_path = &path[1..]; // Remove leading /
            send_file(stream, file_path);
        }

        // POST /feed — feed text to the field
        ("POST", "/feed") => {
            let text = match extract_json_field(&request.body, "text") {
                Some(t) if !t.is_empty() => t,
                _ => {
                    send_response(stream, 400, r#"{"error":"missing 'text' field"}"#);
                    return;
                }
            };
            let (response_tx, response_rx) = std::sync::mpsc::channel();
            let cmd = ApiCommand::Feed { text, response_tx };
            if cmd_tx.send(cmd).is_err() {
                send_response(stream, 500, r#"{"error":"engine unavailable"}"#);
                return;
            }
            match response_rx.recv_timeout(std::time::Duration::from_secs(10)) {
                Ok(resp) => send_response(stream, 200, &resp),
                Err(_) => send_response(stream, 500, r#"{"error":"timeout"}"#),
            }
        }

        // POST /query — query the field
        ("POST", "/query") => {
            let text = match extract_json_field(&request.body, "text") {
                Some(t) => t,
                None => String::new(),
            };
            let (response_tx, response_rx) = std::sync::mpsc::channel();
            let cmd = ApiCommand::Query { text, response_tx };
            if cmd_tx.send(cmd).is_err() {
                send_response(stream, 500, r#"{"error":"engine unavailable"}"#);
                return;
            }
            match response_rx.recv_timeout(std::time::Duration::from_secs(10)) {
                Ok(resp) => send_response(stream, 200, &resp),
                Err(_) => send_response(stream, 500, r#"{"error":"timeout"}"#),
            }
        }

        // POST /reset — reset the field
        ("POST", "/reset") => {
            let (response_tx, response_rx) = std::sync::mpsc::channel();
            let cmd = ApiCommand::Reset { response_tx };
            if cmd_tx.send(cmd).is_err() {
                send_response(stream, 500, r#"{"error":"engine unavailable"}"#);
                return;
            }
            match response_rx.recv_timeout(std::time::Duration::from_secs(10)) {
                Ok(resp) => send_response(stream, 200, &resp),
                Err(_) => send_response(stream, 500, r#"{"error":"timeout"}"#),
            }
        }

        // GET /status — get engine status
        ("GET", "/status") => {
            let (response_tx, response_rx) = std::sync::mpsc::channel();
            let cmd = ApiCommand::Status { response_tx };
            if cmd_tx.send(cmd).is_err() {
                send_response(stream, 500, r#"{"error":"engine unavailable"}"#);
                return;
            }
            match response_rx.recv_timeout(std::time::Duration::from_secs(10)) {
                Ok(resp) => send_response(stream, 200, &resp),
                Err(_) => send_response(stream, 500, r#"{"error":"timeout"}"#),
            }
        }

        // POST /idle — advance idle steps
        ("POST", "/idle") => {
            let steps = extract_json_int(&request.body, "steps").unwrap_or(10);
            let (response_tx, response_rx) = std::sync::mpsc::channel();
            let cmd = ApiCommand::Idle { steps, response_tx };
            if cmd_tx.send(cmd).is_err() {
                send_response(stream, 500, r#"{"error":"engine unavailable"}"#);
                return;
            }
            match response_rx.recv_timeout(std::time::Duration::from_secs(10)) {
                Ok(resp) => send_response(stream, 200, &resp),
                Err(_) => send_response(stream, 500, r#"{"error":"timeout"}"#),
            }
        }

        // Serve ring dashboard
        ("GET", "/ring") => {
            send_file(stream, "static/dashboard_ring.html");
        }

        // POST /feed/<node_id> — feed text to a specific ring node
        ("POST", path) if path.starts_with("/feed/") => {
            let node_id_str = &path[6..];
            let node_id: usize = match node_id_str.parse() {
                Ok(id) => id,
                Err(_) => {
                    send_response(stream, 400, r#"{"error":"invalid node_id"}"#);
                    return;
                }
            };
            let text = match extract_json_field(&request.body, "text") {
                Some(t) if !t.is_empty() => t,
                _ => {
                    send_response(stream, 400, r#"{"error":"missing 'text' field"}"#);
                    return;
                }
            };
            let (response_tx, response_rx) = std::sync::mpsc::channel();
            let cmd = ApiCommand::FeedNode { node_id, text, response_tx };
            if cmd_tx.send(cmd).is_err() {
                send_response(stream, 500, r#"{"error":"engine unavailable"}"#);
                return;
            }
            match response_rx.recv_timeout(std::time::Duration::from_secs(10)) {
                Ok(resp) => send_response(stream, 200, &resp),
                Err(_) => send_response(stream, 500, r#"{"error":"timeout"}"#),
            }
        }

        // POST /query/<node_id> — query a specific ring node
        ("POST", path) if path.starts_with("/query/") => {
            let node_id_str = &path[7..];
            let node_id: usize = match node_id_str.parse() {
                Ok(id) => id,
                Err(_) => {
                    send_response(stream, 400, r#"{"error":"invalid node_id"}"#);
                    return;
                }
            };
            let text = match extract_json_field(&request.body, "text") {
                Some(t) => t,
                None => String::new(),
            };
            let (response_tx, response_rx) = std::sync::mpsc::channel();
            let cmd = ApiCommand::QueryNode { node_id, text, response_tx };
            if cmd_tx.send(cmd).is_err() {
                send_response(stream, 500, r#"{"error":"engine unavailable"}"#);
                return;
            }
            match response_rx.recv_timeout(std::time::Duration::from_secs(10)) {
                Ok(resp) => send_response(stream, 200, &resp),
                Err(_) => send_response(stream, 500, r#"{"error":"timeout"}"#),
            }
        }

        // POST /ring/step — trigger a ring propagation step
        ("POST", "/ring/step") => {
            let (response_tx, response_rx) = std::sync::mpsc::channel();
            let cmd = ApiCommand::RingStep { response_tx };
            if cmd_tx.send(cmd).is_err() {
                send_response(stream, 500, r#"{"error":"engine unavailable"}"#);
                return;
            }
            match response_rx.recv_timeout(std::time::Duration::from_secs(10)) {
                Ok(resp) => send_response(stream, 200, &resp),
                Err(_) => send_response(stream, 500, r#"{"error":"timeout"}"#),
            }
        }

        // POST /idle/all — idle all ring nodes
        ("POST", "/idle/all") => {
            let steps = extract_json_int(&request.body, "steps").unwrap_or(10);
            let (response_tx, response_rx) = std::sync::mpsc::channel();
            let cmd = ApiCommand::IdleAll { steps, response_tx };
            if cmd_tx.send(cmd).is_err() {
                send_response(stream, 500, r#"{"error":"engine unavailable"}"#);
                return;
            }
            match response_rx.recv_timeout(std::time::Duration::from_secs(10)) {
                Ok(resp) => send_response(stream, 200, &resp),
                Err(_) => send_response(stream, 500, r#"{"error":"timeout"}"#),
            }
        }

        // --- Proto 3: Torus endpoints ---

        // Serve torus dashboard
        ("GET", "/torus") => {
            send_file(stream, "static/dashboard_torus.html");
        }

        // POST /torus/feed/all — broadcast feed to all torus nodes
        ("POST", "/torus/feed/all") => {
            let text = match extract_json_field(&request.body, "text") {
                Some(t) if !t.is_empty() => t,
                _ => {
                    send_response(stream, 400, r#"{"error":"missing 'text' field"}"#);
                    return;
                }
            };
            let (response_tx, response_rx) = std::sync::mpsc::channel();
            let cmd = ApiCommand::FeedTorusAll { text, response_tx };
            if cmd_tx.send(cmd).is_err() {
                send_response(stream, 500, r#"{"error":"engine unavailable"}"#);
                return;
            }
            match response_rx.recv_timeout(std::time::Duration::from_secs(10)) {
                Ok(resp) => send_response(stream, 200, &resp),
                Err(_) => send_response(stream, 500, r#"{"error":"timeout"}"#),
            }
        }

        // POST /torus/feed/<node_id> — feed text to a specific torus node
        ("POST", path) if path.starts_with("/torus/feed/") => {
            let node_id_str = &path[12..];
            let node_id: usize = match node_id_str.parse() {
                Ok(id) => id,
                Err(_) => {
                    send_response(stream, 400, r#"{"error":"invalid node_id"}"#);
                    return;
                }
            };
            let text = match extract_json_field(&request.body, "text") {
                Some(t) if !t.is_empty() => t,
                _ => {
                    send_response(stream, 400, r#"{"error":"missing 'text' field"}"#);
                    return;
                }
            };
            let (response_tx, response_rx) = std::sync::mpsc::channel();
            let cmd = ApiCommand::FeedTorus { node_id, text, response_tx };
            if cmd_tx.send(cmd).is_err() {
                send_response(stream, 500, r#"{"error":"engine unavailable"}"#);
                return;
            }
            match response_rx.recv_timeout(std::time::Duration::from_secs(10)) {
                Ok(resp) => send_response(stream, 200, &resp),
                Err(_) => send_response(stream, 500, r#"{"error":"timeout"}"#),
            }
        }

        // POST /torus/query/<node_id> — query a specific torus node
        ("POST", path) if path.starts_with("/torus/query/") => {
            let node_id_str = &path[13..];
            let node_id: usize = match node_id_str.parse() {
                Ok(id) => id,
                Err(_) => {
                    send_response(stream, 400, r#"{"error":"invalid node_id"}"#);
                    return;
                }
            };
            let text = match extract_json_field(&request.body, "text") {
                Some(t) => t,
                None => String::new(),
            };
            let (response_tx, response_rx) = std::sync::mpsc::channel();
            let cmd = ApiCommand::QueryTorus { node_id, text, response_tx };
            if cmd_tx.send(cmd).is_err() {
                send_response(stream, 500, r#"{"error":"engine unavailable"}"#);
                return;
            }
            match response_rx.recv_timeout(std::time::Duration::from_secs(10)) {
                Ok(resp) => send_response(stream, 200, &resp),
                Err(_) => send_response(stream, 500, r#"{"error":"timeout"}"#),
            }
        }

        // POST /torus/step — trigger a torus propagation step
        ("POST", "/torus/step") => {
            let (response_tx, response_rx) = std::sync::mpsc::channel();
            let cmd = ApiCommand::TorusStep { response_tx };
            if cmd_tx.send(cmd).is_err() {
                send_response(stream, 500, r#"{"error":"engine unavailable"}"#);
                return;
            }
            match response_rx.recv_timeout(std::time::Duration::from_secs(10)) {
                Ok(resp) => send_response(stream, 200, &resp),
                Err(_) => send_response(stream, 500, r#"{"error":"timeout"}"#),
            }
        }

        // POST /torus/coupling — set torus coupling learning rate
        ("POST", "/torus/coupling") => {
            let value = match extract_json_float(&request.body, "value") {
                Some(v) => v,
                None => {
                    send_response(stream, 400, r#"{"error":"missing 'value' field"}"#);
                    return;
                }
            };
            let (response_tx, response_rx) = std::sync::mpsc::channel();
            let cmd = ApiCommand::TorusCoupling { value, response_tx };
            if cmd_tx.send(cmd).is_err() {
                send_response(stream, 500, r#"{"error":"engine unavailable"}"#);
                return;
            }
            match response_rx.recv_timeout(std::time::Duration::from_secs(10)) {
                Ok(resp) => send_response(stream, 200, &resp),
                Err(_) => send_response(stream, 500, r#"{"error":"timeout"}"#),
            }
        }

        // POST /torus/idle — idle all torus nodes
        ("POST", "/torus/idle") => {
            let steps = extract_json_int(&request.body, "steps").unwrap_or(10);
            let (response_tx, response_rx) = std::sync::mpsc::channel();
            let cmd = ApiCommand::TorusIdleAll { steps, response_tx };
            if cmd_tx.send(cmd).is_err() {
                send_response(stream, 500, r#"{"error":"engine unavailable"}"#);
                return;
            }
            match response_rx.recv_timeout(std::time::Duration::from_secs(10)) {
                Ok(resp) => send_response(stream, 200, &resp),
                Err(_) => send_response(stream, 500, r#"{"error":"timeout"}"#),
            }
        }

        _ => {
            send_response(stream, 404, r#"{"error":"not found"}"#);
        }
    }
}
