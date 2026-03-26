//! Minimal WebSocket server using std::net only. No external crates.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

/// The WebSocket magic string used in the handshake.
const WS_MAGIC: &str = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

/// List of connected WebSocket clients.
pub type WsSender = Arc<Mutex<Vec<TcpStream>>>;

// ---------- Minimal SHA-1 implementation (for WebSocket handshake only) ----------

fn sha1(data: &[u8]) -> [u8; 20] {
    let mut h0: u32 = 0x67452301;
    let mut h1: u32 = 0xEFCDAB89;
    let mut h2: u32 = 0x98BADCFE;
    let mut h3: u32 = 0x10325476;
    let mut h4: u32 = 0xC3D2E1F0;

    // Pre-processing: pad message
    let bit_len = (data.len() as u64) * 8;
    let mut msg = data.to_vec();
    msg.push(0x80);
    while (msg.len() % 64) != 56 {
        msg.push(0x00);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    // Process each 512-bit (64-byte) chunk
    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 80];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[i * 4],
                chunk[i * 4 + 1],
                chunk[i * 4 + 2],
                chunk[i * 4 + 3],
            ]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }

        let (mut a, mut b, mut c, mut d, mut e) = (h0, h1, h2, h3, h4);

        for i in 0..80 {
            let (f, k) = match i {
                0..=19 => ((b & c) | ((!b) & d), 0x5A827999u32),
                20..=39 => (b ^ c ^ d, 0x6ED9EBA1u32),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1BBCDCu32),
                _ => (b ^ c ^ d, 0xCA62C1D6u32),
            };
            let temp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(w[i]);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = temp;
        }

        h0 = h0.wrapping_add(a);
        h1 = h1.wrapping_add(b);
        h2 = h2.wrapping_add(c);
        h3 = h3.wrapping_add(d);
        h4 = h4.wrapping_add(e);
    }

    let mut result = [0u8; 20];
    result[0..4].copy_from_slice(&h0.to_be_bytes());
    result[4..8].copy_from_slice(&h1.to_be_bytes());
    result[8..12].copy_from_slice(&h2.to_be_bytes());
    result[12..16].copy_from_slice(&h3.to_be_bytes());
    result[16..20].copy_from_slice(&h4.to_be_bytes());
    result
}

// ---------- Base64 encoding ----------

const BASE64_CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64_encode(data: &[u8]) -> String {
    let mut result = String::new();
    let chunks = data.chunks(3);
    for chunk in chunks {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;

        result.push(BASE64_CHARS[((triple >> 18) & 0x3F) as usize] as char);
        result.push(BASE64_CHARS[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            result.push(BASE64_CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(BASE64_CHARS[(triple & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}

// ---------- WebSocket handshake ----------

/// Compute the Sec-WebSocket-Accept value from the client's key.
pub fn ws_accept_key(client_key: &str) -> String {
    let combined = format!("{}{}", client_key.trim(), WS_MAGIC);
    let hash = sha1(combined.as_bytes());
    base64_encode(&hash)
}

/// Perform the WebSocket handshake on a TcpStream.
/// Returns true if the handshake succeeds.
pub fn ws_handshake(stream: &mut TcpStream) -> bool {
    let mut buf = [0u8; 4096];
    let n = match stream.read(&mut buf) {
        Ok(n) if n > 0 => n,
        _ => return false,
    };

    let request = match std::str::from_utf8(&buf[..n]) {
        Ok(s) => s,
        Err(_) => return false,
    };

    // Check it's a WebSocket upgrade request
    if !request.contains("Upgrade: websocket") && !request.contains("Upgrade: Websocket") {
        return false;
    }

    // Extract Sec-WebSocket-Key
    let key = request
        .lines()
        .find(|line| line.to_lowercase().starts_with("sec-websocket-key:"))
        .and_then(|line| line.splitn(2, ':').nth(1))
        .map(|k| k.trim().to_string());

    let key = match key {
        Some(k) => k,
        None => return false,
    };

    let accept = ws_accept_key(&key);

    let response = format!(
        "HTTP/1.1 101 Switching Protocols\r\n\
         Upgrade: websocket\r\n\
         Connection: Upgrade\r\n\
         Sec-WebSocket-Accept: {}\r\n\
         \r\n",
        accept
    );

    stream.write_all(response.as_bytes()).is_ok()
}

// ---------- WebSocket frame sending ----------

/// Send a text frame over a WebSocket connection.
pub fn ws_send_text(stream: &mut TcpStream, text: &str) -> std::io::Result<()> {
    let payload = text.as_bytes();
    let len = payload.len();

    // Text frame opcode = 0x81 (FIN + text)
    let mut frame = Vec::with_capacity(10 + len);
    frame.push(0x81);

    if len < 126 {
        frame.push(len as u8);
    } else if len < 65536 {
        frame.push(126);
        frame.push((len >> 8) as u8);
        frame.push((len & 0xFF) as u8);
    } else {
        frame.push(127);
        let len64 = len as u64;
        frame.extend_from_slice(&len64.to_be_bytes());
    }

    frame.extend_from_slice(payload);
    stream.write_all(&frame)
}

// ---------- Server ----------

/// Start the WebSocket server on the given port.
///
/// Returns a join handle for the server thread and a shared client list
/// that can be used to broadcast messages.
pub fn start_ws_server(port: u16) -> (std::thread::JoinHandle<()>, WsSender) {
    let clients: WsSender = Arc::new(Mutex::new(Vec::new()));
    let clients_clone = Arc::clone(&clients);

    let handle = std::thread::spawn(move || {
        let addr = format!("0.0.0.0:{}", port);
        let listener = match TcpListener::bind(&addr) {
            Ok(l) => {
                println!("[WS] Listening on {}", addr);
                l
            }
            Err(e) => {
                eprintln!("\n[ERROR] Cannot bind WebSocket port {}: {}", port, e);
                eprintln!("  Hint: kill previous instance with: lsof -ti:{} | xargs kill -9\n", port);
                std::process::exit(1);
            }
        };

        for stream in listener.incoming() {
            match stream {
                Ok(mut stream) => {
                    if ws_handshake(&mut stream) {
                        // Set a write timeout to avoid blocking on dead clients
                        let _ = stream.set_write_timeout(Some(std::time::Duration::from_secs(5)));
                        if let Ok(mut clients) = clients_clone.lock() {
                            clients.push(stream);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("[WS] Accept error: {}", e);
                }
            }
        }
    });

    (handle, clients)
}

/// Broadcast a text message to all connected WebSocket clients.
/// Removes clients that have disconnected.
pub fn broadcast(clients: &WsSender, text: &str) {
    if let Ok(mut clients) = clients.lock() {
        let mut i = 0;
        while i < clients.len() {
            match ws_send_text(&mut clients[i], text) {
                Ok(_) => i += 1,
                Err(_) => {
                    // Client disconnected — remove it
                    clients.swap_remove(i);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ws_accept_key() {
        // Standard test vector from RFC 6455 section 4.2.2
        let key = "dGhlIHNhbXBsZSBub25jZQ==";
        let accept = ws_accept_key(key);
        assert_eq!(accept, "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=");
    }

    #[test]
    fn test_sha1_empty() {
        let hash = sha1(b"");
        let hex: String = hash.iter().map(|b| format!("{:02x}", b)).collect();
        assert_eq!(hex, "da39a3ee5e6b4b0d3255bfef95601890afd80709");
    }

    #[test]
    fn test_base64_encode() {
        assert_eq!(base64_encode(b"Hello"), "SGVsbG8=");
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
    }
}
