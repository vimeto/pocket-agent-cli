#!/usr/bin/env python3
"""Minimal OpenAI-compatible server for Gemma 3n using mlx_lm.generate.

The built-in mlx_lm.server has a bug with Gemma 3n's shared KV cache
when using BatchGenerator. This workaround uses the single-sequence
generate() path, which works correctly.

Only implements /v1/models and /v1/chat/completions (non-streaming).
"""

import json
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

MODEL_ID = "google/gemma-3n-E2B-it"
PORT = 8080

# Load model once at startup
print(f"Loading {MODEL_ID}...")
model, tokenizer = load(MODEL_ID)
print(f"Model loaded. Starting server on port {PORT}...")

# Serialize generation to avoid concurrent access to the model
generate_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Quieter logging
        pass

    def do_GET(self):
        if self.path == "/v1/models":
            data = {
                "object": "list",
                "data": [
                    {
                        "id": MODEL_ID,
                        "object": "model",
                        "created": int(time.time()),
                    }
                ],
            }
            self._respond(200, data)
        else:
            self._respond(404, {"error": "Not found"})

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length))
            self._handle_chat(body)
        else:
            self._respond(404, {"error": "Not found"})

    def _handle_chat(self, body):
        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens", 2048)
        temperature = body.get("temperature", 0.7)

        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback: manual formatting
            parts = []
            for m in messages:
                role = m["role"]
                content = m["content"]
                parts.append(f"<start_of_turn>{role}\n{content}<end_of_turn>")
            parts.append("<start_of_turn>model\n")
            prompt = "\n".join(parts)

        sampler = make_sampler(temp=temperature)

        with generate_lock:
            t0 = time.time()
            response_text = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
            )
            gen_time = time.time() - t0

        # Estimate token counts (rough)
        prompt_tokens = len(tokenizer.encode(prompt))
        completion_tokens = len(tokenizer.encode(response_text))

        data = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "model": MODEL_ID,
            "created": int(time.time()),
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                    },
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        self._respond(200, data)

    def _respond(self, status, data):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Serving on http://127.0.0.1:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()
