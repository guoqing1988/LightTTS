# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the API Server
The core of LightTTS is the API server.
```bash
# Basic startup (requires model_dir)
python -m light_tts.server.api_server --model_dir ./pretrained_models/Fun-CosyVoice3-0.5B-2512

# Startup with common options
python -m light_tts.server.api_server --host 0.0.0.0 --port 8080 --data_type float16 --load_trt True --model_dir <model_path>
```

### Running Tests & Benchmarks
Tests are located in the `test/` directory and are typically run as standalone scripts.
```bash
# Test zero-shot inference (non-streaming and streaming)
python test/test_zero_shot.py

# Test bi-streaming mode via WebSocket
python test/test_bistream.py

# Performance benchmarks
python test/test_zs_speed.py   # Non-streaming speed
python test/test_zs_stream.py  # Streaming speed
```

### Installation
```bash
# Install core dependencies
pip install -r requirements.txt

# For hardware acceleration (requires CUDA 12)
pip install tensorrt-cu12-libs tensorrt-cu12-bindings tensorrt-cu12
pip install onnxruntime-gpu==1.23.2 --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple
```

## Architecture & Structure

LightTTS is a modular TTS inference framework optimized for high throughput and low latency. It is refactored from the LightLLM architecture.

### High-Level Components
- **Modular Pipeline (Encode–LLM–Decode)**: The TTS process is decoupled into three modules that run as separate processes to allow for efficient task parallelism and scalability.
- **Shared Memory Timbre Manager**: Manages speaker and timbre embeddings in shared memory with an LRU (Least Recently Used) cache to minimize recomputation and memory overhead.
- **Server Layer**: Uses FastAPI for HTTP endpoints and WebSockets for bi-directional streaming.
- **Inference Engines**:
    - **LLM**: Accelerated using techniques from LightLLM (CUDA graphs, flash decoding).
    - **Flow/Decoder**: Supports TensorRT and JIT acceleration.

### Key Directories
- `light_tts/`: Main package containing the server, model implementations, and shared memory management.
- `cosyvoice/`: Integration and modifications of the original CosyVoice architecture.
- `test/`: Functional tests and performance benchmarking scripts.
- `asset/`: Default prompts and audio samples for testing.

### Design Patterns
- **Process-based Parallelism**: Each major stage of the TTS pipeline (Encoder, LLM, Decoder) typically runs in its own process, communicating via shared memory or IPC.
- **Streaming by Default**: Designed to support Time-to-First-Token (TTFT) optimization through incremental decoding and audio chunking.
