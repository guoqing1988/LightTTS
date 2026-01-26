const { createApp, ref, onMounted, computed, nextTick } = Vue;

createApp({
    setup() {
        const form = ref({
            text: "2026年，我们正站在智能文明的潮头。浪潮汹涌，唯有勇者与智者能立于潮头。作为时代的“造浪者”，我们手中握有的，正是一副足以定义未来的“王炸”。这副王炸，就是“科艺商潮”。这不是一句简单的口号，而是我们基因里的超代码，是我们从“记录时代”迈向“定义时代”的终极武器。2026年，我们的任务只有一个：打好这副王炸，掀起属于超媒体的“Meta Wave”！",
            mode: "sft",
            speaker: "",
            prompt_text: "",
            prompt_wav_path: "",
            instruct_text: "",
            source_wav_path: "",
            stream: false,
            speed: 1.0,
            output_format: "pcm",
            protocol: "http" // 'http' or 'ws'
        });

        const health = ref({});
        const voices = ref([]);
        const loading = ref(false);
        const audioUrl = ref(null);
        const audioSampleRate = ref(24000);
        const error = ref(null);
        const audioPlayer = ref(null);
        const promptAudioPreview = ref(null);
        const sourceAudioPreview = ref(null);
        const uploading = ref(false);

        let abortController = null;
        let ws = null;
        let audioContext = null;
        let nextStartTime = 0;

        // --- Utilities ---

        const getAudioContext = (rate) => {
            const targetRate = rate || audioSampleRate.value;
            if (audioContext && audioContext.sampleRate !== targetRate) {
                audioContext.close();
                audioContext = null;
            }
            if (!audioContext) {
                audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: targetRate });
            }
            if (audioContext.state === 'suspended') {
                audioContext.resume();
            }
            return audioContext;
        };

        const getMimeType = (format) => {
            const fmt = (format || form.value.output_format).toLowerCase();
            if (fmt === 'mp3') return 'audio/mpeg';
            if (fmt === 'wav') return 'audio/wav';
            return 'audio/pcm';
        };

        const int16ToFloat32 = (buffer) => {
            const int16Array = new Int16Array(buffer);
            const float32Array = new Float32Array(int16Array.length);
            for (let i = 0; i < int16Array.length; i++) {
                float32Array[i] = int16Array[i] / 32768.0;
            }
            return float32Array;
        };

        const createWavBlob = (samples, sampleRate) => {
            const buffer = new ArrayBuffer(44 + samples.length * 2);
            const view = new DataView(buffer);
            const writeString = (offset, string) => {
                for (let i = 0; i < string.length; i++) {
                    view.setUint8(offset + i, string.charCodeAt(i));
                }
            };

            writeString(0, 'RIFF');
            view.setUint32(4, 36 + samples.length * 2, true);
            writeString(8, 'WAVE');
            writeString(12, 'fmt ');
            view.setUint32(16, 16, true);
            view.setUint16(20, 1, true);
            view.setUint16(22, 1, true);
            view.setUint32(24, sampleRate, true);
            view.setUint32(28, sampleRate * 2, true);
            view.setUint16(32, 2, true);
            view.setUint16(34, 16, true);
            writeString(36, 'data');
            view.setUint32(40, samples.length * 2, true);

            for (let i = 0; i < samples.length; i++) {
                const s = Math.max(-1, Math.min(1, samples[i]));
                view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
            }
            return new Blob([buffer], { type: 'audio/wav' });
        };

        // --- Core Playback Logic ---

        const playChunk = (float32Array) => {
            if (float32Array.length === 0) return;
            const ctx = getAudioContext();
            const audioBuffer = ctx.createBuffer(1, float32Array.length, ctx.sampleRate);
            audioBuffer.getChannelData(0).set(float32Array);

            const source = ctx.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(ctx.destination);

            const currentTime = ctx.currentTime;
            if (nextStartTime < currentTime) {
                nextStartTime = currentTime + 0.1; // 稍微增加缓冲时间解决微小破音
            }

            source.start(nextStartTime);
            nextStartTime += audioBuffer.duration;
        };

        const decodeAndPlayChunk = async (arrayBuffer) => {
            // decodeAudioData 会分离 (detach) buffer，需要复制一份
            const bufferCopy = arrayBuffer.slice(0);
            const ctx = getAudioContext();
            try {
                const audioBuffer = await ctx.decodeAudioData(bufferCopy);
                const source = ctx.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(ctx.destination);

                const currentTime = ctx.currentTime;
                if (nextStartTime < currentTime) {
                    nextStartTime = currentTime + 0.1;
                }

                source.start(nextStartTime);
                nextStartTime += audioBuffer.duration;
            } catch (e) {
                console.error("MP3 解码失败:", e);
            }
        };

        const finalizeAudio = (buffers, format) => {
            if (!buffers || buffers.length === 0) return;
            const fmt = format.toLowerCase();
            let blob;

            // 在流式和 WebSocket 逻辑中，PCM 和 WAV 最终都以 Float32Array 形式存在 buffers 中
            if (fmt === 'pcm' || fmt === 'wav') {
                const totalLength = buffers.reduce((sum, arr) => sum + (arr.length || 0), 0);
                const combined = new Float32Array(totalLength);
                let offset = 0;
                for (const arr of buffers) {
                    if (arr instanceof Float32Array) {
                        combined.set(arr, offset);
                        offset += arr.length;
                    }
                }
                blob = createWavBlob(combined, audioSampleRate.value);
            } else {
                // MP3 格式
                const totalLength = buffers.reduce((sum, arr) => {
                    if (arr instanceof Uint8Array) return sum + arr.length;
                    if (arr instanceof ArrayBuffer) return sum + arr.byteLength;
                    return sum + (arr.length || 0);
                }, 0);
                const combined = new Uint8Array(totalLength);
                let offset = 0;
                for (const arr of buffers) {
                    const view = (arr instanceof Uint8Array) ? arr : new Uint8Array(arr instanceof ArrayBuffer ? arr : arr.buffer);
                    combined.set(view, offset);
                    offset += view.length;
                }
                blob = new Blob([combined], { type: getMimeType(fmt) });
            }

            if (audioUrl.value) URL.revokeObjectURL(audioUrl.value);
            audioUrl.value = URL.createObjectURL(blob);

            // 只有非流式模式且非 WebSocket 才在这里触发自动播放
            // 流式模式在接收过程中已经实时播放过了
            if (!form.value.stream && form.value.protocol !== 'ws') {
                nextTick(() => audioPlayer.value?.play().catch(e => console.warn("Auto-play prevented", e)));
            }
        };

        // --- API Handlers ---

        const handleFileUpload = async (event, targetField) => {
            const file = event.target.files[0];
            if (!file) return;

            const formData = new FormData();
            formData.append('file', file);

            try {
                uploading.value = true;
                const res = await fetch('/v1/upload', { method: 'POST', body: formData });
                if (!res.ok) throw new Error((await res.json()).detail || '上传失败');

                const data = await res.json();
                form.value[targetField] = data.local_path;
                if (targetField === 'prompt_wav_path') {
                    promptAudioPreview.value = data.url;
                    if (data.text) form.value.prompt_text = data.text;
                } else {
                    sourceAudioPreview.value = data.url;
                }
            } catch (e) {
                alert("上传失败: " + e.message);
                event.target.value = '';
            } finally {
                uploading.value = false;
            }
        };

        const fetchHealth = async () => {
            try {
                const res = await fetch('/v1/health');
                health.value = await res.json();
            } catch (e) { health.value = { status: 'error' }; }
        };

        const fetchVoices = async () => {
            try {
                const res = await fetch('/v1/voices');
                voices.value = (await res.json()).voices;
            } catch (e) { console.error("Failed to fetch voices", e); }
        };

        const handleSync = async () => {
            const res = await fetch('/v1/tts', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(form.value),
                signal: abortController?.signal
            });

            if (!res.ok) throw new Error((await res.json()).detail || '请求失败');

            const data = await res.json();
            audioSampleRate.value = data.sample_rate;
            // 确保 AudioContext 与返回的采样率同步
            getAudioContext(data.sample_rate);

            const fmt = (data.format || form.value.output_format).toLowerCase();
            const binaryString = atob(data.audio);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) bytes[i] = binaryString.charCodeAt(i);

            if (fmt === 'pcm') {
                finalizeAudio([int16ToFloat32(bytes.buffer)], 'pcm');
            } else if (fmt === 'wav') {
                // 如果是 WAV，跳过 44 字节头并转为 pcm 模式统一处理下载
                const float32 = int16ToFloat32(bytes.buffer.slice(44));
                finalizeAudio([float32], 'pcm');
            } else {
                // MP3 等格式直接生成 Blob
                const blob = new Blob([bytes.buffer], { type: getMimeType(fmt) });
                audioUrl.value = URL.createObjectURL(blob);
                nextTick(() => audioPlayer.value?.play().catch(e => console.warn("Auto-play prevented", e)));
            }
        };

        const handleStreaming = async () => {
            const res = await fetch('/v1/tts', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ...form.value, stream: true }),
                signal: abortController?.signal
            });

            if (!res.ok) throw new Error((await res.json()).detail || '流式请求失败');

            // 从 Header 获取准确采样率
            const rate = parseInt(res.headers.get('X-Sample-Rate')) || audioSampleRate.value;
            audioSampleRate.value = rate;
            getAudioContext(rate);

            const reader = res.body.getReader();
            const buffers = [];
            let pendingBytes = new Uint8Array(0);
            let isFirstChunk = true;
            const outputFormat = form.value.output_format.toLowerCase();
            nextStartTime = 0;

            while (true) {
                const { done, value } = await reader.read();
                if (done) {
                    if (outputFormat === 'pcm' || outputFormat === 'wav') {
                        if (pendingBytes.length >= 2) {
                            const float32 = int16ToFloat32(pendingBytes.buffer.slice(0, Math.floor(pendingBytes.length / 2) * 2));
                            playChunk(float32);
                            buffers.push(float32);
                        }
                    }
                    break;
                }

                if (outputFormat === 'pcm' || outputFormat === 'wav') {
                    let currentData = value;
                    if (isFirstChunk && outputFormat === 'wav') {
                        currentData = value.slice(44);
                        isFirstChunk = false;
                    }

                    const combined = new Uint8Array(pendingBytes.length + currentData.length);
                    combined.set(pendingBytes);
                    combined.set(currentData, pendingBytes.length);

                    const completeLength = Math.floor(combined.length / 2) * 2;
                    if (completeLength > 0) {
                        const chunkBuffer = combined.buffer.slice(0, completeLength);
                        const float32 = int16ToFloat32(chunkBuffer);
                        playChunk(float32);
                        buffers.push(float32);
                    }
                    pendingBytes = combined.slice(completeLength);
                } else if (outputFormat === 'mp3') {
                    const chunk = value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength);
                    await decodeAndPlayChunk(chunk);
                    buffers.push(new Uint8Array(chunk));
                } else {
                    buffers.push(new Uint8Array(value));
                }
            }
            finalizeAudio(buffers, form.value.output_format);
        };

        const handleWebSocket = () => {
            return new Promise((resolve, reject) => {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${protocol}//${window.location.host}/ws/v1/tts`);
                const buffers = [];
                nextStartTime = 0;
                let lastDecodePromise = Promise.resolve();

                ws.onopen = () => ws.send(JSON.stringify(form.value));
                ws.onmessage = async (event) => {
                    const data = event.data;
                    const outputFormat = form.value.output_format.toLowerCase();
                    if (typeof data === 'string') {
                        const msg = JSON.parse(data);
                        if (msg.error) reject(new Error(msg.error));
                        if (msg.done) ws.close();
                    } else if (data instanceof Blob) {
                        const arrayBuffer = await data.arrayBuffer();
                        if (outputFormat === 'pcm') {
                            const float32 = int16ToFloat32(arrayBuffer);
                            playChunk(float32);
                            buffers.push(float32);
                        } else if (outputFormat === 'wav') {
                            let currentBuffer = arrayBuffer;
                            const view = new DataView(arrayBuffer);
                            if (arrayBuffer.byteLength >= 44 && view.getUint32(0) === 0x52494646) {
                                currentBuffer = arrayBuffer.slice(44);
                            }
                            const float32 = int16ToFloat32(currentBuffer);
                            playChunk(float32);
                            buffers.push(float32);
                        } else if (outputFormat === 'mp3') {
                            const chunkCopy = arrayBuffer.slice(0);
                            lastDecodePromise = lastDecodePromise.then(() => decodeAndPlayChunk(chunkCopy));
                            buffers.push(new Uint8Array(arrayBuffer));
                        } else {
                            buffers.push(new Uint8Array(arrayBuffer));
                        }
                    }
                };
                ws.onerror = (e) => { error.value = "WebSocket 连接错误"; loading.value = false; reject(e); };
                ws.onclose = () => {
                    loading.value = false;
                    // 等待所有 MP3 解码完成再生成最终文件
                    lastDecodePromise.then(() => {
                        finalizeAudio(buffers, form.value.output_format);
                        resolve();
                    });
                };
            });
        };

        const generateAudio = async () => {
            loading.value = true;
            error.value = null;
            audioUrl.value = null;
            abortController = new AbortController();

            try {
                if (form.value.protocol === 'ws') await handleWebSocket();
                else if (form.value.stream) await handleStreaming();
                else await handleSync();
            } catch (e) {
                if (e.name !== 'AbortError') error.value = e.message;
            } finally {
                if (form.value.protocol !== 'ws') loading.value = false;
                abortController = null;
            }
        };

        const stopGeneration = () => {
            abortController?.abort();
            ws?.close();
            ws = null;
            loading.value = false;
        };

        onMounted(() => {
            fetchHealth();
            fetchVoices();
            setInterval(fetchHealth, 30000);
        });

        return {
            form, health, voices, loading, audioUrl, audioSampleRate, error,
            audioPlayer, statusClass: computed(() => {
                if (health.value.status === 'ok') return 'bg-green-500';
                if (health.value.status === 'error') return 'bg-red-500';
                return 'bg-yellow-500';
            }),
            generateAudio, stopGeneration, promptAudioPreview, sourceAudioPreview,
            uploading, handleFileUpload
        };
    }
}).mount('#app');
