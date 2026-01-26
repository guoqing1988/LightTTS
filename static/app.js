const { createApp, ref, onMounted, computed, nextTick } = Vue;

createApp({
    setup() {
        const form = ref({
            text: "2026年，我们正站在智能文明的潮头。浪潮汹涌，唯有勇者与智者能立于潮头。作为时代的“造浪者”，我们手中握有的，正是一副足以定义未来的“王炸”。这副王炸，就是“科艺商潮”。这不是一句简单的口号，而是我们基因里的超代码，是我们从“记录时代”迈向“定义时代”的终极武器。2026年，我们的任务只有一个：打好这副王炸，掀起属于超媒体的“Meta Wave”！",
            mode: "sft",
            voice_id: "", // Renamed from speaker to voice_id for consistency
            prompt_text: "",
            prompt_wav_file: null, // Store File object
            instruct_text: "",
            source_wav_file: null, // Store File object
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

        const handleFileSelect = (event, type) => {
            const file = event.target.files[0];
            if (!file) return;

            if (type === 'prompt') {
                form.value.prompt_wav_file = file;
                promptAudioPreview.value = URL.createObjectURL(file);
                // Reset prompt text if you want, or keep it
            } else if (type === 'source') {
                form.value.source_wav_file = file;
                sourceAudioPreview.value = URL.createObjectURL(file);
            }
        };

        const fetchHealth = async () => {
            try {
                const res = await fetch('/health');
                health.value = await res.json();
            } catch (e) { health.value = { status: 'error' }; }
        };

        const fetchVoices = async () => {
            try {
                const res = await fetch('/tone_list');
                voices.value = await res.json();
            } catch (e) { console.error("Failed to fetch voices", e); }
        };

        const handleSync = async () => {
            const formData = new FormData();
            formData.append('tts_text', form.value.text);
            formData.append('speed', form.value.speed);
            formData.append('stream', 'false');

            if (form.value.mode === 'sft') {
                if (!form.value.voice_id) throw new Error("请选择音色");
                formData.append('voice_id', form.value.voice_id);
            } else if (form.value.mode === 'zero_shot') {
                if (!form.value.prompt_wav_file || !form.value.prompt_text) throw new Error("请上传参考音频并输入Prompt文本");
                formData.append('prompt_wav', form.value.prompt_wav_file);
                formData.append('prompt_text', form.value.prompt_text);
            }

            const res = await fetch('/inference_zero_shot', {
                method: 'POST',
                body: formData,
                signal: abortController?.signal
            });

            if (!res.ok) throw new Error((await res.json()).detail || (await res.json()).message || '请求失败');

            const contentDispostion = res.headers.get('content-disposition');
            // Check if it's JSON error (sometimes APIs return 200 OK but body is error json? No, standard is status code)
            // But let's assume binary response. 
            // The original code expected JSON with base64 audio. The new endpoint returns StreamingResponse (binary).

            // Wait, the python code endpoint returns StreamingResponse(generate_data(ans_objs)).
            // generate_data yields tts_audio bytes.

            const arrayBuffer = await res.arrayBuffer();
            // Since it's a straight binary stream of PCM data (from generate_data)
            // We need to know if it's PCM or WAV?
            // "tts_audio = (i["tts_speech"] * (2 ** 15)).astype(np.int16).tobytes()" 
            // It looks like raw PCM (int16).

            // However, the frontend display expects specific format handling.
            // Let's assume default PCM (int16, 24000Hz usually for CosyVoice).
            // We'll wrap it.

            audioSampleRate.value = 22050; // Or 24000? CosyVoice is usually 22050 or 24000. 
            // Let's rely on backend if possible, but HTTP streaming doesn't give metadata easily in raw body.
            // Assumption: 22050 for now or stick to 24000.
            audioSampleRate.value = 24000; // Updated assumption

            const float32 = int16ToFloat32(arrayBuffer);
            finalizeAudio([float32], 'pcm');
        };

        const handleStreaming = async () => {
            const formData = new FormData();
            formData.append('tts_text', form.value.text);
            formData.append('speed', form.value.speed);
            formData.append('stream', 'true');

            if (form.value.mode === 'sft') {
                if (!form.value.voice_id) throw new Error("请选择音色");
                formData.append('voice_id', form.value.voice_id);
            } else if (form.value.mode === 'zero_shot') {
                if (!form.value.prompt_wav_file || !form.value.prompt_text) throw new Error("请上传参考音频并输入Prompt文本");
                formData.append('prompt_wav', form.value.prompt_wav_file);
                formData.append('prompt_text', form.value.prompt_text);
            }

            const res = await fetch('/inference_zero_shot', {
                method: 'POST',
                body: formData,
                signal: abortController?.signal
            });

            if (!res.ok) throw new Error((await res.json()).detail || (await res.json()).message || '流式请求失败');

            // The new backend doesn't seem to set 'X-Sample-Rate'.
            audioSampleRate.value = 24000;
            getAudioContext(24000);

            const reader = res.body.getReader();
            const buffers = [];
            let pendingBytes = new Uint8Array(0);

            while (true) {
                const { done, value } = await reader.read();
                if (done) {
                    if (pendingBytes.length >= 2) {
                        const float32 = int16ToFloat32(pendingBytes.buffer.slice(0, Math.floor(pendingBytes.length / 2) * 2));
                        playChunk(float32);
                        buffers.push(float32);
                    }
                    break;
                }

                const combined = new Uint8Array(pendingBytes.length + value.length);
                combined.set(pendingBytes);
                combined.set(value, pendingBytes.length);

                const completeLength = Math.floor(combined.length / 2) * 2;
                if (completeLength > 0) {
                    const chunkBuffer = combined.buffer.slice(0, completeLength);
                    const float32 = int16ToFloat32(chunkBuffer);
                    playChunk(float32);
                    buffers.push(float32);
                }
                pendingBytes = combined.slice(completeLength);
            }
            finalizeAudio(buffers, 'pcm');
        };

        const handleWebSocket = () => {
            return new Promise((resolve, reject) => {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                // New endpoint
                ws = new WebSocket(`${protocol}//${window.location.host}/inference_zero_shot_bistream`);
                const buffers = [];
                nextStartTime = 0;

                audioSampleRate.value = 24000; // Assumption
                getAudioContext(24000);

                ws.onopen = async () => {
                    // Initial JSON
                    const initData = {
                        tts_text: form.value.text,
                        prompt_text: form.value.prompt_text,
                        tts_model_name: "default"
                    };

                    if (form.value.mode === 'sft') {
                        initData.voice_id = form.value.voice_id;
                    }

                    ws.send(JSON.stringify(initData));

                    // If Zero Shot, send audio bytes next
                    if (form.value.mode === 'zero_shot' && form.value.prompt_wav_file) {
                        const arrayBuffer = await form.value.prompt_wav_file.arrayBuffer();
                        ws.send(arrayBuffer);
                    }
                };

                ws.onmessage = async (event) => {
                    const data = event.data;
                    if (typeof data === 'string') {
                        try {
                            const msg = JSON.parse(data);
                            if (msg.error) {
                                reject(new Error(msg.error));
                                ws.close();
                            }
                        } catch (e) { }
                    } else if (data instanceof Blob) {
                        const arrayBuffer = await data.arrayBuffer();
                        const float32 = int16ToFloat32(arrayBuffer);
                        playChunk(float32);
                        buffers.push(float32);
                    }
                };
                ws.onerror = (e) => { error.value = "WebSocket 连接错误"; loading.value = false; reject(e); };
                ws.onclose = () => {
                    loading.value = false;
                    finalizeAudio(buffers, 'pcm');
                    resolve();
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
            uploading, handleFileSelect
        };
    }
}).mount('#app');
