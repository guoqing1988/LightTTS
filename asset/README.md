# Asset 目录说明

此目录用于存放预加载音色的参考音频文件。

## 使用方法

1. 将音色参考音频文件 (WAV 格式) 放置在此目录
2. 在 `app/config.py` 的 `VOICE_CONFIGS` 中配置音色信息
3. 重启服务,音色将自动加载

## 默认音色

默认需要一个名为 `zero_shot_prompt.wav` 的文件作为默认音色。

您可以:
- 从 CosyVoice 官方仓库的 `asset/` 目录复制
- 使用自己录制的 3-10 秒音频 (建议采样率 16kHz 或 24kHz)

## 音频要求

- 格式: WAV
- 时长: 3-10 秒
- 采样率: 16000Hz 或 24000Hz
- 声道: 单声道或立体声均可
- 内容: 清晰的人声,无背景噪音

## 示例配置

```python
VOICE_CONFIGS = [
    {
        "id": "default",
        "file": "zero_shot_prompt.wav",
        "prompt_text": "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
        "description": "默认女声"
    }
]
```
