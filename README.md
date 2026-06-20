# Local Real-Time Translator

Real-time speech translation using **only local services**:

- **Speech-to-text**: faster-whisper on your Mac
- **Translation**: LM Studio local server
- **UI**: Python Tkinter

No Cloudflare, ngrok, or external LLM APIs.

## Prerequisites

1. **LM Studio** running with a loaded model and local server on port `1234`
2. **Microphone permission** for Terminal or Python (System Settings → Privacy & Security → Microphone)
3. Python 3.10+

## Setup

```bash
cd ~/Projects/realtime-translator
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional environment variables:

```bash
export LM_STUDIO_URL=http://localhost:1234
export LM_STUDIO_MODEL=google/gemma-4-e4b
export WHISPER_MODEL=base
```

## Run

```bash
source .venv/bin/activate
python app.py
```

## How it works

1. Choose **Source Language** and **Target Language** from the dropdowns.
2. Speak into the microphone. The app listens continuously.
3. When you stop speaking for **2 seconds**, the captured sentence is:
   - transcribed locally with Whisper
   - translated locally through LM Studio
4. Original and translated text appear together in the text area.
5. Click **Continue** to clear the text and start listening again.

## UI

- Resizable window
- Source / target language selectors
- Combined display of original + translation
- Continue button to reset and resume listening

## Troubleshooting

| Issue | Fix |
|---|---|
| LM Studio error | Start LM Studio server and load a model |
| No transcription | Grant mic access; speak louder/closer |
| Slow first run | Whisper downloads the model on first use |
| Wrong language detected | Match the Source Language dropdown to what you speak |
