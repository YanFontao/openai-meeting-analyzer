# OpenAI Meeting Analyzer

OpenAI Meeting Analyzer is a robust AI-powered transcription tool with speaker diarization and intelligent content analysis. It is designed for professionals who work with recordings from meetings, interviews, lectures, and other conversational environments.

<p align="center">
  <img src="banner.png" width="800"/>
</p>

---

## 🚀 Value Proposition

Transform audio and video recordings into structured and readable documentation with speaker identification, multi-format export, and automated conversation analysis.

### Who is it for?

Professionals in design, product, marketing, education, legal, and other fields that rely on accurate records of conversations.

### What problem does it solve?

It removes the manual effort of listening, transcribing, and interpreting long recordings by centralizing the entire workflow into a simple command-line tool.

---

## ⚙️ Technologies Used

* **Whisper (OpenAI)** — Automatic speech recognition
* **PyAnnote** — Speaker diarization
* **GPT (optional)** — Conversation understanding and analysis
* **FFmpeg** — Audio extraction and media conversion
* **Python** — CLI-based processing pipeline
* **FPDF** — PDF generation

---

## 📥 Input

* Audio files: `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`, etc.
* Video files: `.mp4`, `.mov`, `.webm`, etc. (audio is automatically extracted)

---

## 📤 Output

By default, the system generates a `.zip` file containing:

* `.txt` — Full transcript with speaker labels
* `.srt` — Subtitle file
* `.pdf` — Formatted readable transcript
* `.json` — Structured transcript with timestamps
* `analysis.txt` — GPT-based conversation interpretation (optional)

Users can select which outputs to generate using flags.

---

## 📊 Performance Estimation

| File Type           | Audio Length | File Size | Whisper Model | Estimated GPU Time |
| ------------------- | ------------ | --------- | ------------- | ------------------ |
| .mp3 (clean)        | 30 min       | 25 MB     | tiny          | ~1m30s             |
| .mp4 (video)        | 30 min       | 500 MB    | base          | ~2m30s             |
| .wav (uncompressed) | 30 min       | 300 MB    | tiny          | ~2m00s             |
| .mp3 (noisy audio)  | 30 min       | 25 MB     | medium        | ~3m30s             |

---

## 💡 Usage Example

```bash
python transcrevegpt.py "audio1.mp4" "audio2.mp3" pt tiny --no-pdf --no-json --no-gpt
```

Process multiple files in a single run:

```bash
python transcrevegpt.py meeting1.mp3 meeting2.wav lecture.mp4
```

Disable AI analysis:

```bash
python transcrevegpt.py recording.mp3 --no-gpt
```

---

## 🔧 Available Flags

* `--no-pdf` — Disable PDF export
* `--no-json` — Disable JSON export
* `--no-zip` — Disable ZIP generation
* `--no-gpt` — Disable AI-powered analysis

---

## 📈 Scalability Vision

This project was designed with future SaaS evolution in mind.

Potential extensions include:

* Web-based upload interface
* Cloud processing pipeline
* User authentication system
* Credit-based billing model
* Dashboard with processing history
* API access for integrations
* Enterprise deployment support

---

## 🎯 Use Cases

* Meeting documentation and minutes generation
* UX and product research interviews
* Academic lectures and study material creation
* Podcast and video content repurposing
* Legal and compliance transcription workflows

---

## 📌 Key Differentiators

Unlike simple transcription tools, this system combines:

* Speaker diarization
* Structured multi-format export
* AI-powered conversation understanding
* Batch processing capabilities
* Developer-friendly CLI workflow

---

## 🧠 Educational & Professional Value

This project demonstrates the integration of multiple AI and systems components into a unified pipeline, including speech recognition, natural language processing, media processing, and automated document generation.

It showcases practical experience in building production-ready AI tools with scalability in mind.

## Author

Yan Matheus Gonçalves Fontão

Product Designer • AI Engineer • Full Stack Developer • Systems Builder
