# TeachFlow RAG 🎓

An AI teaching assistant built on **Retrieval-Augmented Generation (RAG)**. It turns recorded course videos into a searchable knowledge base, so students can ask questions in plain language and get answers that point to the exact lecture and timestamp where the topic is covered.

The current knowledge base covers an introductory **C++ programming course**: variables, input/output, namespaces, preprocessor directives, flowcharts, pattern printing, and more.

---

## ✨ How It Works

```
Videos ──► Audio (MP3) ──► Transcripts (JSON) ──► Embeddings ──► Query + LLM ──► Answer
```

1. **Extract audio**: lecture videos are converted to MP3.
2. **Transcribe**: OpenAI Whisper turns the audio into timestamped text chunks.
3. **Embed**: each chunk is converted into a vector embedding and saved to `embeddings.joblib`.
4. **Retrieve**: the user's question is embedded and matched against all chunks using cosine similarity.
5. **Generate**: the best-matching chunks are passed to an LLM, which writes an answer and points to the relevant video and timestamp.

---

## 📁 Project Structure

| File / Folder | Purpose |
|---|---|
| `video_to_mp3.py` | Converts lecture videos into MP3 audio files |
| `mp3_to_json.py` | Transcribes audio with Whisper into timestamped JSON chunks |
| `preprocess_json.py` | Generates embeddings for every chunk and saves them |
| `process_incoming.py` | Takes a user question, retrieves relevant chunks, and queries the LLM |
| `Jsons/` | Transcribed lecture chunks, one file per video |
| `embeddings.joblib` | Precomputed embedding store (DataFrame of chunks + vectors) |
| `prompt.txt` | Last prompt sent to the LLM (for debugging) |
| `response.txt` | Last response from the LLM |

---

## 🛠️ Tech Stack

- **Python**
- **FFmpeg**: video → audio conversion
- **OpenAI Whisper**: speech-to-text transcription
- **Embedding model**: ✏️ e.g. `bge-m3` via Ollama
- **LLM**: ✏️ e.g. `llama3.2` via Ollama
- **pandas, NumPy, scikit-learn, joblib**: data handling and similarity search

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- [FFmpeg](https://ffmpeg.org/download.html) installed and added to PATH
- [Ollama](https://ollama.com/) running locally ✏️

### Installation
```bash
git clone https://github.com/Himanshu15004/TeachFlow_RAG.git
cd TeachFlow_RAG
pip install openai-whisper pandas numpy scikit-learn joblib requests
```

### Usage

**To ask questions using the existing knowledge base:**
```bash
python process_incoming.py
```

**To build a knowledge base from your own videos:**
```bash
# 1. Put your lecture videos in a Videos/ folder
python video_to_mp3.py      # Videos → Audios/
python mp3_to_json.py       # Audios → Jsons/
python preprocess_json.py   # Jsons → embeddings.joblib
python process_incoming.py  # Start asking questions
```

---

## 💬 Example

> **Q:** Where is the concept of namespaces taught?
>
> **A:** Namespaces are explained in the video *"What is Namespace?"* around the 2:15 mark, where the instructor covers why `using namespace std;` is used...

---

## 🔮 Future Improvements

- Web interface (Streamlit / Flask)
- Vector database (FAISS / ChromaDB) for faster retrieval
- Support for multiple courses and languages
- Clickable timestamp links to jump straight into videos

---

## 👥 Contributors

- [Himanshu](https://github.com/Himanshu15004)
