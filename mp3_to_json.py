import whisper
import json
import os

model = whisper.load_model("large-v2")


audio_folder = r"D:\audio_test"

output_folder = r"D:\RAG_Teaching_Assistant\Jsons"
os.makedirs(output_folder, exist_ok=True)

audios = os.listdir(audio_folder)

for audio in audios:

    
    if not audio.lower().endswith((".mp3")):
        continue

    print(f"\n Processing: {audio}")

    
    audio_path = os.path.join(audio_folder, audio)

    
    title = audio.rsplit(".", 1)[0]

    
    number = ""

    
    result = model.transcribe(
        audio=audio_path,
        language="hi",        
        task="translate",
        word_timestamps=False
    )

    
    chunks = []
    for segment in result["segments"]:
        chunks.append({
            "number": number,
            "title": title,
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"]
        })

    
    final_json = {
        "file": audio,
        "title": title,
        "number": number,
        "text": result["text"],
        "chunks": chunks
    }

    
    save_path = os.path.join(output_folder, f"{title}.json")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(final_json, f, ensure_ascii=False, indent=4)

    print(f"Saved: {save_path}")
