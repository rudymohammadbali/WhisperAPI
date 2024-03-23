import atexit
import json
import os
import tempfile

import filetype
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException

from whisper_script import WhisperTranscriber

app = FastAPI()


def cleanup_temp_file(path):
    if os.path.exists(path):
        os.remove(path)


@app.post("/transcribe")
async def transcribe(audio_file: UploadFile = File(...), options: str = Form(...)):
    contents = await audio_file.read()

    max_file_size = 500 * 1024 * 1024
    if len(contents) > max_file_size:
        raise HTTPException(status_code=400, detail="File is too large")

    check_file = filetype.guess(contents)
    if check_file is None or not check_file.mime.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File is not an audio file")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio.write(contents)
            temp_path = temp_audio.name

        try:
            load_options = json.loads(options)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Options are not valid JSON")

        model_size = load_options.get("model_size")
        language = load_options.get("language")
        task = load_options.get("task")
        prompts = {
            "verbose": load_options.get("verbose"),
            "temperature": load_options.get("temperature"),
            "compression_ratio_threshold": load_options.get("compression_ratio_threshold"),
            "logprob_threshold": load_options.get("logprob_threshold"),
            "no_speech_threshold": load_options.get("no_speech_threshold"),
            "condition_on_previous_text": load_options.get("condition_on_previous_text"),
            "initial_prompt": load_options.get("initial_prompt"),
            "word_timestamps": load_options.get("word_timestamps"),
            "prepend_punctuations": load_options.get("prepend_punctuations"),
            "append_punctuations": load_options.get("append_punctuations")
        }

        transcriber = WhisperTranscriber(temp_path, model_size=model_size, language=language, task=task, prompt=prompts)
        transcription = transcriber.transcribe()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        atexit.register(cleanup_temp_file, temp_path)

    return {"filename": audio_file.filename, "transcription": transcription}


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)
