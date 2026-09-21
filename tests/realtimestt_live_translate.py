import argparse
import json
import os
import queue
import threading
import urllib.request


class LlamaCppTranslator:
    def __init__(
        self, base_url, model, on_result, target_language="English", timeout=30
    ):
        self.url = base_url.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.on_result = on_result
        self.target_language = target_language
        self.timeout = timeout
        self.requests = queue.PriorityQueue()
        self.lock = threading.Lock()
        self.latest_preview = 0
        self.sequence = 0
        self.cache = {}
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def submit(self, text, final=False, context=None):
        text = text.strip()
        if not text:
            return
        with self.lock:
            self.sequence += 1
            request_id = self.sequence
            self.latest_preview = request_id
        priority = 0 if final else 1
        self.requests.put((priority, request_id, text, final, context))

    def cancel_previews(self):
        with self.lock:
            self.sequence += 1
            self.latest_preview = self.sequence

    def close(self):
        self.cancel_previews()
        self.requests.put((-1, 0, None, False, None))

    def _run(self):
        while True:
            _, request_id, source_text, final, context = self.requests.get()
            if source_text is None:
                return
            with self.lock:
                is_latest = request_id == self.latest_preview
            if not final and not is_latest:
                continue
            try:
                translated_text = self.cache.get(source_text)
                if translated_text is None:
                    translated_text = self._translate(source_text)
                    self.cache[source_text] = translated_text
                    if len(self.cache) > 128:
                        self.cache.pop(next(iter(self.cache)))
                error = None
            except Exception as exc:
                translated_text = ""
                error = str(exc)
            with self.lock:
                is_latest = request_id == self.latest_preview
            if final or is_latest:
                self.on_result(
                    source_text,
                    translated_text,
                    error,
                    final,
                    context,
                )

    def _translate(self, text):
        target_language = getattr(self, "target_language", "English")
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"Translate the text into '{target_language}' language. "
                        "Preserve meaning, names, numbers, and tone. "
                        "Output only the translation, and nothing else."
                    ),
                },
                {"role": "user", "content": text},
            ],
            "temperature": 0,
            "max_tokens": 512,
            "stop": ["\n"],
            "chat_template_kwargs": {"enable_thinking": False},
        }
        headers = {"Content-Type": "application/json"}
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode(),
            headers=headers,
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        return result["choices"][0]["message"]["content"].strip().strip('"')


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe audio and translate stable text with llama.cpp."
    )
    parser.add_argument("--model", default="large-v2")
    parser.add_argument("--rt-model", default="small")
    parser.add_argument("--language", default="ja")
    parser.add_argument("--target-language", default="English")
    parser.add_argument("--input-device-index", type=int)
    parser.add_argument("--pulse-source")
    parser.add_argument("--llama-url", default="http://127.0.0.1:8080")
    parser.add_argument("--llama-model", default="Gemma-4-E4B")
    parser.add_argument("--history-size", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.pulse_source:
        os.environ["PULSE_SOURCE"] = args.pulse_source

    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text
    from RealtimeSTT import AudioToTextRecorder

    console = Console()
    live = Live(console=console, refresh_per_second=10, screen=False)
    state_lock = threading.RLock()
    history = []
    next_history_id = 0
    current_raw = ""
    current_stable = ""
    current_translation = ""
    current_error = ""

    def format_entry(entry):
        content = Text()
        content += Text(entry["source"], style="yellow")
        content += Text("\n")
        if entry["error"]:
            content += Text(entry["error"], style="red")
        else:
            content += Text(
                entry["translation"] or "Translating...",
                style="cyan",
            )
        return content

    def render():
        with state_lock:
            content = Text()
            visible_history = history[-args.history_size:]
            for index, entry in enumerate(visible_history):
                if index:
                    content += Text("\n\n")
                content += format_entry(entry)

            live_source = current_stable or current_raw
            if live_source:
                if visible_history:
                    content += Text("\n\n")
                content += Text(live_source, style="bold yellow")
                content += Text("\n")
                if current_error:
                    content += Text(current_error, style="red")
                else:
                    content += Text(
                        current_translation or "Waiting for stable text...",
                        style="cyan",
                    )

            if not content.plain:
                content += Text("Waiting for audio...", style="cyan bold")
            live.update(
                Panel(
                    content,
                    title="[bold green]Live Translation[/bold green]",
                    border_style="bold green",
                )
            )

    def translation_result(source, translated, error, final, context):
        nonlocal current_translation, current_error
        completed_entry = None
        with state_lock:
            if final:
                for index, entry in enumerate(history):
                    if entry["id"] == context:
                        entry["translation"] = translated
                        entry["error"] = error or ""
                        completed_entry = history.pop(index)
                        break
            else:
                current_translation = translated
                current_error = error or ""
        render()
        if completed_entry is not None:
            live.console.print(
                Panel(
                    format_entry(completed_entry),
                    title="[bold green]Live Translation[/bold green]",
                    border_style="green",
                )
            )

    translator = LlamaCppTranslator(
        args.llama_url,
        args.llama_model,
        translation_result,
        target_language=args.target_language,
    )

    def recording_started():
        nonlocal current_raw, current_stable, current_translation, current_error
        translator.cancel_previews()
        with state_lock:
            current_raw = ""
            current_stable = ""
            current_translation = ""
            current_error = ""
        render()

    def realtime_update(text):
        nonlocal current_raw
        with state_lock:
            current_raw = text.strip()
        render()

    def stabilization_update(event):
        nonlocal current_stable
        display_text = event.display_text.strip()
        if display_text:
            with state_lock:
                current_stable = display_text
            render()
        stable_text = event.stable_text.strip()
        if event.has_new_stable_text and stable_text:
            translator.submit(stable_text)

    def final_text(text):
        nonlocal next_history_id, current_raw, current_stable
        text = text.strip()
        if not text:
            return
        with state_lock:
            next_history_id += 1
            history_id = next_history_id
            history.append(
                {
                    "id": history_id,
                    "source": text,
                    "translation": "",
                    "error": "",
                }
            )
            current_raw = ""
            current_stable = ""
        translator.submit(text, final=True, context=history_id)
        render()

    recorder_config = {
        "spinner": False,
        "model": args.model,
        "realtime_model_type": args.rt_model,
        "language": args.language,
        "input_device_index": args.input_device_index,
        "enable_realtime_transcription": True,
        "on_realtime_transcription_update": realtime_update,
        "on_realtime_text_stabilization_update": stabilization_update,
        "on_recording_start": recording_started,
        "realtime_processing_pause": 0.4,
        "realtime_transcription_use_syllable_boundaries": True,
        "realtime_boundary_detector_sensitivity": 0.6,
        "realtime_boundary_followup_delays": 0.5,
        "initial_prompt_realtime": None,
        "silero_sensitivity": 0.05,
        "webrtc_sensitivity": 3,
        "silero_deactivity_detection": True,
        "silero_use_onnx": True,
        "faster_whisper_vad_filter": False,
        "post_speech_silence_duration": 0.7,        # Reduce for fast speech, increase for slow speech
        "min_length_of_recording": 1.1,
        "min_gap_between_recordings": 0,
        "beam_size": 5,
        "beam_size_realtime": 3,
        "no_log_file": True,
        "realtime_punctuation_split_marks": "sentence",
    }

    console.print("System initializing, please wait")
    recorder = None
    live.start()
    try:
        recorder = AudioToTextRecorder(**recorder_config)
        render()
        while True:
            recorder.text(final_text)
    except KeyboardInterrupt:
        pass
    finally:
        translator.close()
        if recorder is not None:
            recorder.shutdown()
        live.stop()
        console.print("Translation stopped.")


if __name__ == "__main__":
    main()
