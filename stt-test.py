
import os
import queue
import threading

import numpy as np
import sounddevice as sd
import torch
from dotenv import load_dotenv
from deepgram import DeepgramClient
from deepgram.core.events import EventType
from silero_vad import load_silero_vad

# Load API key from .env
load_dotenv()

if not os.getenv("DEEPGRAM_API_KEY"):
    raise RuntimeError(
        "DEEPGRAM_API_KEY not found. Check your .env file."
    )

# Audio configuration
SAMPLE_RATE = 16000
FRAME_SIZE = 512
MIC_DEVICE = 1
SILENCE_MS = 700

audio_queue = queue.Queue()
print_lock = threading.Lock()
stop_event = threading.Event()

vad = load_silero_vad()
vad.eval()

client = DeepgramClient()
final_parts = []


def on_message(message):
    channel = getattr(message, "channel", None)
    if channel is None:
        return

    alternatives = getattr(channel, "alternatives", [])
    if not alternatives:
        return

    transcript = getattr(
        alternatives[0], "transcript", ""
    ).strip()

    if not transcript:
        return

    is_final = getattr(message, "is_final", False)
    speech_final = getattr(message, "speech_final", False)

    with print_lock:
        if is_final:
            final_parts.append(transcript)

        if speech_final:
            complete = " ".join(final_parts)
            print(f"\n\nFINAL: {complete}\n", flush=True)
            final_parts.clear()
        else:
            display = " ".join(final_parts)

            if not is_final:
                display = f"{display} {transcript}".strip()

            print(f"\rLIVE: {display}", end="", flush=True)


def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"\nMicrophone status: {status}", flush=True)

    audio_queue.put(indata[:, 0].copy())


def on_error(error):
    with print_lock:
        print(f"\nDeepgram error: {error}", flush=True)
    stop_event.set()


def main():
    speaking = False
    silence_samples = 0
    silence_limit = SAMPLE_RATE * SILENCE_MS // 1000

    print("Connecting to Deepgram Nova-3...")

    with client.listen.v1.connect(
        model="nova-3",
        language="ne",
        encoding="linear16",
        sample_rate=SAMPLE_RATE,
        interim_results=True,
        endpointing=300,
        vad_events=True,
        punctuate=True,
    ) as connection:

        connection.on(
            EventType.OPEN,
            lambda _: print("Connected!", flush=True),
        )
        connection.on(EventType.MESSAGE, on_message)
        connection.on(EventType.ERROR, on_error)

        # IMPORTANT: start receiving in the background.
        listener_thread = threading.Thread(
            target=connection.start_listening,
            daemon=True,
        )
        listener_thread.start()

        print("Speak Nepali now. Press Ctrl+C to stop.\n")

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=FRAME_SIZE,
                device=MIC_DEVICE,
                callback=audio_callback,
            ):
                while not stop_event.is_set():
                    try:
                        frame = audio_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue

                    # Convert float audio to 16-bit PCM.
                    pcm = (
                        np.clip(frame, -1.0, 1.0) * 32767
                    ).astype("<i2")

                    # Send audio continuously for live results.
                    connection.send_media(pcm.tobytes())

                    # Run local Silero VAD.
                    with torch.inference_mode():
                        probability = vad(
                            torch.from_numpy(frame.copy()),
                            SAMPLE_RATE,
                        ).item()

                    if probability >= 0.5:
                        if not speaking:
                            print(
                                "\n[VAD] Speech started",
                                flush=True,
                            )
                        speaking = True
                        silence_samples = 0

                    elif speaking:
                        silence_samples += len(frame)

                        if silence_samples >= silence_limit:
                            print(
                                "\n[VAD] Speech ended",
                                flush=True,
                            )
                            speaking = False
                            silence_samples = 0
                            vad.reset_states()

        finally:
            stop_event.set()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as exc:
        print(f"\nStopped due to error: {exc}")
