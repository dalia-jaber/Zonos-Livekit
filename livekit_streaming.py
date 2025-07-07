import asyncio
import os
import io
import openai
import torchaudio
import torch
from livekit import rtc
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device


async def transcribe_audio(audio_bytes: bytes, sample_rate: int) -> str:
    """Send raw PCM audio bytes to OpenAI's Whisper API and return the transcript."""
    wav = io.BytesIO()
    tensor = torch.frombuffer(audio_bytes, dtype=torch.int16).float().unsqueeze(0) / 32768.0
    torchaudio.save(wav, tensor, sample_rate=sample_rate, format="wav")
    wav.seek(0)
    transcript = openai.audio.transcriptions.create(model="whisper-1", file=wav)
    return transcript.text


def generate_reply(prompt: str):
    """Stream tokens from OpenAI Chat API."""
    messages = [{"role": "user", "content": prompt}]
    return openai.chat.completions.create(model="gpt-4o", messages=messages, stream=True)


class ZonosStreamer:
    """Generate audio with Zonos and stream frames to a LiveKit track."""

    def __init__(self, room: rtc.Room):
        self.room = room
        self.source = rtc.AudioSource()
        self.track = rtc.LocalAudioTrack.create_audio_track("zonos", self.source)
        self.model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
        self.model.requires_grad_(False).eval()

    async def publish(self):
        await self.room.local_participant.publish_track(self.track)

    def tts_stream(self, text: str, speaker: torch.Tensor | None = None, language: str = "en-us"):
        cond = make_cond_dict(text=text, speaker=speaker, language=language)
        conditioning = self.model.prepare_conditioning(cond)

        def on_frame(frame: torch.Tensor, _step: int, _max: int) -> bool:
            audio = self.model.autoencoder.decode(frame).cpu()
            samples = audio.squeeze(0).numpy()
            aframe = rtc.AudioFrame(samples=samples, sample_rate=self.model.autoencoder.sampling_rate)
            self.source.capture_frame(aframe)
            return True

        self.model.generate(conditioning, callback=on_frame)


async def handle_track(room: rtc.Room, track: rtc.RemoteAudioTrack):
    """Handle incoming audio from LiveKit, run STT -> LLM -> TTS."""
    buf = bytearray()
    async for frame in track:
        buf.extend(frame.samples)
        if len(buf) > track.sample_rate * 2 * 5:  # roughly 5 seconds buffer
            break

    text = await asyncio.to_thread(transcribe_audio, bytes(buf), track.sample_rate)
    print("Transcribed:", text)

    streamer = ZonosStreamer(room)
    await streamer.publish()

    reply_stream = generate_reply(text)
    reply_text = ""
    for chunk in reply_stream:
        token = chunk.choices[0].delta.content
        if token:
            reply_text += token
    print("LLM reply:", reply_text)
    streamer.tts_stream(reply_text)


async def main():
    openai.api_key = os.environ["OPENAI_API_KEY"]
    url = os.environ["LIVEKIT_URL"]
    token = os.environ["LIVEKIT_TOKEN"]

    room = rtc.Room()
    await room.connect(url, token, auto_subscribe=True)
    room.on(rtc.RoomEvent.TRACK_SUBSCRIBED, lambda track, pub, p: asyncio.create_task(handle_track(room, track)))
    await room.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
