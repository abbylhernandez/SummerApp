"""Microphone capture: live time-aligned envelope + paused-aware WAV recording."""

import wave
import logging
import threading

import numpy as np
import sounddevice as sd

from config import AUDIO_DEVICE, AUDIO_PREFERRED_NAMES, AUDIO_RATE, AUDIO_BLOCK, AUDIO_ENV_CHUNK


class AudioCapture:
    """
    Captures the (camera) microphone via sounddevice. While a trial records, it
    (a) writes full-resolution mono audio to a paused-aware WAV for muxing, and
    (b) builds a time-stamped amplitude envelope (peak per short window) on a
    paused-excluded clock, so the sound graph aligns in time with the EMG graph.
    """

    def __init__(self, device=AUDIO_DEVICE, samplerate=AUDIO_RATE,
                 blocksize=AUDIO_BLOCK, chunk=AUDIO_ENV_CHUNK):
        self.device = device
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.chunk = chunk
        self.stream = None
        self.running = False

        # WAV recording (paused-aware)
        self._wav = None
        self._wav_path = None
        self._wav_recording = False
        self._wav_lock = threading.Lock()

        # time-stamped envelope for plotting (paused-excluded, sample-count clock)
        self.env_t = []            # seconds since trial start (excludes pauses)
        self.env_a = []            # peak amplitude in that window
        self._rec_samples = 0
        self._env_lock = threading.Lock()

    def _resolve_device(self):
        """Use an explicit device if set; otherwise prefer the MacBook microphone."""
        if self.device is not None:
            return self.device

        try:
            devices = sd.query_devices()
        except Exception as e:
            logging.warning("⚠️ Could not list audio devices; using default input: %s", e)
            return None

        for idx, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) <= 0:
                continue
            name = dev.get("name", "")
            if any(preferred.lower() in name.lower() for preferred in AUDIO_PREFERRED_NAMES):
                logging.info("🎤 Using microphone: %s (device %d)", name, idx)
                return idx

        logging.info("🎤 MacBook/built-in microphone not found; using default input.")
        return None

    def start(self):
        try:
            self.stream = sd.InputStream(
                device=self.device,
                samplerate=self.samplerate,
                channels=1,
                blocksize=self.blocksize,
                dtype="float32",
                callback=self._callback,
            )
            self.stream.start()
            self.running = True
            logging.info("🎤 Microphone capture started.")
            return True
        except Exception as e:
            logging.error("❌ Microphone capture failed: %s", e)
            self.stream = None
            return False

    def _callback(self, indata, frames, time_info, status):
        # runs on PortAudio's thread; keep it cheap
        x = indata[:, 0]
        with self._wav_lock:
            rec = self._wav is not None and self._wav_recording
            if rec:
                clipped = np.clip(x, -1.0, 1.0)
                self._wav.writeframes((clipped * 32767.0).astype("<i2").tobytes())
        if not rec:
            return
        # build the envelope (peak per chunk); time = recorded-sample count / rate
        with self._env_lock:
            n = x.shape[0]
            for i in range(0, n, self.chunk):
                seg = x[i:i + self.chunk]
                if seg.size == 0:
                    continue
                self.env_t.append(self._rec_samples / self.samplerate)
                self.env_a.append(float(np.max(np.abs(seg))))
                self._rec_samples += seg.size

    def get_envelope(self):
        with self._env_lock:
            return list(self.env_t), list(self.env_a)

    # ---- WAV recording (Start/Pause/Resume/End mirror the EMG log) ----
    def start_recording(self, path):
        if not self.running:
            return False
        try:
            w = wave.open(str(path), "wb")
            w.setnchannels(1)
            w.setsampwidth(2)            # 16-bit PCM
            w.setframerate(self.samplerate)
        except Exception as e:
            logging.error("❌ Failed to open WAV %s: %s", path, e)
            return False
        with self._env_lock:
            self.env_t = []
            self.env_a = []
            self._rec_samples = 0
        with self._wav_lock:
            self._wav = w
            self._wav_path = str(path)
            self._wav_recording = True
        logging.info("🎙️ Audio recording started: %s", path)
        return True

    def pause_recording(self):
        with self._wav_lock:
            self._wav_recording = False

    def resume_recording(self):
        with self._wav_lock:
            if self._wav is not None:
                self._wav_recording = True

    def stop_recording(self):
        """Finalise the WAV; return its path (or None if nothing was recorded)."""
        with self._wav_lock:
            self._wav_recording = False
            w, path = self._wav, self._wav_path
            self._wav = None
            self._wav_path = None
        if w is not None:
            try:
                w.close()
            except Exception:
                pass
            logging.info("🎙️ Audio recording saved: %s", path)
            return path
        return None

    def stop(self):
        self.stop_recording()
        self.running = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                logging.warning("⚠️ Error closing audio stream: %s", e)
            self.stream = None
        logging.info("🎤 Microphone capture stopped.")
