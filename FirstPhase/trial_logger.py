import sys
import os
import re
import time
import wave
import logging
import threading
import queue
import subprocess
from collections import deque
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

import serial
import cv2
import numpy as np
import platform
import sounddevice as sd
import imageio_ffmpeg
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg


# =============================================================
# Configuration (edit to match your hardware)
# =============================================================
# Set SIMULATE = True to test the GUI WITHOUT any EMG hardware connected.
# It feeds flat (straight-line) values for all three channels.
# Set it back to False once the real EMG sensors are plugged in.
SIMULATE      = True
SIM_RATE_HZ   = 1000          # fake sample rate while simulating
SIM_LEVELS    = (0.5, 1.5, 2.5)  # straight-line value (V) for ch1, ch2, ch3

SERIAL_PORT   = "COM8"        # USB serial port of the EMG microcontroller
BAUD_RATE     = 500000
START_CMD     = b"c"          # byte sent to MCU to start streaming
STOP_CMD      = b"v"          # byte sent to MCU to stop streaming

CAM_INDEX     = None          # None = auto-detect (prefer external); 0 = built-in
CAM_PROBE_MAX = 4             # how many indices to probe when auto-detecting
CAM_WIDTH     = 640
CAM_HEIGHT    = 360
CAM_FPS       = 30

AUDIO_DEVICE  = None          # mic input device (None = prefer MacBook/built-in input)
AUDIO_PREFERRED_NAMES = ("MacBook", "Built-in Microphone", "Built-in Input")
AUDIO_RATE    = 44100         # sample rate (Hz)
AUDIO_BLOCK   = 1024          # samples per audio callback block
AUDIO_DECIM   = 16            # (unused) kept for reference
AUDIO_POINTS  = 2000          # (unused) kept for reference
AUDIO_ENV_CHUNK = 441         # samples per envelope point (~10 ms at 44.1 kHz)
AUDIO_MIN_SPAN = 0.02         # smallest height so silence doesn't over-zoom

VREF          = 3.0           # ADC reference voltage
ADC_RES       = 4095.0        # 12-bit ADC full scale
ADC_MIN       = 0
ADC_MAX       = 4095

PREVIEW_POINTS = 1500         # rolling window length for preview mode
PLOT_UPDATE_MS = 33           # graph redraw interval

SAVE_ROOT      = "trial_logs"  # where session folders are created
PACIFIC_TZ     = ZoneInfo("America/Los_Angeles")  # Pacific time (PST/PDT)


# =============================================================
# Camera capture (threaded, Windows DirectShow) -- from realtime.py
# =============================================================
class CameraCapture:
    def __init__(self, cam_index=CAM_INDEX, width=CAM_WIDTH, height=CAM_HEIGHT, fps=CAM_FPS):
        self.cam_index = cam_index
        self.width = width
        self.height = height
        self.fps = fps

        self.cap = None
        self.running = False
        self.thread = None
        self.frame_queue = queue.Queue(maxsize=1)

        # per-trial recording
        self.recording = False         # True only while a trial is actively logging
        self.video_writer = None
        self.record_path = None
        self._rec_lock = threading.Lock()

        # per-frame timestamps (frame_idx, t_ns) using a paused-time-excluded clock
        self.frame_ts = []
        self._frame_idx = 0
        self._rec_elapsed_ns = 0
        self._rec_last_ns = None

    @staticmethod
    def _backend_for_current_os():
        """Choose the OpenCV camera backend for the current OS."""
        system = platform.system()
        if system == "Windows":
            return getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY)
        if system == "Darwin":
            return getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY)
        return getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)

    def _detect_indices(self, limit=CAM_PROBE_MAX):
        """Return indices that actually deliver frames. Chooses backend per OS."""
        found = []
        backend = self._backend_for_current_os()

        for i in range(limit):
            try:
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        found.append(i)
                cap.release()
            except Exception:
                continue
        return found

    def start(self):
        # resolve the camera index (auto-pick the external one when not set)
        if self.cam_index is None:
            found = self._detect_indices()
            logging.info("📷 Cameras detected at indices: %s", found)
            # built-in laptop cam is usually index 0; external USB cams enumerate
            # after it, so prefer the highest available index.
            self.cam_index = max(found) if found else 0
            logging.info("📷 Using camera index %s", self.cam_index)

        backend = self._backend_for_current_os()

        try:
            self.cap = cv2.VideoCapture(self.cam_index, backend)
        except Exception:
            # fallback to default constructor
            self.cap = cv2.VideoCapture(self.cam_index)
        if not self.cap.isOpened():
            logging.error("❌ Failed to open camera index %s.", self.cam_index)
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logging.info("✅ Camera started at %dx%d", self.width, self.height)
        return True

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            resized = cv2.resize(frame, (self.width, self.height))   # BGR

            # write to the trial video only while actively recording (Pause skips)
            with self._rec_lock:
                if self.recording and self.video_writer is not None:
                    now = time.perf_counter_ns()
                    dt = 0 if self._rec_last_ns is None else (now - self._rec_last_ns)
                    self._rec_elapsed_ns += dt
                    self._rec_last_ns = now
                    self.frame_ts.append((self._frame_idx, self._rec_elapsed_ns))
                    self._frame_idx += 1
                    self.video_writer.write(resized)

            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            if not self.frame_queue.full():
                self.frame_queue.put(rgb)

    def get_latest_frame(self):
        if not self.frame_queue.empty():
            return self.frame_queue.get()
        return None

    # ---- per-trial recording (Start/Pause/Resume/End mirror the EMG log) ----
    def start_recording(self, path):
        """Open an AVI writer and begin recording immediately."""
        if self.cap is None or not self.running:
            logging.warning("⚠️ Camera not running; cannot record %s", path)
            return False
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(str(path), fourcc, self.fps, (self.width, self.height))
        if not writer.isOpened():
            logging.error("❌ Failed to open video writer: %s", path)
            return False
        with self._rec_lock:
            self.video_writer = writer
            self.record_path = str(path)
            self.recording = True
            self.frame_ts = []
            self._frame_idx = 0
            self._rec_elapsed_ns = 0
            self._rec_last_ns = None
        logging.info("🎥 Recording started: %s", path)
        return True

    def pause_recording(self):
        with self._rec_lock:
            self.recording = False
            self._rec_last_ns = None       # exclude the paused gap from frame t_ns

    def resume_recording(self):
        with self._rec_lock:
            if self.video_writer is not None:
                self.recording = True

    def get_frame_timestamps(self):
        with self._rec_lock:
            return list(self.frame_ts)

    def stop_recording(self):
        """Stop and finalise the current trial video (if any)."""
        with self._rec_lock:
            self.recording = False
            writer, path = self.video_writer, self.record_path
            self.video_writer = None
            self.record_path = None
        if writer is not None:
            writer.release()
            logging.info("🎞️ Recording saved: %s", path)

    def stop(self):
        self.stop_recording()
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        logging.info("📷 Camera stopped.")


# =============================================================
# Microphone capture (live waveform, not recorded)
# =============================================================
class AudioCapture:
    """
    Captures the microphone via sounddevice. While a trial records, it
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
            device = self._resolve_device()
            self.stream = sd.InputStream(
                device=device,
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


# =============================================================
# Serial EMG handler (flexible 3+ channel parser)
# =============================================================
class SerialEMGHandler:
    """
    Reads comma-separated lines from the MCU and converts the first three
    integer fields to voltages. Lines may contain extra trailing fields
    (e.g. "ch1,ch2,ch3,pred,button") -- only the first three are used.

    callback(ch1_v, ch2_v, ch3_v) is fired on a background thread for every
    valid sample.
    """

    def __init__(self, port=SERIAL_PORT, baudrate=BAUD_RATE):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.thread = None
        self.running = False
        self.callback = None

    def set_callback(self, func):
        self.callback = func

    def start(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            self.ser.flushInput()
            self.ser.write(START_CMD)
            logging.info("✅ Opened %s @ %d, sent %r to start streaming.",
                         self.port, self.baudrate, START_CMD)
        except serial.SerialException as e:
            logging.error("❌ Failed to open serial port %s: %s", self.port, e)
            self.ser = None
            return False

        self.running = True
        self.thread = threading.Thread(target=self.read_loop, daemon=True)
        self.thread.start()
        return True

    def read_loop(self):
        while self.running and self.ser is not None:
            try:
                raw = self.ser.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line or line.count(",") < 2:
                    continue

                parts = [p.strip() for p in line.split(",")]
                try:
                    raws = [int(parts[i]) for i in range(3)]
                except ValueError:
                    continue

                # Range sanity check (skip corrupted / merged lines)
                if any(v < ADC_MIN or v > ADC_MAX for v in raws):
                    logging.debug("Skipping out-of-range line: %r", line)
                    continue

                volts = [(v / ADC_RES) * VREF for v in raws]
                if self.callback:
                    self.callback(volts[0], volts[1], volts[2])

            except Exception as e:
                logging.warning("⚠️ Serial read error: %s", e)
                continue

    def stop(self):
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(STOP_CMD)
                logging.info("▶️ Sent %r to stop streaming.", STOP_CMD)
            except Exception as e:
                logging.warning("⚠️ Failed to send stop command: %s", e)
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
                logging.info("🔌 Serial port closed.")
            except Exception as e:
                logging.warning("⚠️ Error closing serial port: %s", e)


# =============================================================
# Simulated EMG source (no hardware) -- same interface as SerialEMGHandler
# =============================================================
class SimulatedEMGHandler:
    """
    Drop-in replacement for SerialEMGHandler used for testing without an EMG
    device. Fires callback(ch1, ch2, ch3) at SIM_RATE_HZ with flat values, so
    each channel draws a straight horizontal line.
    """

    def __init__(self, levels=SIM_LEVELS, rate_hz=SIM_RATE_HZ):
        self.levels = levels
        self.period = 1.0 / float(rate_hz)
        self.callback = None
        self.running = False
        self.thread = None

    def set_callback(self, func):
        self.callback = func

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        logging.info("🧪 Simulated EMG source started (flat %.2f/%.2f/%.2f V).",
                     *self.levels)
        return True

    def _loop(self):
        c1, c2, c3 = self.levels
        while self.running:
            if self.callback:
                self.callback(c1, c2, c3)
            time.sleep(self.period)

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        logging.info("🧪 Simulated EMG source stopped.")


# =============================================================
# Main application
# =============================================================
class TrialLoggerApp(QtWidgets.QWidget):

    # State machine values
    IDLE    = "idle"      # trial set up but not started (or fresh / after redo)
    LOGGING = "logging"   # actively recording into the current trial
    PAUSED  = "paused"    # current trial paused
    ENDED   = "ended"     # current trial finished, waiting for "Next Trial"
    PREVIEW = "preview"   # live view only, no logging

    # Background-thread -> GUI-thread bridge
    sample_sig = QtCore.pyqtSignal(float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("EMG Trial Logger + Live Video")

        # ---- per-object folders ----
        # Each object gets ONE folder "[Object]Data-[Date] [Time]" (Pacific
        # time), created the first time that object is logged this run and
        # reused whenever you return to it. Trials are numbered per object and
        # numbering CONTINUES when you come back (it does not reset to 1).
        self.object_dirs = {}      # object name -> Path
        self.object_counts = {}    # object name -> highest trial number so far
        self.group_dir = None      # current object's folder (Path)
        self.current_obj = None    # object locked in for the in-progress trial

        # ---- trial state ----
        self.trial_num = 1         # number the next/active trial will use
        self.state = self.IDLE
        self.prev_state = self.IDLE        # state to restore when leaving preview
        self.elapsed_ns = 0                # logged nanoseconds (excludes paused gaps)
        self.last_perf_ns = None

        # per-trial buffers (written to file when the trial ends)
        self.tns_buf = []                  # nanosecond timestamps
        self.t_buf = []                    # seconds (for the live graph x-axis)
        self.c1_buf, self.c2_buf, self.c3_buf = [], [], []
        # preview rolling buffers
        self.pv_t = deque(maxlen=PREVIEW_POINTS)
        self.pv_c1 = deque(maxlen=PREVIEW_POINTS)
        self.pv_c2 = deque(maxlen=PREVIEW_POINTS)
        self.pv_c3 = deque(maxlen=PREVIEW_POINTS)
        self.pv_idx = 0

        # completed trial labels {(object, num): QPushButton}
        self.trial_labels = {}
        # cached per-trial graph data {(object, num): {...}} for in-place review
        self.trial_data = {}
        self.review_key = None        # which trial is shown in-place (None = live)
        self._review_video_path = None

        # ---- EMG source (real serial, or simulated flat lines for testing) ----
        if SIMULATE:
            self.serial = SimulatedEMGHandler()
        else:
            self.serial = SerialEMGHandler()
        self.serial.set_callback(self._on_sample_threadsafe)

        # camera selection state: allow switching to built-in camera on demand
        self.use_builtin_camera = False
        self.camera = CameraCapture()
        self.cam_timer = QtCore.QTimer(self)
        self.cam_timer.timeout.connect(self._update_camera)

        self.audio = AudioCapture()

        self._build_ui()

        # connect the cross-thread sample signal
        self.sample_sig.connect(self._on_sample)

        # start camera display
        if self.camera.start():
            self.cam_timer.start(int(1000 / CAM_FPS))
        else:
            self.video_label.setText("Camera unavailable")

        # start microphone waveform
        if not self.audio.start():
            self.audio_plot.setTitle("Microphone unavailable")

        # start serial streaming (samples are ignored unless logging/preview)
        if not self.serial.start():
            self.status_lbl.setText("Status: serial port unavailable — check SERIAL_PORT")

        # plot redraw timer
        self.plot_timer = QtCore.QTimer(self)
        self.plot_timer.timeout.connect(self._refresh_plot)
        self.plot_timer.start(PLOT_UPDATE_MS)

        self._update_controls()

    # ----------------------------------------------------------
    # UI
    # ----------------------------------------------------------
    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # status
        self.status_lbl = QtWidgets.QLabel("Status: Idle")
        self.status_lbl.setStyleSheet("font-weight: bold; color: #2c3e50;")
        layout.addWidget(self.status_lbl)

        # ---- object name input (defaults to the previous object; click to switch) ----
        obj_row = QtWidgets.QHBoxLayout()
        obj_row.addWidget(QtWidgets.QLabel("Object:"))
        self.object_edit = QtWidgets.QLineEdit()
        self.object_edit.setPlaceholderText("e.g. Apple — type to switch the object for the next trial")
        self.object_edit.textChanged.connect(self.on_object_changed)
        obj_row.addWidget(self.object_edit, stretch=1)
        layout.addLayout(obj_row)

        # ---- live camera feed (on top) ----
        self.video_label = QtWidgets.QLabel("Camera starting…")
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setFixedSize(CAM_WIDTH, CAM_HEIGHT)   # keep video size fixed
        self.video_label.setStyleSheet("background-color: black; color: white;")
        layout.addWidget(self.video_label, alignment=QtCore.Qt.AlignHCenter)

        # small control to let the user prefer the built-in laptop camera
        cam_row = QtWidgets.QHBoxLayout()
        self.builtin_cam_chk = QtWidgets.QCheckBox("Use built-in camera")
        self.builtin_cam_chk.setChecked(False)
        self.builtin_cam_chk.toggled.connect(self._on_builtin_cam_toggled)
        cam_row.addStretch(1)
        cam_row.addWidget(self.builtin_cam_chk)
        cam_row.addStretch(1)
        layout.addLayout(cam_row)

        pg.setConfigOptions(antialias=False)

        # ---- sound graph (on top of EMG, same size) ----
        self.audio_plot = pg.PlotWidget()
        self.audio_plot.setBackground("w")
        self.audio_plot.setTitle("Microphone")
        self.audio_plot.showGrid(x=True, y=True, alpha=0.2)
        self.audio_plot.setLabel("left", "Amplitude")
        self.audio_plot.getPlotItem().hideAxis("bottom")      # shares EMG's time axis below
        self.audio_plot.setMouseEnabled(x=False, y=False)     # pan via the EMG plot
        self.audio_curve = self.audio_plot.plot([], [], pen=pg.mkPen("#9B59B6", width=1))
        layout.addWidget(self.audio_plot, stretch=1)

        # ---- EMG plot (same size, below the sound graph) ----
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.plot_widget.setLabel("left", "EMG (V)")
        self.plot_widget.setLabel("bottom", "Time (s)")
        self.plot_widget.setYRange(0.0, VREF, padding=0)
        self.plot_widget.addLegend(offset=(10, 10))
        self.curve1 = self.plot_widget.plot([], [], pen=pg.mkPen("#FF6B6B", width=2), name="ch1")
        self.curve2 = self.plot_widget.plot([], [], pen=pg.mkPen("#22B14C", width=2), name="ch2")
        self.curve3 = self.plot_widget.plot([], [], pen=pg.mkPen("#3498DB", width=2), name="ch3")
        layout.addWidget(self.plot_widget, stretch=1)

        # link the time axes so the two graphs pan/zoom together
        self.audio_plot.setXLink(self.plot_widget)

        # ---- completed-trial labels row ----
        comp_box = QtWidgets.QHBoxLayout()
        comp_box.addWidget(QtWidgets.QLabel("Completed:"))
        self.labels_layout = QtWidgets.QHBoxLayout()
        comp_box.addLayout(self.labels_layout)
        comp_box.addStretch(1)
        self.review_video_btn = QtWidgets.QPushButton("▶ Open trial video")
        self.review_video_btn.setVisible(False)
        self.review_video_btn.clicked.connect(self._open_review_video)
        comp_box.addWidget(self.review_video_btn)
        layout.addLayout(comp_box)

        # ---- control buttons ----
        btn_row = QtWidgets.QHBoxLayout()
        self.trial_btn = QtWidgets.QPushButton("Start Trial 1")
        self.end_btn   = QtWidgets.QPushButton("End Trial")
        self.redo_btn  = QtWidgets.QPushButton("Redo Trial")
        self.next_btn  = QtWidgets.QPushButton("Start Trial 2  ▶")
        self.preview_btn = QtWidgets.QPushButton("Preview")
        self.preview_btn.setCheckable(True)

        for b in (self.trial_btn, self.end_btn, self.redo_btn, self.next_btn, self.preview_btn):
            b.setMinimumHeight(34)
            btn_row.addWidget(b)
        layout.addLayout(btn_row)

        self.trial_btn.clicked.connect(self.on_trial_btn)
        self.end_btn.clicked.connect(self.on_end_btn)
        self.redo_btn.clicked.connect(self.on_redo_btn)
        self.next_btn.clicked.connect(self.on_next_btn)
        self.preview_btn.clicked.connect(self.on_preview_btn)

    # ----------------------------------------------------------
    # Button handlers
    # ----------------------------------------------------------
    def on_trial_btn(self):
        self._exit_review()
        if self.state == self.IDLE:
            self._start_logging()
        elif self.state == self.LOGGING:
            self._pause_logging()
        elif self.state == self.PAUSED:
            self._resume_logging()

    def on_end_btn(self):
        if self.state in (self.LOGGING, self.PAUSED):
            self._end_trial()

    def on_redo_btn(self):
        self._exit_review()
        if self.state in (self.LOGGING, self.PAUSED, self.ENDED):
            self._redo_trial()

    def on_next_btn(self):
        self._exit_review()
        if self.state == self.ENDED:
            self._reset_to_idle()   # next trial number is derived from the object

    def on_preview_btn(self):
        self._exit_review()
        if self.preview_btn.isChecked():
            self._enter_preview()
        else:
            self._exit_preview()

    def on_object_changed(self, _text):
        # While idle, switching the object updates which trial number is next.
        if self.state == self.IDLE:
            self.trial_num = self._next_trial_number()
            self._update_controls()

    # ----------------------------------------------------------
    # Trial lifecycle
    # ----------------------------------------------------------
    def _current_object(self):
        """Sanitised object name from the input box (Windows-safe)."""
        obj = self.object_edit.text().strip() or "Object"
        return re.sub(r'[<>:"/\\|?*]', "", obj).strip() or "Object"

    def _next_trial_number(self):
        """Next number for the selected object (continues its own count)."""
        return self.object_counts.get(self._current_object(), 0) + 1

    def _folder_for_object(self, obj):
        """Return this object's folder, creating it (timestamped) on first use."""
        if obj in self.object_dirs:
            return self.object_dirs[obj]

        now = datetime.now(PACIFIC_TZ)
        date_str = now.strftime("%Y-%m-%d")        # e.g. 2026-06-17
        time_str = now.strftime("%I-%M-%S %p")     # 12-hour, e.g. 02-45-30 PM
        base = f"{obj}Data-{date_str} {time_str}"

        path = Path(SAVE_ROOT) / base
        n = 2
        while path.exists():                       # avoid collisions in the same second
            path = Path(SAVE_ROOT) / f"{base} ({n})"
            n += 1
        path.mkdir(parents=True, exist_ok=True)
        self.object_dirs[obj] = path
        logging.info("📂 Created folder for %s: %s", obj, path)
        return path

    def _start_logging(self):
        obj = self._current_object()

        # this object's folder (created once, reused on return)
        self.group_dir = self._folder_for_object(obj)
        self.current_obj = obj
        self.trial_num = self.object_counts.get(obj, 0) + 1

        # follow live data on x; the linked sound graph follows too. Panning
        # later (e.g. when stopped) disengages this and moves both together.
        self.plot_widget.enableAutoRange(axis="x", enable=True)

        # start buffering samples for this trial (file is written on End)
        self.elapsed_ns = 0
        self.last_perf_ns = None
        self.tns_buf = []
        self.t_buf = []
        self.c1_buf, self.c2_buf, self.c3_buf = [], [], []

        # start recording the trial video + audio alongside the .txt
        self.camera.start_recording(self.group_dir / f"video{self.trial_num}.avi")
        self.audio.start_recording(self.group_dir / f"_audio{self.trial_num}.wav")

        self.state = self.LOGGING
        logging.info("▶️ Trial %d started.", self.trial_num)
        self._update_controls()

    def _pause_logging(self):
        self.state = self.PAUSED
        self.last_perf_ns = None       # don't count the paused gap as elapsed time
        self.camera.pause_recording()  # also pause the video log
        self.audio.pause_recording()   # and the audio
        logging.info("⏸️ Trial %d paused.", self.trial_num)
        self._update_controls()

    def _resume_logging(self):
        self.state = self.LOGGING
        self.last_perf_ns = None       # next sample restarts the clock cleanly
        self.camera.resume_recording()
        self.audio.resume_recording()
        logging.info("▶️ Trial %d resumed.", self.trial_num)
        self._update_controls()

    def _write_trial_file(self):
        """Write trial<N>.txt: 't_ns,ch1_V,ch2_V,ch3_V' header + comma rows."""
        path = self.group_dir / f"trial{self.trial_num}.txt"
        with open(path, "w", newline="") as f:
            f.write("t_ns,ch1_V,ch2_V,ch3_V\n")
            for t_ns, a, b, c in zip(self.tns_buf, self.c1_buf, self.c2_buf, self.c3_buf):
                f.write(f"{t_ns},{a:.6f},{b:.6f},{c:.6f}\n")
        logging.info("💾 Wrote %s (%d samples).", path, len(self.tns_buf))
        return path

    def _write_video_timestamps(self):
        """Write video<N>timestamps.csv: 'frame_idx,t_ns' for the trial video."""
        rows = self.camera.get_frame_timestamps()
        if not rows:
            return None
        path = self.group_dir / f"video{self.trial_num}timestamps.csv"
        with open(path, "w", newline="") as f:
            f.write("frame_idx,t_ns\n")
            for idx, t_ns in rows:
                f.write(f"{idx},{t_ns}\n")
        logging.info("💾 Wrote %s (%d frames).", path, len(rows))
        return path

    def _mux_audio_into_video(self, trial_num, wav_path):
        """Merge the trial's WAV into video<N>.avi (audio track), then drop the WAV.
        If audio/ffmpeg is unavailable, the silent video is kept as-is."""
        video = self.group_dir / f"video{trial_num}.avi"
        if not wav_path or not Path(wav_path).exists() or not video.exists():
            if wav_path and Path(wav_path).exists():
                os.remove(wav_path)
            return
        out = self.group_dir / f"_muxed{trial_num}.avi"
        cmd = [
            imageio_ffmpeg.get_ffmpeg_exe(), "-y",
            "-i", str(video), "-i", str(wav_path),
            "-c:v", "copy", "-c:a", "pcm_s16le", "-shortest", str(out),
        ]
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # no console flash on Windows
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, creationflags=flags)
            os.replace(out, video)
            logging.info("🔊 Merged audio into %s", video)
        except Exception as e:
            logging.warning("⚠️ Audio mux failed (%s); video kept silent.", e)
            if out.exists():
                os.remove(out)
        finally:
            if Path(wav_path).exists():
                os.remove(wav_path)

    def _end_trial(self):
        self.camera.stop_recording()           # finalise video<N>.avi
        wav_path = self.audio.stop_recording()  # finalise the temp WAV
        self._write_trial_file()
        self._write_video_timestamps()
        self._mux_audio_into_video(self.trial_num, wav_path)
        self.state = self.ENDED
        self.object_counts[self.current_obj] = self.trial_num

        key = (self.current_obj, self.trial_num)

        # cache this trial's graphs so the green box can reopen them instantly
        et, ea = self.audio.get_envelope()
        self.trial_data[key] = {
            "title": f"{self.current_obj} Trial {self.trial_num}",
            "emg": (list(self.t_buf), list(self.c1_buf), list(self.c2_buf), list(self.c3_buf)),
            "audio": (et, ea),
            "video": self.group_dir / f"video{self.trial_num}.avi",
        }

        # add a clickable label/button for this completed trial
        btn = QtWidgets.QPushButton(f"{self.current_obj} Trial {self.trial_num} ✓")
        btn.setCursor(QtCore.Qt.PointingHandCursor)
        btn.setToolTip("Click to view this trial's graphs")
        btn.setStyleSheet(
            "QPushButton { border: 1px solid #27ae60; border-radius: 6px;"
            " padding: 2px 8px; color: #27ae60; font-weight: bold; background: white; }"
            "QPushButton:hover { background: #eafaf0; }"
        )
        btn.clicked.connect(lambda _=False, k=key: self._open_review(k))
        self.labels_layout.addWidget(btn)
        self.trial_labels[key] = btn

        logging.info("⏹️ %s Trial %d ended (%d samples).",
                     self.current_obj, self.trial_num, len(self.t_buf))
        self._update_controls()

    def _open_review(self, key):
        # clicking the active trial again returns to the live view
        if self.review_key == key:
            self._exit_review()
        else:
            self._enter_review(key)

    def _enter_review(self, key):
        data = self.trial_data.get(key)
        if not data:
            return
        self.review_key = key

        # draw the cached graphs in place of the live ones
        t, c1, c2, c3 = data["emg"]
        self.curve1.setData(t, c1)
        self.curve2.setData(t, c2)
        self.curve3.setData(t, c3)

        et, ea = data["audio"]
        if et:
            self.audio_curve.setData(et, ea)
            span = max(max(ea) * 1.2, AUDIO_MIN_SPAN)
            self.audio_plot.setYRange(0.0, span, padding=0)
        else:
            self.audio_curve.setData([], [])

        if t:
            self.plot_widget.setXRange(t[0], t[-1], padding=0.02)

        self._review_video_path = data.get("video")
        has_video = bool(self._review_video_path and Path(self._review_video_path).exists())
        self.review_video_btn.setVisible(has_video)

        self._restyle_trial_buttons()
        self.status_lbl.setText(
            f"Reviewing {data['title']} — click the trial again to return to live"
        )

    def _exit_review(self):
        if self.review_key is None:
            return
        self.review_key = None
        self._review_video_path = None
        self.review_video_btn.setVisible(False)
        self._restyle_trial_buttons()
        self._update_controls()      # restore the status line
        self._refresh_plot()         # repaint the live view immediately

    def _restyle_trial_buttons(self):
        for k, btn in self.trial_labels.items():
            if k == self.review_key:
                btn.setStyleSheet(
                    "QPushButton { border: 1px solid #27ae60; border-radius: 6px;"
                    " padding: 2px 8px; color: white; font-weight: bold; background: #27ae60; }"
                )
            else:
                btn.setStyleSheet(
                    "QPushButton { border: 1px solid #27ae60; border-radius: 6px;"
                    " padding: 2px 8px; color: #27ae60; font-weight: bold; background: white; }"
                    "QPushButton:hover { background: #eafaf0; }"
                )

    def _open_review_video(self):
        try:
            if self._review_video_path:
                os.startfile(str(self._review_video_path))   # Windows default player
        except Exception as e:
            logging.warning("⚠️ Could not open video: %s", e)

    def _redo_trial(self):
        # delete this trial's data and set up the SAME number again, stopped.
        self.camera.stop_recording()   # close the video if the trial was still active
        self.audio.stop_recording()    # close the audio if it was still active

        doomed = [
            self.group_dir / f"trial{self.trial_num}.txt",
            self.group_dir / f"video{self.trial_num}.avi",
            self.group_dir / f"video{self.trial_num}timestamps.csv",
            self.group_dir / f"_audio{self.trial_num}.wav",
            self.group_dir / f"_muxed{self.trial_num}.avi",
        ]
        for path in doomed:
            try:
                if path.exists():
                    os.remove(path)
                    logging.info("🗑️ Deleted %s for redo.", path)
            except Exception as e:
                logging.warning("⚠️ Could not delete %s: %s", path, e)

        # forget the cached graphs for this trial
        self.trial_data.pop((self.current_obj, self.trial_num), None)

        # remove the completed label if the trial had already ended
        lbl = self.trial_labels.pop((self.current_obj, self.trial_num), None)
        if lbl is not None:
            self.labels_layout.removeWidget(lbl)
            lbl.deleteLater()

        # free this object's trial number so the same one is reused
        if self.object_counts.get(self.current_obj) == self.trial_num:
            self.object_counts[self.current_obj] = self.trial_num - 1

        redo_obj = self.current_obj
        self._reset_to_idle()
        logging.info("↺ %s Trial %d ready to redo (stopped).", redo_obj, self.trial_num)

    def _reset_to_idle(self):
        self.state = self.IDLE
        self.current_obj = None
        self.elapsed_ns = 0
        self.last_perf_ns = None
        self.tns_buf = []
        self.t_buf = []
        self.c1_buf, self.c2_buf, self.c3_buf = [], [], []
        # next number: continue this group, or restart at 1 if object changed
        self.trial_num = self._next_trial_number()
        self._clear_curves()
        self._update_controls()

    # ----------------------------------------------------------
    # Preview mode
    # ----------------------------------------------------------
    def _enter_preview(self):
        self.prev_state = self.state
        self.state = self.PREVIEW
        self.pv_t.clear(); self.pv_c1.clear(); self.pv_c2.clear(); self.pv_c3.clear()
        self.pv_idx = 0
        self._clear_curves()
        self._update_controls()

    def _exit_preview(self):
        self.state = self.prev_state
        self._clear_curves()
        # restore the trial's accumulated data on screen if one is in progress
        self._refresh_plot()
        self._update_controls()

    # ----------------------------------------------------------
    # Sample handling
    # ----------------------------------------------------------
    def _on_sample_threadsafe(self, c1, c2, c3):
        # called from the serial background thread
        self.sample_sig.emit(c1, c2, c3)

    @QtCore.pyqtSlot(float, float, float)
    def _on_sample(self, c1, c2, c3):
        if self.state == self.LOGGING:
            now = time.perf_counter_ns()
            dt = 0 if self.last_perf_ns is None else (now - self.last_perf_ns)
            self.elapsed_ns += dt
            self.last_perf_ns = now

            self.tns_buf.append(self.elapsed_ns)
            self.t_buf.append(self.elapsed_ns / 1e9)
            self.c1_buf.append(c1)
            self.c2_buf.append(c2)
            self.c3_buf.append(c3)

        elif self.state == self.PREVIEW:
            self.pv_t.append(self.pv_idx)
            self.pv_c1.append(c1)
            self.pv_c2.append(c2)
            self.pv_c3.append(c3)
            self.pv_idx += 1
        # IDLE / PAUSED / ENDED -> sample ignored (not plotted, not logged)

    def _refresh_plot(self):
        # while reviewing a past trial, the graphs are static — leave them be
        if self.review_key is not None:
            return

        # sound graph: time-aligned envelope, shown during a trial (logging,
        # paused) and after it ends so it scrolls with the EMG; blank otherwise
        if self.state in (self.LOGGING, self.PAUSED, self.ENDED):
            et, ea = self.audio.get_envelope()
            if et:
                self.audio_curve.setData(et, ea)
                # auto-zoom Y to the peak so quiet mics still show clear spikes
                span = max(max(ea) * 1.2, AUDIO_MIN_SPAN)
                self.audio_plot.setYRange(0.0, span, padding=0)
            else:
                self.audio_curve.setData([], [])
        else:
            self.audio_curve.setData([], [])

        if self.state in (self.LOGGING, self.PAUSED, self.ENDED) and self.t_buf:
            self.curve1.setData(self.t_buf, self.c1_buf)
            self.curve2.setData(self.t_buf, self.c2_buf)
            self.curve3.setData(self.t_buf, self.c3_buf)
        elif self.state == self.PREVIEW and self.pv_t:
            x = list(self.pv_t)
            self.curve1.setData(x, list(self.pv_c1))
            self.curve2.setData(x, list(self.pv_c2))
            self.curve3.setData(x, list(self.pv_c3))

    def _clear_curves(self):
        self.curve1.setData([], [])
        self.curve2.setData([], [])
        self.curve3.setData([], [])

    # ----------------------------------------------------------
    # Camera display
    # ----------------------------------------------------------
    def _update_camera(self):
        frame = self.camera.get_latest_frame()
        if frame is None:
            return
        h, w, ch = frame.shape
        qimg = QtGui.QImage(frame.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            self.video_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pix)

    def _on_builtin_cam_toggled(self, checked: bool):
        """User toggled the 'Use built-in camera' checkbox.
        When enabled we force camera index 0; when disabled we auto-detect.
        """
        self.use_builtin_camera = bool(checked)
        # set camera index to 0 for built-in, None to auto-detect otherwise
        self.camera.cam_index = 0 if self.use_builtin_camera else None
        self._restart_camera()

    def _restart_camera(self):
        """Stop and restart the camera + timer according to current settings."""
        try:
            self.cam_timer.stop()
        except Exception:
            pass
        try:
            self.camera.stop()
        except Exception:
            pass

        if self.camera.start():
            self.cam_timer.start(int(1000 / CAM_FPS))
            self.video_label.setText("")
        else:
            self.video_label.setText("Camera unavailable")

    # ----------------------------------------------------------
    # Control enable/label state
    # ----------------------------------------------------------
    def _update_controls(self):
        n = self.trial_num
        s = self.state
        # object in play: locked one during a trial, else the box selection
        obj = self.current_obj if self.current_obj else self._current_object()

        # primary trial button text
        if s == self.IDLE:
            self.trial_btn.setText(f"Start {obj} Trial {n}")
        elif s == self.LOGGING:
            self.trial_btn.setText(f"Pause {obj} Trial {n}")
        elif s == self.PAUSED:
            self.trial_btn.setText(f"Resume {obj} Trial {n}")
        else:  # ENDED / PREVIEW
            self.trial_btn.setText(f"Start {obj} Trial {n}")

        self.next_btn.setText(f"Start {obj} Trial {n + 1}  ▶")

        # enable/disable per state
        active = s in (self.LOGGING, self.PAUSED)
        self.trial_btn.setEnabled(s in (self.IDLE, self.LOGGING, self.PAUSED))
        self.end_btn.setEnabled(active)
        self.redo_btn.setEnabled(s in (self.LOGGING, self.PAUSED, self.ENDED))
        self.next_btn.setVisible(s == self.ENDED)
        # preview only allowed when no trial is mid-recording
        self.preview_btn.setEnabled(s in (self.IDLE, self.ENDED, self.PREVIEW))

        if s == self.PREVIEW:
            # lock trial controls while previewing
            self.trial_btn.setEnabled(False)
            self.end_btn.setEnabled(False)
            self.redo_btn.setEnabled(False)
            self.next_btn.setVisible(False)

        # status text
        status = {
            self.IDLE:    f"Status: {obj} Trial {n} ready — press Start to log",
            self.LOGGING: f"Status: Logging {obj} Trial {n}…",
            self.PAUSED:  f"Status: {obj} Trial {n} paused",
            self.ENDED:   f"Status: {obj} Trial {n} ended — Redo, or start next trial",
            self.PREVIEW: "Status: Preview (not logging)",
        }[s]
        self.status_lbl.setText(status)

    # ----------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------
    def closeEvent(self, event):
        try:
            self.cam_timer.stop()
            self.camera.stop()
            self.audio.stop()
            self.serial.stop()
        finally:
            super().closeEvent(event)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    app = QtWidgets.QApplication(sys.argv)
    w = TrialLoggerApp()
    w.resize(720, 760)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
