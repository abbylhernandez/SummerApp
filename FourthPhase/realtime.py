import sys
import logging
import wave
import subprocess
from datetime import datetime
from pathlib import Path
import threading
import queue
import os
from collections import deque
import time

import numpy as np
import serial
import cv2
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

try:
    import sounddevice as sd
except Exception as _e:  # pragma: no cover - environment dependent
    sd = None
    logging.warning("sounddevice unavailable; microphone disabled: %s", _e)

try:
    import imageio_ffmpeg
except Exception as _e:  # pragma: no cover - environment dependent
    imageio_ffmpeg = None
    logging.warning("imageio_ffmpeg unavailable; audio will not be muxed into video: %s", _e)


# =========================
# Audio config
# =========================
AUDIO_DEVICE = None                # mic input device (None = system default)
AUDIO_PREFERRED_NAMES = ("MacBook", "Built-in Microphone", "Built-in Input")
AUDIO_RATE = 44100                 # sample rate (Hz)
AUDIO_BLOCK = 1024                 # samples per audio callback block
AUDIO_ENV_CHUNK = 441              # samples per envelope point (~10 ms at 44.1 kHz)
AUDIO_MIN_SPAN = 0.02              # smallest y-height so silence doesn't over-zoom

# =========================
# Theme palettes (mirrors FirstPhase/theme.py)
# =========================
THEMES = {
    "light": {
        "window": "#eef2f7", "panel": "#ffffff", "text": "#1e293b",
        "border": "#bcccdc", "accent": "#2563eb", "plot_bg": "w",
        "axis": "#334155", "grid": 0.20, "btn_text": "Dark",
    },
    "dark": {
        "window": "#0f172a", "panel": "#1e293b", "text": "#e2e8f0",
        "border": "#334155", "accent": "#3b82f6", "plot_bg": "#111827",
        "axis": "#94a3b8", "grid": 0.30, "btn_text": "Light",
    },
}

# EMG channel + microphone curve colors (bright; readable on both themes)
CH_COLORS = ("#FF6B6B", "#22B14C", "#3498DB")
MIC_COLOR = "#9B59B6"


def build_stylesheet(pal):
    """Qt stylesheet for the main window for a given palette dict."""
    return f"""
    QWidget {{ background-color: {pal['window']}; color: {pal['text']}; }}
    QLabel {{ color: {pal['text']}; background: transparent; }}
    QLineEdit {{
        background-color: {pal['panel']}; color: {pal['text']};
        border: 1px solid {pal['border']}; border-radius: 4px; padding: 3px;
    }}
    QPushButton {{
        background-color: {pal['panel']}; color: {pal['text']};
        border: 1px solid {pal['border']}; border-radius: 6px; padding: 4px 10px;
    }}
    QPushButton:hover {{ border-color: {pal['accent']}; }}
    QPushButton:checked {{ background-color: {pal['accent']}; color: white; }}
    QPushButton:disabled {{ color: gray; border-color: {pal['border']}; }}
    """


# =========================
# AudioCapture (ported from FirstPhase/audio_capture.py)
# =========================
class AudioCapture:
    """
    Captures the microphone via sounddevice. While a trial records, it
    (a) writes full-resolution mono audio to a WAV for muxing into the video, and
    (b) builds a time-stamped amplitude envelope (peak per short window) so the
    sound graph aligns in time with the trial.
    """

    def __init__(self, device=AUDIO_DEVICE, samplerate=AUDIO_RATE,
                 blocksize=AUDIO_BLOCK, chunk=AUDIO_ENV_CHUNK):
        self.device = device
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.chunk = chunk
        self.stream = None
        self.running = False

        # WAV recording
        self._wav = None
        self._wav_path = None
        self._wav_recording = False
        self._wav_lock = threading.Lock()

        # time-stamped envelope for plotting (recorded-sample clock)
        self.env_t = []
        self.env_a = []
        self._rec_samples = 0
        self._env_lock = threading.Lock()

    def start(self):
        if sd is None:
            logging.error("Microphone unavailable: sounddevice not installed.")
            return False
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
            logging.info("Microphone capture started.")
            return True
        except Exception as e:
            logging.error("Microphone capture failed: %s", e)
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
        logging.info("Audio recording started: %s", path)
        return True

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
            logging.info("Audio recording saved: %s", path)
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
                logging.warning("Error closing audio stream: %s", e)
            self.stream = None
        logging.info("Microphone capture stopped.")


# =========================
# CameraCapture class
# =========================
class CameraCapture:
    def __init__(self, cam_index=0, width=640, height=360, fps=30):
        self.cam_index = cam_index  # camera index
        self.width = width
        self.height = height
        self.fps = fps

        self.cap = None
        self.running = False
        self.thread = None

        self.frame_queue = queue.Queue(maxsize=1)

        self.recording = False
        self.video_writer = None
        self.record_path = None

    def start(self):
        # CAP_DSHOW is good for Windows
        self.cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            logging.error("❌ Failed to open camera.")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logging.info(f"✅ Camera started at {self.width}x{self.height}")
        return True

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            resized = cv2.resize(frame, (self.width, self.height))

            # If recording, overlay timestamp
            if self.recording:
                ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                cv2.putText(
                    resized,
                    ts,
                    (10, self.height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            # For display in Qt (RGB)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            if not self.frame_queue.full():
                self.frame_queue.put(rgb)

            # For saving to disk (BGR)
            if self.recording and self.video_writer is not None:
                self.video_writer.write(resized)

    def get_latest_frame(self):
        if not self.frame_queue.empty():
            return self.frame_queue.get()
        return None

    def start_recording(self, filename=None, base_folder=None):
        """
        Start recording to AVI file.

        - If base_folder given, file is saved inside that folder.
        - If filename is None, default to 'video.avi'.
        """
        if filename is None:
            filename = "video.avi"
        if base_folder is not None:
            os.makedirs(base_folder, exist_ok=True)
            filename = os.path.join(base_folder, filename)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # good on Windows
        self.record_path = filename
        self.video_writer = cv2.VideoWriter(
            filename, fourcc, self.fps, (self.width, self.height)
        )

        if not self.video_writer.isOpened():
            logging.error(f"❌ Failed to open video file for writing: {filename}")
            self.video_writer = None
            self.recording = False
            return False

        self.recording = True
        logging.info(f"🎥 Video recording started: {filename}")
        return True

    def stop_recording(self):
        if self.recording:
            self.recording = False
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            logging.info(f"🎞️ Video recording stopped and saved to: {self.record_path}")

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        logging.info("📷 Camera stopped.")


# =========================
# SerialEMGHandler
# =========================
class SerialEMGHandler:
    """
    Talks to STM32 over USART2.

    MCU sends, for uart_flag3 == 1, lines like:

        "%d,%d,%d,%c,%d\r\n"

    That is:

        ch1_raw, ch2_raw, ch3_raw, pred_char, button

    Example:

        "2212,1138,2415,2,1"
    """
 
    def __init__(self, port='COM8', baudrate=500000):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.thread = None
        self.running = False

        # Callbacks:
        #   sample_callback(timestamp_str, ch1_v, ch2_v, ch3_v, pred_class_or_char, button)
        # Backward compatibility:
        #   emg_callback(timestamp_str, ch1_v, ch2_v, ch3_v)
        #   pred_callback(timestamp_str, pred_class_or_char, button)
        self.sample_callback = None
        self.emg_callback = None
        self.pred_callback = None

    # ---------- Public API ----------

    def set_sample_callback(self, func):
        self.sample_callback = func

    def set_emg_callback(self, func):
        self.emg_callback = func

    def set_pred_callback(self, func):
        self.pred_callback = func

    def open(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            self.ser.flushInput()
            logging.info(f"✅ Opened {self.port} @ {self.baudrate}")
        except serial.SerialException as e:
            logging.error(f"❌ Failed to open serial port {self.port}: {e}")
            self.ser = None

    def start_stream(self, mode_char=b'c'):
        """
        Send 'c' to MCU to enable uart_flag3 (EMG + predictions).
        """
        if not self.ser or not self.ser.is_open:
            self.open()
            if not self.ser:
                return

        try:
            self.ser.write(mode_char)  # single char 'c'
            logging.info(f"▶️ Sent {mode_char!r} to MCU")
        except Exception as e:
            logging.error(f"❌ Failed to send start command: {e}")
            return

        self.running = True
        self.thread = threading.Thread(target=self.read_loop, daemon=True)
        self.thread.start()

    def stop_stream(self, stop_char=b'v'):
        """
        Send 'v' → firmware goes to 'else' and stops ADC.
        """
        logging.info("🛑 Stopping serial handler...")
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(stop_char)
                logging.info(f"▶️ Sent {stop_char!r} to MCU")
            except Exception as e:
                logging.warning(f"⚠️ Failed to send stop command: {e}")

        self.running = False
        if self.thread:
            self.thread.join(timeout=1)

    def close(self):
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
                logging.info("🔌 Closed serial port")
            except Exception as e:
                logging.warning(f"⚠️ Error closing serial: {e}")

    # ---------- Reader (HARDENED FOR NEW FORMAT) ----------

    def read_loop(self):
        VREF = 3.0
        ADC_RES = 4095.0
        MIDPOINT = 0.0

        ADC_MIN = 0
        ADC_MAX = 4095

        while self.running:
            try:
                if not self.ser:
                    break

                raw = self.ser.readline()
                if not raw:
                    continue

                line = raw.decode('utf-8', errors='ignore').strip()
                if not line:
                    continue

                logging.debug(f"RAW LINE: {repr(line)}")
                now_str = datetime.now().strftime('%H:%M:%S.%f')[:-3]

                # Expect exactly: v1,v2,v3,pred_char,button  -> 4 commas
                if line.count(',') != 4:
                    logging.debug(f"[IGNORED] Not 5 fields: {repr(line)}")
                    continue

                try:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) != 5:
                        raise ValueError(f"expected 5 fields, got {len(parts)}")

                    a, b, c, pred_token, button_token = parts

                    # ---------- Parse and validate ADC fields ----------
                    vals = []
                    for idx, p in enumerate((a, b, c)):
                        # must be digits only (no '-', no junk)
                        if not p.isdigit():
                            raise ValueError(f"field {idx} not all digits: {p!r}")

                        # at most 4 digits for 0..4095
                        if len(p) > 4:
                            raise ValueError(f"field {idx} too many digits: {p!r}")

                        v = int(p)

                        # must be in [0, 4095]
                        if v < ADC_MIN or v > ADC_MAX:
                            logging.error(
                                f"🚨 RAW SPIKE DETECTED in line '{line}': "
                                f"field {idx} = {v} (valid {ADC_MIN}..{ADC_MAX})"
                            )
                            raise ValueError(
                                f"field {idx} out of range {ADC_MIN}..{ADC_MAX}: {v}"
                            )

                        vals.append(v)

                    ch1_raw, ch2_raw, ch3_raw = vals

                    # ---------- Parse prediction char ----------
                    if len(pred_token) == 0:
                        raise ValueError("empty prediction token")

                    # Use first character as class (e.g. '1', '2', '?', etc.)
                    pred_char = pred_token[0]

                    # If more than one char, log but still continue
                    if len(pred_token) > 1:
                        logging.warning(
                            f"⚠️ pred_token has extra chars: {pred_token!r}, using {pred_char!r}"
                        )

                    # ---------- Parse button state ----------
                    if not button_token.isdigit():
                        raise ValueError(f"button token is not digits: {button_token!r}")

                    button_state = int(button_token)

                    # ---------- Convert to voltages ----------
                    ch1_v = (ch1_raw / ADC_RES) * VREF - MIDPOINT
                    ch2_v = (ch2_raw / ADC_RES) * VREF - MIDPOINT
                    ch3_v = (ch3_raw / ADC_RES) * VREF - MIDPOINT

                    # Extra sanity: voltage should be 0..3.1 V
                    if not (0.0 <= ch1_v <= 3.1 and
                            0.0 <= ch2_v <= 3.1 and
                            0.0 <= ch3_v <= 3.1):
                        logging.error(
                            "🚨 VOLTAGE SPIKE DETECTED from line '%s': "
                            "V = (%.6f, %.6f, %.6f)",
                            line, ch1_v, ch2_v, ch3_v
                        )
                        raise ValueError(
                            f"voltage out of expected range: "
                            f"{ch1_v:.6f}, {ch2_v:.6f}, {ch3_v:.6f}"
                        )

                    # If class is digit, keep it as int for easier downstream handling.
                    if isinstance(pred_char, str) and pred_char.isdigit():
                        pred_val = int(pred_char)
                    else:
                        pred_val = pred_char

                    # ---------- Everything is clean: fire callbacks ----------
                    if self.sample_callback:
                        self.sample_callback(
                            now_str,
                            ch1_v,
                            ch2_v,
                            ch3_v,
                            pred_val,
                            button_state,
                        )
                    else:
                        # Fallback to older split callbacks.
                        if self.emg_callback:
                            self.emg_callback(now_str, ch1_v, ch2_v, ch3_v)
                        if self.pred_callback:
                            self.pred_callback(now_str, pred_val, button_state)

                except Exception as e:
                    # Anything weird (merged lines, garbage, spike) is skipped
                    logging.warning(
                        f"⚠️ Skipping suspicious line '{line}': {e}"
                    )

            except Exception as e:
                logging.warning(f"⚠️ Serial read error: {e}")
                continue
# =========================
# RealTimeTestApp (EMG + camera + video)
# =========================
class RealTimeTestApp(QtWidgets.QWidget):
    PLOT_DOWNSAMPLE = 30
    PLOT_MAX_POINTS = 180
    PLOT_UPDATE_MS = 250
    EMG_LABEL_STRIDE = 10

    # Signals to safely update GUI from callbacks (background thread)
    emg_sig = QtCore.pyqtSignal(float, float, float)
    plot_sig = QtCore.pyqtSignal(float, float, float)
    pred_sig = QtCore.pyqtSignal(object, int)

    def __init__(self, port='COM8', baudrate=500000, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Real-Time EMG + Prediction + Video")

        self._theme_name = "dark"

        # Serial
        self.serial = SerialEMGHandler(port=port, baudrate=baudrate)
        self.serial.set_sample_callback(self.on_sample)

        # Camera
        self.camera = CameraCapture(cam_index=0, width=640, height=360, fps=30)
        self.cam_timer = QtCore.QTimer(self)
        self.cam_timer.timeout.connect(self.update_camera_frame)

        # Microphone
        self.audio = AudioCapture()

        # Start camera ONCE at app startup
        if self.camera.start():
            self.cam_timer.start(30)
        else:
            QtWidgets.QMessageBox.warning(self, "Camera Error", "Could not start camera.")

        # Start microphone ONCE at app startup
        self.audio_available = self.audio.start()

        # Files
        self.collecting = False
        self.current_set_dir: Path | None = None
        self.emg_file = None
        self.pred_file = None
        self.button_file = None
        self.t0_ns = None
        self.sample_counter = 0
        self.last_pred_ui = None
        self.last_button_ui = None

        # Plot buffers (kept intentionally small for low-resolution plotting)
        self.plot_x = deque(maxlen=self.PLOT_MAX_POINTS)
        self.plot_ch1 = deque(maxlen=self.PLOT_MAX_POINTS)
        self.plot_ch2 = deque(maxlen=self.PLOT_MAX_POINTS)
        self.plot_ch3 = deque(maxlen=self.PLOT_MAX_POINTS)
        self.plot_point_idx = 0

        # ---------- NEW SAVING STRUCTURE ----------
        # Root folder
        self.base_dir = Path("realtimetest")
        self.base_dir.mkdir(exist_ok=True)

        # Create one session folder when app opens
        session_name = datetime.now().strftime("SESSION_%Y%m%d_%H%M%S")
        self.session_dir = self.base_dir / session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Subfolder for all prediction text files
        self.predictions_dir = self.session_dir / "predictions"
        self.predictions_dir.mkdir(exist_ok=True)

        # trial_1.txt, trial_2.txt, ...
        self.trial_counter = 0
        # ------------------------------------------

        # --- UI widgets ---
        self.start_btn = QtWidgets.QPushButton("Start (mode 'c')")
        self.stop_btn = QtWidgets.QPushButton("Stop (send 'v')")
        self.theme_btn = QtWidgets.QPushButton("Dark")
        self.status_lbl = QtWidgets.QLabel("Status: Idle")
        self.pred_label = QtWidgets.QLabel("Last prediction: -")
        self.button_label = QtWidgets.QLabel("Last button: -")
        self.emg_label = QtWidgets.QLabel("Last EMG: -")
        self.video_label = QtWidgets.QLabel("Camera starting...")
        self.video_label.setFixedSize(640, 360)
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")

        pg.setConfigOptions(antialias=False)

        # Microphone plot (on top of EMG, mirrors FirstPhase)
        self.audio_plot = pg.PlotWidget()
        self.audio_plot.setFixedHeight(120)
        self.audio_plot.setTitle("Microphone")
        self.audio_plot.setLabel("left", "Amplitude")
        self.audio_plot.getPlotItem().hideAxis("bottom")
        self.audio_plot.setMouseEnabled(x=False, y=False)
        self.audio_curve = self.audio_plot.plot([], [], pen=pg.mkPen(MIC_COLOR, width=1))
        if not self.audio_available:
            self.audio_plot.setTitle("Microphone unavailable")

        # Low-cost EMG plot (downsampled + timer-driven redraw)
        self.emg_plot = pg.PlotWidget()
        self.emg_plot.setFixedHeight(170)
        self.emg_plot.setLabel("left", "EMG (V)")
        self.emg_plot.setLabel("bottom", "Low-res index")
        self.emg_plot.setYRange(0.0, 3.0, padding=0)
        self.emg_plot.getPlotItem().setDownsampling(auto=False, ds=1, mode='peak')
        self.emg_plot.getPlotItem().setClipToView(True)
        self._legend = self.emg_plot.addLegend(offset=(10, 10))

        self.plot_curve1 = self.emg_plot.plot([], [], pen=pg.mkPen(CH_COLORS[0], width=2), name="ch1")
        self.plot_curve2 = self.emg_plot.plot([], [], pen=pg.mkPen(CH_COLORS[1], width=2), name="ch2")
        self.plot_curve3 = self.emg_plot.plot([], [], pen=pg.mkPen(CH_COLORS[2], width=2), name="ch3")

        # Layout
        top_row = QtWidgets.QHBoxLayout()
        top_row.addWidget(self.status_lbl, stretch=1)
        top_row.addWidget(self.theme_btn)

        btn_layout = QtWidgets.QHBoxLayout()
        self.start_btn.setMinimumHeight(34)
        self.stop_btn.setMinimumHeight(34)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)

        label_layout = QtWidgets.QVBoxLayout()
        label_layout.addWidget(self.pred_label)
        label_layout.addWidget(self.button_label)
        label_layout.addWidget(self.emg_label)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(top_row)
        main_layout.addLayout(btn_layout)
        main_layout.addLayout(label_layout)
        main_layout.addWidget(self.video_label, alignment=QtCore.Qt.AlignHCenter)
        main_layout.addWidget(self.audio_plot)
        main_layout.addWidget(self.emg_plot)
        self.setLayout(main_layout)

        # Connections
        self.start_btn.clicked.connect(self.handle_start)
        self.stop_btn.clicked.connect(self.handle_stop)
        self.theme_btn.clicked.connect(self.on_theme_btn)

        self.emg_sig.connect(self.update_emg_label)
        self.plot_sig.connect(self.on_plot_sample)
        self.pred_sig.connect(self.update_pred_labels)

        self.plot_timer = QtCore.QTimer(self)
        self.plot_timer.timeout.connect(self.refresh_plot)
        self.plot_timer.start(self.PLOT_UPDATE_MS)

        self._apply_theme(self._theme_name)

    # ---------- Theme ----------

    def on_theme_btn(self):
        self._apply_theme("dark" if self._theme_name == "light" else "light")

    def _apply_theme(self, name):
        self._theme_name = name
        pal = THEMES[name]
        self.setStyleSheet(build_stylesheet(pal))
        self.status_lbl.setStyleSheet(f"font-weight: bold; color: {pal['text']};")
        self.theme_btn.setText(pal["btn_text"])

        for p in (self.audio_plot, self.emg_plot):
            p.setBackground(pal["plot_bg"])
            pi = p.getPlotItem()
            pi.showGrid(x=True, y=True, alpha=pal["grid"])
            for axname in ("left", "bottom"):
                ax = pi.getAxis(axname)
                ax.setPen(pg.mkPen(pal["axis"]))
                ax.setTextPen(pg.mkPen(pal["axis"]))

        self.audio_plot.setTitle(
            "Microphone" if self.audio_available else "Microphone unavailable",
            color=pal["axis"],
        )
        self.audio_plot.setLabel("left", "Amplitude", color=pal["axis"])
        self.emg_plot.setLabel("left", "EMG (V)", color=pal["axis"])
        self.emg_plot.setLabel("bottom", "Low-res index", color=pal["axis"])
        try:
            self._legend.setLabelTextColor(pal["axis"])
        except Exception:
            pass

    # ---------- Start / Stop ----------

    def handle_start(self):
        if self.collecting:
            return

        # Each press of Start = new trial number
        self.trial_counter += 1
        trial_idx = self.trial_counter

        # Folder where EMG + videos live (the session folder)
        self.current_set_dir = self.session_dir

        # EMG file in session folder: trial_1.txt, trial_2.txt, ...
        emg_path = self.current_set_dir / f"trial_{trial_idx}.txt"

        # Prediction file in predictions subfolder: trial_1.txt, trial_2.txt, ...
        pred_path = self.predictions_dir / f"trial_{trial_idx}.txt"
        # Button file in session folder: button_1.txt, button_2.txt, ...
        button_path = self.current_set_dir / f"button_{trial_idx}.txt"

        # Open EMG / prediction / button files
        self.emg_file = open(emg_path, "w", buffering=1)
        self.pred_file = open(pred_path, "w", buffering=1)
        self.button_file = open(button_path, "w", buffering=1)

        # Same style as Application5: button file uses t_ns from trial start.
        self.t0_ns = time.perf_counter_ns()

        self.emg_file.write("timestamp\tch1\tch2\tch3\n")
        self.pred_file.write("timestamp\tclass\tbutton\n")
        self.button_file.write("t_ns,button\n")

        self.collecting = True
        self.sample_counter = 0
        self.last_pred_ui = None
        self.last_button_ui = None
        self.plot_point_idx = 0
        self.plot_x.clear()
        self.plot_ch1.clear()
        self.plot_ch2.clear()
        self.plot_ch3.clear()

        # Start serial stream
        self.serial.start_stream(mode_char=b'c')

        # Start video recording in same session folder
        if self.camera.cap is not None:
            self.camera.start_recording(
                filename=f"video_{trial_idx}.avi",
                base_folder=str(self.current_set_dir),
            )

        # Start audio recording (WAV, muxed into video at trial end)
        if self.audio_available:
            self.audio.start_recording(self.current_set_dir / f"_audio_{trial_idx}.wav")

        self.pred_label.setText("Last prediction: -")
        self.button_label.setText("Last button: -")
        self.emg_label.setText("Last EMG: -")
        self.audio_curve.setData([], [])

        self.status_lbl.setText(f"Status: Logging trial {trial_idx}…")
        logging.info(f"✅ Started new trial {trial_idx} in {self.current_set_dir}")

    def handle_stop(self):
        if not self.collecting:
            return

        trial_idx = self.trial_counter

        # Stop serial stream
        self.serial.stop_stream(stop_char=b'v')

        # Close EMG / prediction / button files
        if self.emg_file:
            self.emg_file.close()
            self.emg_file = None
        if self.pred_file:
            self.pred_file.close()
            self.pred_file = None
        if self.button_file:
            self.button_file.close()
            self.button_file = None

        self.t0_ns = None
        self.collecting = False
        logging.info("🛑 Stopped EMG/prediction/button recording.")

        # Stop video recording only (keep camera running)
        if self.camera.recording:
            self.camera.stop_recording()

        # Stop audio, write the envelope CSV, and mux the WAV into the video
        if self.audio_available:
            wav_path = self.audio.stop_recording()
            self._write_audio_envelope(trial_idx)
            self._mux_audio_into_video(trial_idx, wav_path)

        self.status_lbl.setText(f"Status: Trial {trial_idx} saved.")

    # ---------- Audio persistence ----------

    def _write_audio_envelope(self, trial_idx):
        """Write the per-trial microphone envelope (audio_N.csv) like FirstPhase."""
        et, ea = self.audio.get_envelope()
        if not et:
            return None
        path = self.session_dir / f"audio_{trial_idx}.csv"
        try:
            with open(path, "w", newline="") as f:
                f.write("t_s,amp\n")
                for t_s, amp in zip(et, ea):
                    f.write(f"{t_s:.6f},{amp:.6f}\n")
            logging.info("Wrote %s (%d points).", path, len(et))
        except Exception as e:
            logging.warning("Failed to write audio envelope %s: %s", path, e)
            return None
        return path

    def _mux_audio_into_video(self, trial_idx, wav_path):
        """Merge the recorded WAV into video_N.avi, then delete the WAV."""
        video = self.session_dir / f"video_{trial_idx}.avi"
        if not wav_path or not Path(wav_path).exists() or not video.exists():
            if wav_path and Path(wav_path).exists():
                os.remove(wav_path)
            return
        if imageio_ffmpeg is None:
            logging.warning("imageio_ffmpeg missing; keeping video silent (WAV kept: %s)", wav_path)
            return
        out = self.session_dir / f"_muxed_{trial_idx}.avi"
        cmd = [
            imageio_ffmpeg.get_ffmpeg_exe(), "-y",
            "-i", str(video), "-i", str(wav_path),
            "-c:v", "copy", "-c:a", "pcm_s16le", "-shortest", str(out),
        ]
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, creationflags=flags)
            os.replace(out, video)
            logging.info("Merged audio into %s", video)
        except Exception as e:
            logging.warning("Audio mux failed (%s); video kept silent.", e)
            if out.exists():
                os.remove(out)
        finally:
            if Path(wav_path).exists():
                os.remove(wav_path)

    # ---------- Camera display ----------

    def update_camera_frame(self):
        frame = self.camera.get_latest_frame()
        if frame is None:
            return
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pix)

    # ---------- Callbacks from SerialEMGHandler (background thread) ----------

    def on_sample(self, timestamp_str, ch1, ch2, ch3, pred_class, button_state):
        """
        Single per-sample callback from serial parser.
        One line in all files shares exactly the same timestamp.
        """
        if not self.collecting:
            return

        if self.emg_file is not None:
            try:
                self.emg_file.write(f"{timestamp_str}\t{ch1:.6f}\t{ch2:.6f}\t{ch3:.6f}\n")
            except Exception as e:
                logging.warning(f"⚠️ Failed to write EMG line: {e}")

        if self.pred_file is not None:
            try:
                self.pred_file.write(f"{timestamp_str}\t{pred_class}\t{button_state}\n")
            except Exception as e:
                logging.warning(f"⚠️ Failed to write prediction line: {e}")

        if self.button_file is not None:
            try:
                if self.t0_ns is not None:
                    t_ns = time.perf_counter_ns() - self.t0_ns
                else:
                    t_ns = time.perf_counter_ns()
                self.button_file.write(f"{t_ns},{button_state}\n")
            except Exception as e:
                logging.warning(f"⚠️ Failed to write button line: {e}")

        # Throttle GUI updates so serial + camera paths stay fast.
        self.sample_counter += 1

        if self.sample_counter == 1 or (self.sample_counter % self.EMG_LABEL_STRIDE == 0):
            self.emg_sig.emit(ch1, ch2, ch3)

        if self.sample_counter == 1 or (self.sample_counter % self.PLOT_DOWNSAMPLE == 0):
            self.plot_sig.emit(ch1, ch2, ch3)

        if pred_class != self.last_pred_ui or button_state != self.last_button_ui:
            self.last_pred_ui = pred_class
            self.last_button_ui = button_state
            self.pred_sig.emit(pred_class, button_state)

    # ---------- Slots (GUI thread) ----------

    @QtCore.pyqtSlot(float, float, float)
    def update_emg_label(self, ch1, ch2, ch3):
        self.emg_label.setText(f"Last EMG: {ch1:.3f}, {ch2:.3f}, {ch3:.3f}")

    @QtCore.pyqtSlot(object, int)
    def update_pred_labels(self, pred_class, button_state):
        self.pred_label.setText(f"Last prediction: {pred_class}")
        self.button_label.setText(f"Last button: {button_state}")

    @QtCore.pyqtSlot(float, float, float)
    def on_plot_sample(self, ch1, ch2, ch3):
        self.plot_x.append(self.plot_point_idx)
        self.plot_ch1.append(ch1)
        self.plot_ch2.append(ch2)
        self.plot_ch3.append(ch3)
        self.plot_point_idx += 1

    def refresh_plot(self):
        # Live microphone envelope (only while a trial is recording)
        if self.audio_available and self.collecting:
            et, ea = self.audio.get_envelope()
            if et:
                self.audio_curve.setData(et, ea)
                span = max(max(ea) * 1.2, AUDIO_MIN_SPAN)
                self.audio_plot.setYRange(0.0, span, padding=0)

        if not self.plot_x:
            return

        x = list(self.plot_x)
        self.plot_curve1.setData(x, list(self.plot_ch1))
        self.plot_curve2.setData(x, list(self.plot_ch2))
        self.plot_curve3.setData(x, list(self.plot_ch3))

    # ---------- Cleanup ----------

    def closeEvent(self, event):
        try:
            # stop any ongoing recording
            if self.camera.recording:
                self.camera.stop_recording()
            # stop camera thread
            if self.camera.running:
                self.camera.stop()
            # stop microphone
            if self.audio.running:
                self.audio.stop()
            # stop EMG stuff
            self.serial.stop_stream(stop_char=b'v')
            self.serial.close()
        finally:
            super().closeEvent(event)


def main():
    from pathlib import Path
    import sys

    # Where to save the log file
    log_dir = Path("realtimetest_logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "realtimetest.log"   # or use a timestamped name if you want

    # Configure logging to both file and console
    logging.basicConfig(
        level=logging.DEBUG,  # so you see spike messages too
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info(f"Logging to file: {log_file}")

    app = QtWidgets.QApplication(sys.argv)
    w = RealTimeTestApp(port='COM8', baudrate=500000)
    w.resize(700, 550)
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
