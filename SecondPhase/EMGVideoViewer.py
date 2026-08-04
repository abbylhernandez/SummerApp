import sys
import os
import glob
import csv
import time
import wave
import tempfile
import subprocess
import threading
import cv2
import numpy as np
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout,
    QPushButton, QHBoxLayout, QInputDialog, QSlider,
    QScrollArea, QFrame, QSizePolicy, QMessageBox, QComboBox
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

import pyqtgraph as pg
# Optional audio playback (extract the video's embedded audio track and play it).
try:
    import sounddevice as sd
    SOUNDDEVICE_OK = True
except Exception:
    SOUNDDEVICE_OK = False

try:
    import imageio_ffmpeg
    FFMPEG_OK = True
except Exception:
    FFMPEG_OK = False

AUDIO_OK = SOUNDDEVICE_OK and FFMPEG_OK


class _AudioStreamPlayer:
    """Plays a mono int16 buffer and reports the actual (latency-corrected)
    playback position, so the video/cursor can follow the audio as the clock."""

    def __init__(self, data, rate):
        self.data = data
        self.rate = rate
        self.pos = 0
        self.out_latency = 0.0
        self.stream = None
        self._lock = threading.Lock()

    def _cb(self, outdata, frames, time_info, status):
        with self._lock:
            start = self.pos
            end = min(start + frames, len(self.data))
            self.pos = end
        n = end - start
        if n > 0:
            outdata[:n, 0] = self.data[start:end]
        if n < frames:
            outdata[n:, 0] = 0
            raise sd.CallbackStop

    def start(self, start_sample):
        self.stop()
        with self._lock:
            self.pos = max(0, min(int(start_sample), len(self.data)))
        self.stream = sd.OutputStream(samplerate=self.rate, channels=1,
                                      dtype="int16", callback=self._cb)
        self.stream.start()
        try:
            self.out_latency = float(self.stream.latency)
        except Exception:
            self.out_latency = 0.0

    def active(self):
        return self.stream is not None and self.stream.active

    def position_s(self):
        with self._lock:
            pos = self.pos
        return max(0.0, pos / self.rate - self.out_latency)

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

# ----------------------------- EMG LOADING ----------------------------------

def extract_num_generic(path, prefixes):
    base = os.path.basename(path)       #extract the file name from path input
    name, _ = os.path.splitext(base)    #extract the name of the file and drops the extension

    # This block extracts the trial number, supports the underscore style too
    for prefix in prefixes:
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix):]
        if suffix.startswith("_"):
            suffix = suffix[1:]
        if suffix.isdigit():
            return int(suffix)
    raise ValueError(f"Unexpected filename format: {base}")


def find_trials_in_folder(folder):
    txt_files = glob.glob(os.path.join(folder, "trial*.txt"))
    video_files = (
        glob.glob(os.path.join(folder, "video*.avi"))
        + glob.glob(os.path.join(folder, "trial*.avi"))
    )
    audio_files = (
        glob.glob(os.path.join(folder, "audio*.csv"))
        + glob.glob(os.path.join(folder, "audio*.wav"))
    )

    txt_nums = {extract_num_generic(p, ("trial",)): p for p in txt_files}
    vid_nums = {extract_num_generic(p, ("video", "trial")): p for p in video_files}
    audio_nums = {extract_num_generic(p, ("audio",)): p for p in audio_files}

    common = sorted(set(txt_nums) & set(vid_nums))
    return [
        {
            "Data": None,
            "trial_num": n,
            "folder": folder,
            "emg_path": txt_nums[n],
            "video_path": vid_nums[n],
            "audio_path": audio_nums.get(n),
        }
        for n in common
    ]


def find_trials_under_root(root_folder):
    session_folders = sorted(
        path
        for path in glob.glob(os.path.join(root_folder, "*"))
        if os.path.isdir(path)
    )

    trials = []
    for session_folder in session_folders:
        trials.extend(find_trials_in_folder(session_folder))
    return trials

def compute_next_destination_for_root(result_root, num_acts=2):
    """
    Return the first unused destination in alternating act order:
    act1/trial1, act2/trial1, act1/trial2, ...

    Looking for the first unused path, rather than counting files, prevents a
    partially populated result tree from pointing at an existing file.
    """
    if num_acts < 1:
        raise ValueError("num_acts must be at least 1")

    trial_idx = 1
    while True:
        for act_idx in range(1, num_acts + 1):
            candidate = os.path.join(
                result_root,
                f"act{act_idx}",
                f"trial_{trial_idx}.txt",
            )
            if not os.path.exists(candidate):
                return act_idx, trial_idx
        trial_idx += 1


def completed_result_trial_numbers(session_folder, num_acts=2):
    """Return trial numbers that have a saved output for every act label."""
    result_root = os.path.join(session_folder, "ResultClip")
    completed_by_act = []
    for act_idx in range(1, num_acts + 1):
        numbers = set()
        pattern = os.path.join(result_root, f"act{act_idx}", "trial_*.txt")
        for path in glob.glob(pattern):
            try:
                numbers.add(extract_num_generic(path, ("trial",)))
            except ValueError:
                continue
        completed_by_act.append(numbers)

    if not completed_by_act:
        return []
    return sorted(set.intersection(*completed_by_act))


def select_startup_trial_index(trials, num_acts=2):
    """
    Pick the raw trial whose number matches the next ResultClip destination.
    This keeps the window title and save destination in sync after restarting.
    """
    folders = []
    for trial in trials:
        if trial["folder"] not in folders:
            folders.append(trial["folder"])

    for folder in folders:
        result_root = os.path.join(folder, "ResultClip")
        _, destination_trial_idx = compute_next_destination_for_root(result_root, num_acts)

        for index, trial in enumerate(trials):
            if trial["folder"] == folder and trial["trial_num"] == destination_trial_idx:
                return index

    return 0
def _unwrap_monotonic_ns(raw_ns):
    """Unwrap signed 32-bit wraps into a monotonic int64 timeline."""
    if not raw_ns:
        return np.array([], dtype=np.int64)

    wrap_mod = 2 ** 32
    offset = 0
    prev = int(raw_ns[0])
    out = [prev]

    for t in raw_ns[1:]:
        t = int(t)
        if t < prev:
            offset += wrap_mod
        out.append(t + offset)
        prev = t

    return np.asarray(out, dtype=np.int64)


def _guess_button_path(emg_path):
    try:
        trial_num = extract_num_generic(emg_path, ("trial",))
    except ValueError:
        return None
    folder = os.path.dirname(emg_path)
    for button_name in (f"button_{trial_num}.txt", f"button{trial_num}.txt"):
        button_path = os.path.join(folder, button_name)
        if os.path.exists(button_path):
            return button_path
    return os.path.join(folder, f"button_{trial_num}.txt")


def _load_button_sidecar(button_path, emg_t_ns):
    """Load button values from button_X.txt and align to EMG rows."""
    out = np.zeros(len(emg_t_ns), dtype=np.int8)
    if not button_path or not os.path.exists(button_path):
        return out

    rows = []
    with open(button_path, "r") as f:
        _ = f.readline()  # optional header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                t_ns = int(parts[0])
                b = 1 if int(parts[1]) == 1 else 0
            except ValueError:
                continue
            rows.append((t_ns, b))

    if not rows:
        return out

    if len(rows) == len(emg_t_ns):
        return np.asarray([b for _, b in rows], dtype=np.int8)

    bmap = {t_ns: b for t_ns, b in rows}
    for i, t_ns in enumerate(emg_t_ns):
        out[i] = bmap.get(int(t_ns), 0)
    return out


def load_emg_file(emg_path):
    """
    Load EMG file in old or new format.

    Old formats:
    - HH:MM:SS.mmm  ch1 ch2 ch3
    - YYYY-MM-DD HH:MM:SS.mmm  ch1 ch2 ch3

    New formats:
    - t_ns,ch1_V,ch2_V,ch3_V
    - t_ns,ch1_V,ch2_V,ch3_V,button

    Returns:
        times_sec: relative time (N,)
        emg:       channels (N,3)
        button:    button state (N,), 0/1
    """
    timestamps_dt = []
    timestamps_ns = []
    ch1 = []
    ch2 = []
    ch3 = []
    button_inline = []
    saw_new = False
    saw_old = False
    saw_inline_button = False

    with open(emg_path, "r") as f:
        _ = f.readline()  # header

        for line in f:
            line = line.strip()
            if not line:
                continue

            if "," in line:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    try:
                        t_ns = int(parts[0])
                        v1 = float(parts[1])
                        v2 = float(parts[2])
                        v3 = float(parts[3])
                    except ValueError:
                        pass
                    else:
                        b = 0
                        if len(parts) >= 5:
                            saw_inline_button = True
                            try:
                                b = 1 if int(parts[4]) == 1 else 0
                            except ValueError:
                                b = 0
                        timestamps_ns.append(t_ns)
                        ch1.append(v1)
                        ch2.append(v2)
                        ch3.append(v3)
                        button_inline.append(b)
                        saw_new = True
                        continue

            parts = line.split()
            if len(parts) < 4:
                continue

            if ":" in parts[0] and "-" not in parts[0]:
                ts_str = parts[0]
                v1_str, v2_str, v3_str = parts[1:4]
                t = datetime.strptime(ts_str, "%H:%M:%S.%f")
            else:
                if len(parts) < 5:
                    continue
                dt_str = f"{parts[0]} {parts[1]}"
                v1_str, v2_str, v3_str = parts[2:5]
                try:
                    t = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S.%f")
                except ValueError:
                    t = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")

            timestamps_dt.append(t)
            ch1.append(float(v1_str))
            ch2.append(float(v2_str))
            ch3.append(float(v3_str))
            button_inline.append(0)
            saw_old = True

    if not ch1:
        raise ValueError(f"No data found in {emg_path}")

    if saw_new and saw_old:
        raise ValueError(f"Mixed timestamp formats found in {emg_path}")

    if saw_new:
        t_ns_unwrapped = _unwrap_monotonic_ns(timestamps_ns)
        times_sec = (t_ns_unwrapped - t_ns_unwrapped[0]).astype(np.float64) / 1e9
        if saw_inline_button:
            button = np.asarray(button_inline, dtype=np.int8)
        else:
            button = _load_button_sidecar(_guess_button_path(emg_path), timestamps_ns)
    else:
        t0 = timestamps_dt[0]
        times_sec = np.array([(t - t0).total_seconds() for t in timestamps_dt], dtype=float)
        button = np.asarray(button_inline, dtype=np.int8)

    emg = np.column_stack([ch1, ch2, ch3]).astype(float)
    return times_sec, emg, button


def load_audio_file(audio_path):
    """Load audio CSV data as relative seconds and amplitude."""
    if not audio_path or not os.path.exists(audio_path):
        return np.array([], dtype=float), np.array([], dtype=float)

    if not audio_path.lower().endswith(".csv"):
        print(f"Audio overlay currently supports CSV files only: {audio_path}")
        return np.array([], dtype=float), np.array([], dtype=float)

    times = []
    amps = []
    with open(audio_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and {"t_s", "amp"}.issubset(reader.fieldnames):
            for row in reader:
                try:
                    times.append(float(row["t_s"]))
                    amps.append(float(row["amp"]))
                except (TypeError, ValueError):
                    continue
        else:
            f.seek(0)
            plain_reader = csv.reader(f)
            for row in plain_reader:
                if len(row) < 2:
                    continue
                try:
                    times.append(float(row[0]))
                    amps.append(float(row[1]))
                except ValueError:
                    continue

    if not times:
        return np.array([], dtype=float), np.array([], dtype=float)

    times = np.asarray(times, dtype=float)
    amps = np.asarray(amps, dtype=float)
    return times - times[0], amps


# ----------------------------- HELPER ---------------------------------------

def format_time_from_seconds(s):
    """Convert seconds -> 'HH:MM:SS.mmm' string starting from 00:00:00.xxx."""
    total_ms = max(0, int(round(float(s) * 1000.0)))
    total_sec, ms = divmod(total_ms, 1000)
    h = total_sec // 3600
    m = (total_sec % 3600) // 60
    sec = total_sec % 60
    return f"{h:02d}:{m:02d}:{sec:02d}.{ms:03d}"


def resample_emg_clip(times, emg, target_samples):
    """Interpolate only the EMG channels onto ``target_samples`` timestamps."""
    times = np.asarray(times, dtype=float)
    emg = np.asarray(emg, dtype=float)
    target_samples = int(target_samples)

    if target_samples < 1:
        raise ValueError("target_samples must be at least 1")
    if len(times) == 0 or len(times) != len(emg):
        raise ValueError("EMG timestamps and samples must be non-empty and aligned")
    if target_samples == len(times):
        return times.copy(), emg.copy()

    if len(times) == 1 or times[-1] <= times[0]:
        return (
            np.full(target_samples, times[0], dtype=float),
            np.repeat(emg[:1], target_samples, axis=0),
        )

    resampled_times = np.linspace(times[0], times[-1], target_samples)
    resampled_emg = np.column_stack([
        np.interp(resampled_times, times, emg[:, channel])
        for channel in range(emg.shape[1])
    ])
    return resampled_times, resampled_emg


# ----------------------------- VIEWER CLASS ---------------------------------

class EMGVideoViewer(QWidget):
    PLAYBACK_SPEED = 1.0  # 1.0 = full speed
    DEFAULT_CLIP_SAMPLES = 500
    THEMES = {
        "light": {
            "window": "#eef2f7",
            "panel": "#ffffff",
            "text": "#1e293b",
            "muted": "#64748b",
            "border": "#bcccdc",
            "accent": "#2563eb",
            "plot_bg": "#ffffff",
            "axis": "#334155",
            "grid": 0.20,
        },
        "dark": {
            "window": "#0f172a",
            "panel": "#1e293b",
            "text": "#e2e8f0",
            "muted": "#94a3b8",
            "border": "#334155",
            "accent": "#3b82f6",
            "plot_bg": "#111827",
            "axis": "#94a3b8",
            "grid": 0.30,
        },
    }

    def __init__(
        self,
        video_path,
        emg_times,
        emg_data,
        button_data,
        emg_path,
        audio_path=None,
        available_trials=None,
        current_trial_index=None,
        parent=None,
    ):
        super().__init__(parent)

        self.video_path = video_path
        self.emg_times = emg_times       # 1D array
        self.emg_data = emg_data         # shape (N, 3)
        self.button_data = button_data   # shape (N,), 0/1
        self.emg_path = emg_path         # full path to original EMG file
        self.audio_path = audio_path
        self.audio_times, self.audio_data = load_audio_file(self.audio_path)
        self.folder = os.path.dirname(self.emg_path)
        self.available_trials = (
            list(available_trials) if available_trials is not None
            else find_trials_in_folder(self.folder)
        )
        self.current_trial_index = current_trial_index
        if self.current_trial_index is None:
            self.current_trial_index = self._find_trial_index_by_emg_path(self.emg_path)
        self.redo_output_trial_idx = None
        self.selection_history = {}
        self.completed_trial_buttons = []
        self.theme_name = "dark"

        # how many acts we want to alternate between
        self.num_acts = 2

        # Open video (for playback)
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30.0  # fallback

        self.frame_idx = 0
        self.is_paused = False
        self.segment_stop_time = None
        self.was_playing_before_seek = False
           # Audio track (extracted from the video) for synced playback.
        self.audio_samples = None
        self.audio_rate = None
        self.audio_player = None
        self.audio_master = False   # True when audio drives the playback clock
        self._play_t0 = 0.0
        self._play_t_origin = 0.0
        self._extract_audio()

        # Two fixed-size clipping windows, one for each label (act1 / act2).
        self.clip_samples = min(self.DEFAULT_CLIP_SAMPLES, len(self.emg_times))
        self.label_regions = []
        self.label_sliders = []
        self.label_position_labels = []
        self.region = None  # compatibility alias for the act1 region
        self.button_regions = []
        diffs = np.diff(self.emg_times)
        positive_diffs = diffs[diffs > 0]
        self.dt_mean = (
            float(np.median(positive_diffs)) if len(positive_diffs) else 0.0
        )

        # ------------------ UI SETUP ------------------
        layout = QVBoxLayout(self)

        theme_row = QHBoxLayout()
        theme_row.addStretch(1)
        self.btn_theme = QPushButton("Switch to Light Mode")
        self.btn_theme.setMinimumHeight(30)
        self.btn_theme.clicked.connect(self.toggle_theme)
        theme_row.addWidget(self.btn_theme)
        layout.addLayout(theme_row)

        # Video label
        self.video_label = QLabel("Video")
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)

        # EMG plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setLabel('left', 'EMG (V)')
        self.plot_widget.setLabel('bottom', 'Time (s)')
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.setMenuEnabled(False)
        layout.addWidget(self.plot_widget)

        # Three curves for channels
        self.curve_ch1 = self.plot_widget.plot(pen='r', name="ch1")
        self.curve_ch2 = self.plot_widget.plot(pen='g', name="ch2")
        self.curve_ch3 = self.plot_widget.plot(pen='b', name="ch3")
        self._setup_audio_overlay()

        # Match video & EMG durations roughly
        self.plot_widget.setXRange(0, self.emg_times[-1], padding=0)
        self._set_audio_y_range()
        self._build_button_regions()

        # ---- Two-label segmentation controls ----
        self.window_size_label = QLabel()
        self.window_size_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.window_size_label)

        label_colors = ("#3b82f6", "#f59e0b")
        for label_idx, color in enumerate(label_colors, start=1):
            row = QHBoxLayout()
            name = QLabel(f"Act {label_idx}")
            name.setMinimumWidth(55)
            name.setStyleSheet(f"font-weight: bold; color: {color};")

            slider = QSlider(Qt.Horizontal)
            slider.setTracking(True)
            slider.setSingleStep(1)
            slider.setPageStep(self.DEFAULT_CLIP_SAMPLES)

            position = QLabel()
            position.setMinimumWidth(245)

            row.addWidget(name)
            row.addWidget(slider, 1)
            row.addWidget(position)
            layout.addLayout(row)

            self.label_sliders.append(slider)
            self.label_position_labels.append(position)
            slider.valueChanged.connect(
                lambda value, idx=label_idx: self._update_label_window(idx, value)
            )

        self.btn_save_both = QPushButton("Save Both Labels")
        self.btn_save_both.setMinimumHeight(38)
        self.btn_save_both.setStyleSheet("font-weight: bold;")
        self.btn_redo = QPushButton("Redo Previous Trial")
        self.btn_redo.setMinimumHeight(38)

        save_row = QHBoxLayout()
        save_row.addWidget(self.btn_save_both, 2)
        save_row.addWidget(self.btn_redo, 1)
        layout.addLayout(save_row)

        self.btn_save_both.clicked.connect(self.save_both_label_segments)
        self.btn_redo.clicked.connect(self.redo_previous_trial)
        self.btn_save_clip = self.btn_save_both  # compatibility alias
        self._configure_label_windows(reset_positions=True)

        # ---- Label showing where the NEXT clip will go ----
        self.next_dest_label = QLabel("")
        layout.addWidget(self.next_dest_label)
        self._update_next_dest_label()

        self.file_mapping_label = QLabel()
        self.file_mapping_label.setObjectName("fileMappingLabel")
        self.file_mapping_label.setWordWrap(True)
        self.file_mapping_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.file_mapping_label.setMinimumHeight(72)
        layout.addWidget(self.file_mapping_label)
        self._update_file_mapping_label()

        # ---- Completed trials, matching First Phase's scrollable row ----
        completed_row = QHBoxLayout()
        self.completed_count_label = QLabel("Finished:")
        completed_row.addWidget(self.completed_count_label)

        self.completed_widget = QWidget()
        self.completed_layout = QHBoxLayout(self.completed_widget)
        self.completed_layout.setContentsMargins(0, 0, 0, 0)
        self.completed_layout.setSpacing(6)

        self.completed_scroll = QScrollArea()
        self.completed_scroll.setWidget(self.completed_widget)
        self.completed_scroll.setWidgetResizable(True)
        self.completed_scroll.setFrameShape(QFrame.NoFrame)
        self.completed_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.completed_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.completed_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.completed_scroll.setFixedHeight(48)
        completed_row.addWidget(self.completed_scroll, 1)
        layout.addLayout(completed_row)
        self._refresh_completed_trials()

        # ---- Playback buttons ----
        btn_layout = QHBoxLayout()
        self.btn_play_pause = QPushButton("Pause")   # playing at start
        self.btn_replay = QPushButton("Replay")
        self.btn_play_act1 = QPushButton("Play Act 1 Segment")
        self.btn_play_act2 = QPushButton("Play Act 2 Segment")
        btn_layout.addWidget(self.btn_play_pause)
        btn_layout.addWidget(self.btn_replay)
        btn_layout.addWidget(self.btn_play_act1)
        btn_layout.addWidget(self.btn_play_act2)
        layout.addLayout(btn_layout)

        self.btn_play_pause.clicked.connect(self.toggle_play_pause)
        self.btn_replay.clicked.connect(self.replay)
        self.btn_play_act1.clicked.connect(lambda: self.play_label_segment(1))
        self.btn_play_act2.clicked.connect(lambda: self.play_label_segment(2))

        playback_row = QHBoxLayout()
        playback_row.addWidget(QLabel("Playback:"))
        self.playback_slider = QSlider(Qt.Horizontal)
        frame_count = max(1, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.playback_slider.setRange(0, frame_count - 1)
        self.playback_slider.setSingleStep(1)
        self.playback_slider.setPageStep(max(1, int(self.fps)))
        self.playback_duration = (frame_count - 1) / self.fps
        playback_row.addWidget(self.playback_slider, 1)

        self.playback_time_label = QLabel(f"0.00 / {self.playback_duration:.2f} s")
        self.playback_time_label.setMinimumWidth(125)
        playback_row.addWidget(self.playback_time_label)
        playback_row.addWidget(QLabel("Speed:"))

        self.speed_combo = QComboBox()
        for text, speed in (
            ("0.25x (very slow)", 0.25),
            ("0.50x (slow)", 0.50),
            ("0.75x", 0.75),
            ("1.00x (normal)", 1.00),
            ("1.50x", 1.50),
            ("2.00x (fast)", 2.00),
            ("4.00x (very fast)", 4.00),
        ):
            self.speed_combo.addItem(text, speed)
        self.speed_combo.setCurrentIndex(3)
        playback_row.addWidget(self.speed_combo)
        self.speed_audio_label = QLabel("Audio on")
        self.speed_audio_label.setMinimumWidth(85)
        playback_row.addWidget(self.speed_audio_label)
        layout.addLayout(playback_row)

        self.playback_slider.sliderPressed.connect(self._playback_seek_started)
        self.playback_slider.sliderMoved.connect(self._seek_playback_frame)
        self.playback_slider.sliderReleased.connect(self._playback_seek_finished)
        self.speed_combo.currentIndexChanged.connect(self._playback_speed_changed)

        self.btn_load_other = QPushButton("Load Another Trial")
        layout.addWidget(self.btn_load_other)
        self.btn_load_other.clicked.connect(self.load_other_trial)

        # Timer for video/plot updates
        self.timer = QTimer(self)
        self.interval_ms = self._playback_interval_ms()
        self.timer.timeout.connect(self._update_frame)
        self._apply_theme(self.theme_name)
        self._begin_playback()

    # ------------------------------------------------------------------
    # Helpers: clip destinations (act1/act2, trial index)

    def _find_trial_index_by_emg_path(self, emg_path):
        target = os.path.abspath(emg_path)
        for index, trial in enumerate(self.available_trials):
            if os.path.abspath(trial["emg_path"]) == target:
                return index
        return None

    def _playback_interval_ms(self):
        return max(1, int(round(1000 / (self.fps * self.PLAYBACK_SPEED))))

    def toggle_theme(self):
        """Switch the complete viewer between dark and light themes."""
        next_theme = "light" if self.theme_name == "dark" else "dark"
        self._apply_theme(next_theme)

    def _apply_theme(self, theme_name):
        """Apply colors to Qt controls and the pyqtgraph plot together."""
        self.theme_name = theme_name
        pal = self.THEMES[theme_name]
        self.setStyleSheet(
            f"""
            QWidget {{
                background-color: {pal['window']};
                color: {pal['text']};
            }}
            QLabel {{
                color: {pal['text']};
                background: transparent;
            }}
            QPushButton {{
                background-color: {pal['panel']};
                color: {pal['text']};
                border: 1px solid {pal['border']};
                border-radius: 6px;
                padding: 4px 10px;
            }}
            QPushButton:hover {{ border-color: {pal['accent']}; }}
            QPushButton:pressed {{ background-color: {pal['accent']}; color: white; }}
            QPushButton:disabled {{ color: {pal['muted']}; }}
            QComboBox {{
                background-color: {pal['panel']};
                color: {pal['text']};
                border: 1px solid {pal['border']};
                border-radius: 5px;
                padding: 4px 8px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {pal['panel']};
                color: {pal['text']};
                selection-background-color: {pal['accent']};
            }}
            QScrollArea {{
                background: transparent;
                border: none;
            }}
            QSlider::groove:horizontal {{
                height: 6px;
                background: {pal['border']};
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {pal['accent']};
                border: 1px solid {pal['text']};
                width: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }}
            """
        )
        self.video_label.setStyleSheet(
            f"background-color: #000000; border: 1px solid {pal['border']};"
        )
        self.file_mapping_label.setStyleSheet(
            f"background-color: {pal['panel']}; color: {pal['text']};"
            f" border: 1px solid {pal['border']}; border-radius: 6px;"
            " padding: 7px; font-weight: 600;"
        )
        self.plot_widget.setBackground(pal["plot_bg"])
        plot_item = self.plot_widget.getPlotItem()
        plot_item.showGrid(x=True, y=True, alpha=pal["grid"])
        for axis_name in ("left", "bottom", "right"):
            axis = plot_item.getAxis(axis_name)
            axis.setPen(pg.mkPen(pal["axis"]))
            axis.setTextPen(pg.mkPen(pal["axis"]))

        self.btn_theme.setText(
            "Switch to Light Mode" if theme_name == "dark"
            else "Switch to Dark Mode"
        )

   # ------------------------------------------------------------------
    # Audio playback (extracted from the video's embedded track)

    def _extract_audio(self):
        """Pull the video's audio track into memory (mono int16) via ffmpeg."""
        self.audio_samples = None
        self.audio_rate = None
        self.audio_player = None
        if not AUDIO_OK or not self.video_path:
            return
        tmp = None
        try:
            ff = imageio_ffmpeg.get_ffmpeg_exe()
            fd, tmp = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            cmd = [ff, "-y", "-i", self.video_path,
                   "-vn", "-ac", "1", "-ar", "44100", "-f", "wav", tmp]
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, creationflags=flags)
            with wave.open(tmp, "rb") as wf:
                self.audio_rate = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
            samples = np.frombuffer(frames, dtype=np.int16)
            if samples.size > 0:
                self.audio_samples = samples
                self.audio_player = _AudioStreamPlayer(samples, self.audio_rate)
        except Exception as e:
            print(f"Audio playback unavailable for {self.video_path}: {e}")
            self.audio_samples = None
            self.audio_rate = None
            self.audio_player = None
        finally:
            if tmp and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    def _start_audio(self, t_s):
        # Only play at 1x; other speeds would change pitch / break sync.
        if self.audio_player is None or abs(self.PLAYBACK_SPEED - 1.0) > 1e-6:
            return
        try:
            self.audio_player.start(int(t_s * self.audio_rate))
        except Exception:
            pass

    def _stop_audio(self):
        if self.audio_player is not None:
            self.audio_player.stop()
        self.audio_master = False

    def _begin_playback(self):
        """(Re)start playback from the current frame. At 1x the audio is the
        master clock so the video and graphs follow the audio's true position
        (staying in sync despite output latency); otherwise a wall clock is
        used and audio is muted."""
        self.is_paused = False
        self.btn_play_pause.setText("Pause")
        self._play_t0 = time.perf_counter()
        self._play_t_origin = self.frame_idx / self.fps
        self._start_audio(self._play_t_origin)
        self.audio_master = (
            abs(self.PLAYBACK_SPEED - 1.0) < 1e-6
            and self.audio_player is not None
            and self.audio_player.active()
        )
        self.timer.start(self.interval_ms)

    def _setup_audio_overlay(self):
        self.audio_view = pg.ViewBox()
        self.plot_widget.showAxis("right")
        self.plot_widget.scene().addItem(self.audio_view)
        self.plot_widget.getAxis("right").linkToView(self.audio_view)
        self.plot_widget.getAxis("right").setLabel("Audio amp")
        self.audio_view.setXLink(self.plot_widget)

        audio_pen = pg.mkPen((255, 220, 0), width=2)
        self.curve_audio = pg.PlotCurveItem(pen=audio_pen, name="audio")
        self.audio_view.addItem(self.curve_audio)

        self.plot_widget.getViewBox().sigResized.connect(self._sync_audio_view)
        self._sync_audio_view()

    def _sync_audio_view(self):
        self.audio_view.setGeometry(self.plot_widget.getViewBox().sceneBoundingRect())
        self.audio_view.linkedViewChanged(
            self.plot_widget.getViewBox(),
            self.audio_view.XAxis,
        )

    def _set_audio_y_range(self):
        if len(self.audio_data) == 0:
            self.curve_audio.clear()
            return

        finite_audio = self.audio_data[np.isfinite(self.audio_data)]
        if len(finite_audio) == 0:
            self.curve_audio.clear()
            return

        y_min = float(np.min(finite_audio))
        y_max = float(np.max(finite_audio))
        if y_min == y_max:
            padding = abs(y_min) * 0.1 or 1.0
            y_min -= padding
            y_max += padding

        self.audio_view.setYRange(y_min, y_max, padding=0.1)

    # ------------------------------------------------------------------
    def _clear_button_regions(self):
        for reg in self.button_regions:
            try:
                self.plot_widget.removeItem(reg)
            except Exception:
                pass
        self.button_regions = []

    def _build_button_regions(self):
        self._clear_button_regions()
        if len(self.emg_times) == 0 or len(self.button_data) == 0:
            return

        n = min(len(self.emg_times), len(self.button_data))
        times = self.emg_times[:n]
        button = np.asarray(self.button_data[:n], dtype=np.int8)

        min_width = self.dt_mean if self.dt_mean > 0 else 0.01
        i = 0
        while i < n:
            if button[i] != 1:
                i += 1
                continue

            start_i = i
            while i + 1 < n and button[i + 1] == 1:
                i += 1
            end_i = i

            t0 = float(times[start_i])
            t1 = float(times[end_i]) + min_width
            if t1 <= t0:
                t1 = t0 + min_width

            reg = pg.LinearRegionItem(
                values=(t0, t1),
                movable=False,
                brush=(255, 0, 0, 35),
                pen=(255, 0, 0, 110),
            )
            reg.setZValue(-20)
            self.plot_widget.addItem(reg)
            self.button_regions.append(reg)
            i += 1

    def _compute_next_destination_for_root(self, result_root):
        return compute_next_destination_for_root(result_root, self.num_acts)

    def _compute_next_destination(self):
        """
        Backwards-compatible helper for the original ResultClip folder.
        """
        base_dir = os.path.dirname(self.emg_path)
        result_root = os.path.join(base_dir, "ResultClip")
        return self._compute_next_destination_for_root(result_root)

    def _update_next_dest_label(self):
        """Describe the paired-label destination for the next save."""
        if self.redo_output_trial_idx is not None:
            self.next_dest_label.setText(
                f"Redo mode: saving will replace both labels for "
                f"trial_{self.redo_output_trial_idx}."
            )
            if hasattr(self, "file_mapping_label"):
                self._update_file_mapping_label()
            return

        act_idx, trial_idx = self._compute_next_destination()
        if self._trial_for_output_index(trial_idx) is None:
            self.next_dest_label.setText(
                "All available trials are labeled. Select a finished trial to redo it."
            )
            if hasattr(self, "file_mapping_label"):
                self._update_file_mapping_label()
            return

        if act_idx == 1:
            text = (
                f"Ready: both labels will be saved as trial_{trial_idx} "
                "in act1 and act2."
            )
        else:
            text = (
                f"Recovery mode: act1/trial_{trial_idx} already exists; "
                "the Act 2 selection will complete the pair."
            )
        self.next_dest_label.setText(text)
        if hasattr(self, "file_mapping_label"):
            self._update_file_mapping_label()

    def _update_file_mapping_label(self):
        """Show the exact source files and paired Act 1/Act 2 destinations."""
        if self.redo_output_trial_idx is not None:
            output_trial_idx = self.redo_output_trial_idx
            redo_mode = True
        else:
            _next_act, output_trial_idx = self._compute_next_destination()
            redo_mode = False
        has_pending_source = (
            redo_mode or self._trial_for_output_index(output_trial_idx) is not None
        )

        base_dir = os.path.dirname(self.emg_path)
        result_root = os.path.join(base_dir, "ResultClip")
        act_paths = {
            act_idx: os.path.join(
                result_root,
                f"act{act_idx}",
                f"trial_{output_trial_idx}.txt",
            )
            for act_idx in range(1, self.num_acts + 1)
        }

        source_trial_num = "?"
        if (
            self.current_trial_index is not None
            and 0 <= self.current_trial_index < len(self.available_trials)
        ):
            source_trial_num = self.available_trials[self.current_trial_index]["trial_num"]

        def destination_status(path):
            if redo_mode:
                return "REPLACE"
            if not has_pending_source:
                return "NO PENDING SAVE"
            return "SAVED" if os.path.exists(path) else "NEXT SAVE"

        session_name = os.path.basename(os.path.normpath(base_dir))
        source_files = [os.path.basename(self.emg_path), os.path.basename(self.video_path)]
        if self.audio_path:
            source_files.append(os.path.basename(self.audio_path))

        if has_pending_source:
            act1_relative = os.path.relpath(act_paths[1], base_dir)
            act2_relative = os.path.relpath(act_paths[2], base_dir)
        else:
            act1_relative = "--"
            act2_relative = "--"
        self.file_mapping_label.setText(
            f"USING SOURCE TRIAL {source_trial_num}  |  Session: {session_name}\n"
            f"Source files: {'  +  '.join(source_files)}\n"
            f"Act 1 [{destination_status(act_paths[1])}]: {act1_relative}\n"
            f"Act 2 [{destination_status(act_paths[2])}]: {act2_relative}"
        )
        self.file_mapping_label.setToolTip(
            f"Source EMG: {self.emg_path}\n"
            f"Source video: {self.video_path}\n"
            f"Source audio: {self.audio_path or 'none'}\n"
            f"Act 1 output: {act_paths[1]}\n"
            f"Act 2 output: {act_paths[2]}"
        )

    def _refresh_completed_trials(self):
        """Rebuild the First Phase-style row of fully labeled trials."""
        while self.completed_layout.count():
            item = self.completed_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        completed = completed_result_trial_numbers(self.folder, self.num_acts)
        total = len(self.available_trials)
        self.completed_count_label.setText(f"Finished: {len(completed)} / {total}")
        self.completed_trial_buttons = []

        if not completed:
            empty = QLabel("No labeled trials yet")
            empty.setStyleSheet("color: #777;")
            self.completed_layout.addWidget(empty)
        else:
            for trial_idx in completed:
                button = QPushButton(f"Trial {trial_idx} done")
                button.setToolTip("Click to load this completed trial for redo")
                button.setStyleSheet(
                    "QPushButton { border: 1px solid #27ae60; border-radius: 6px;"
                    " padding: 3px 8px; color: #1f7a42; font-weight: bold; }"
                    "QPushButton:hover { background: #eafaf0; }"
                )
                button.clicked.connect(
                    lambda _checked=False, n=trial_idx: self._enter_redo_mode(n)
                )
                self.completed_layout.addWidget(button)
                self.completed_trial_buttons.append(button)

        self.completed_layout.addStretch(1)
        self.btn_redo.setEnabled(bool(completed))
        available_numbers = {trial["trial_num"] for trial in self.available_trials}
        all_finished = bool(available_numbers) and available_numbers.issubset(completed)
        self.btn_save_both.setEnabled(
            self.redo_output_trial_idx is not None or not all_finished
        )

    def _trial_for_output_index(self, output_trial_idx):
        """Find the source recording corresponding to an output trial number."""
        history = self.selection_history.get(output_trial_idx)
        if history is not None:
            source_index = history.get("source_index")
            if source_index is not None and 0 <= source_index < len(self.available_trials):
                return self.available_trials[source_index]

        for trial in self.available_trials:
            if trial["trial_num"] == output_trial_idx:
                return trial
        return None

    def _enter_redo_mode(self, output_trial_idx):
        """Load a completed source trial and prepare its outputs for replacement."""
        if output_trial_idx not in completed_result_trial_numbers(
            self.folder, self.num_acts
        ):
            print(f"Trial {output_trial_idx} is not fully labeled yet.")
            self._refresh_completed_trials()
            return

        selected_trial = self._trial_for_output_index(output_trial_idx)
        if selected_trial is None:
            print(f"Could not find source recording for trial {output_trial_idx}.")
            return

        self.redo_output_trial_idx = output_trial_idx
        if not self._load_trial(selected_trial):
            self.redo_output_trial_idx = None
            self._update_next_dest_label()
            return

        history = self.selection_history.get(output_trial_idx)
        if history is not None:
            for slider, start_idx in zip(
                self.label_sliders,
                history.get("starts", []),
            ):
                slider.setValue(start_idx)

        self.btn_save_both.setText("Replace Both Labels")
        self._update_next_dest_label()

    def redo_previous_trial(self):
        """Open the most recently finished trial for correction."""
        completed = completed_result_trial_numbers(self.folder, self.num_acts)
        if not completed:
            print("There is no finished trial to redo.")
            return
        self._enter_redo_mode(completed[-1])

    # ------------------------ EMG save helper ----------------------------
    def _save_clip_to_root(
        self,
        root_name,
        idx,
        target_emg_samples=None,
        destination=None,
        overwrite=False,
    ):
        """Save an EMG clip or resampled EMG variant without copying A/V."""
        if len(idx) == 0:
            print(f"[{root_name}] No samples to save.")
            return
        if overwrite and destination is None:
            raise ValueError("overwrite requires an explicit destination")

        base_dir = os.path.dirname(self.emg_path)
        result_root = os.path.join(base_dir, root_name)

        times_clip_abs = self.emg_times[idx].copy()
        emg_clip = self.emg_data[idx, :].copy()
        if target_emg_samples is not None:
            times_clip_abs, emg_clip = resample_emg_clip(
                times_clip_abs,
                emg_clip,
                target_emg_samples,
            )

        times_clip = times_clip_abs - times_clip_abs[0]

        while True:
            if destination is None:
                act_idx, trial_idx = self._compute_next_destination_for_root(result_root)
            else:
                act_idx, trial_idx = destination
            act_dir = os.path.join(result_root, f"act{act_idx}")
            os.makedirs(act_dir, exist_ok=True)
            emg_out_path = os.path.join(act_dir, f"trial_{trial_idx}.txt")

            temp_path = None
            try:
                if overwrite:
                    fd, temp_path = tempfile.mkstemp(
                        prefix=f".trial_{trial_idx}_",
                        suffix=".tmp",
                        dir=act_dir,
                    )
                    os.close(fd)
                    write_path = temp_path
                    mode = "w"
                else:
                    write_path = emg_out_path
                    mode = "x"

                with open(write_path, mode) as f:
                    f.write("timestamp\tch1\tch2\tch3\n")
                    for t_sec, (c1, c2, c3) in zip(times_clip, emg_clip):
                        ts_formatted = format_time_from_seconds(t_sec)
                        f.write(
                            f"{ts_formatted}\t{c1:.6f}\t{c2:.6f}\t{c3:.6f}\n"
                        )
                if overwrite:
                    os.replace(temp_path, emg_out_path)
                    temp_path = None
                break
            except FileExistsError:
                if destination is not None:
                    print(f"[{root_name}] Keeping existing file: {emg_out_path}")
                    return None
                # Another save won the automatic destination race; retry.
                continue
            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass

        action = "Replaced" if overwrite else "Saved"
        print(
            f"[{root_name}] {action} EMG clip with {len(times_clip)} samples at: "
            f"{emg_out_path}"
        )

        return {
            "act_idx": act_idx,
            "trial_idx": trial_idx,
            "emg_out_path": emg_out_path,
        }

    # --------------- Clip window logic -----------------
    def _load_trial(self, selected_trial):
        """Reload viewer with a selected trial, reusing this window."""
        n = selected_trial["trial_num"]
        new_emg_path = selected_trial["emg_path"]
        new_video_path = selected_trial["video_path"]
        new_audio_path = selected_trial["audio_path"]

        # Validate every new resource before replacing the working trial.
        try:
            new_times, new_data, new_button = load_emg_file(new_emg_path)
            new_audio_times, new_audio_data = load_audio_file(new_audio_path)
        except Exception as e:
            print(f"Could not load trial {n}: {e}")
            return False

        new_cap = cv2.VideoCapture(new_video_path)
        if not new_cap.isOpened():
            new_cap.release()
            print("Could not open new video:", new_video_path)
            return False

        new_fps = new_cap.get(cv2.CAP_PROP_FPS)
        if new_fps <= 0:
            new_fps = 30.0

        print(f"Loading trial {n}:")
        print("  EMG  :", new_emg_path)
        print("  Video:", new_video_path)
        print("  Audio:", new_audio_path)

        self.timer.stop()
        self._stop_audio()
        old_cap = self.cap

        self.emg_times = new_times
        self.emg_data = new_data
        self.button_data = new_button
        self.emg_path = new_emg_path
        self.audio_path = new_audio_path
        self.audio_times = new_audio_times
        self.audio_data = new_audio_data
        self.folder = os.path.dirname(self.emg_path)
        self.video_path = new_video_path
        self.cap = new_cap
        self.fps = new_fps

        if old_cap is not None:
            old_cap.release()

        diffs = np.diff(self.emg_times)
        positive_diffs = diffs[diffs > 0]
        self.dt_mean = (
            float(np.median(positive_diffs)) if len(positive_diffs) else 0.0
        )

        self.curve_ch1.clear()
        self.curve_ch2.clear()
        self.curve_ch3.clear()
        self.curve_audio.clear()
        self._clear_button_regions()

        for region in self.label_regions:
            self.plot_widget.removeItem(region)
        self.label_regions = []
        self.region = None

        self.plot_widget.setXRange(0, self.emg_times[-1], padding=0)
        self._set_audio_y_range()
        self._build_button_regions()

        self._extract_audio()
        self.interval_ms = self._playback_interval_ms()
        self.frame_idx = 0
        self.segment_stop_time = None
        frame_count = max(1, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.playback_duration = (frame_count - 1) / self.fps
        self.playback_slider.blockSignals(True)
        self.playback_slider.setRange(0, frame_count - 1)
        self.playback_slider.setValue(0)
        self.playback_slider.blockSignals(False)
        self.playback_time_label.setText(
            f"0.00 / {self.playback_duration:.2f} s"
        )

        self._configure_label_windows(reset_positions=True)

        # Update label in case emg_path changed
        self.current_trial_index = self._find_trial_index_by_emg_path(self.emg_path)
        self._refresh_completed_trials()
        self._update_next_dest_label()
        self.setWindowTitle(f"Trial {n} - EMG + Video")
        self._begin_playback()
        return True

    def load_next_trial(self):
        """Immediately load the next unfinished source trial after a paired save."""
        _next_act, next_output_trial = self._compute_next_destination()
        selected_trial = self._trial_for_output_index(next_output_trial)
        if selected_trial is None:
            print("All available trials have been labeled.")
            return False

        print(
            f"Both label clips saved. Loading trial "
            f"{selected_trial['trial_num']} automatically."
        )
        return self._load_trial(selected_trial)

    def load_other_trial(self):
        """Reload viewer with another trial in the same folder, reusing this window."""
        was_paused = self.is_paused
        self.timer.stop()
        self._stop_audio()
        self.is_paused = True
        self.btn_play_pause.setText("Play")

        folder = self.folder  # same folder as current EMG/Video

        trials = find_trials_in_folder(folder)
        trials_by_num = {trial["trial_num"]: trial for trial in trials}
        common = sorted(trials_by_num)
        if not common:
            print("No more trials found.")
            if not was_paused:
                self._begin_playback()
            return

        n, ok = QInputDialog.getInt(
            self,
            "Load Trial",
            f"Available trials: {common}\nEnter trial number:",
            min(common),
            min(common),
            max(common),
        )

        if not ok or n not in common:
            if not was_paused:
                self._begin_playback()
            return

        self.redo_output_trial_idx = None
        self.btn_save_both.setText("Save Both Labels")
        if not self._load_trial(trials_by_num[n]) and not was_paused:
            self._begin_playback()

    def _configure_label_windows(self, reset_positions=False):
        """Configure the two bottom sliders and their plot highlights."""
        self.clip_samples = min(self.DEFAULT_CLIP_SAMPLES, len(self.emg_times))
        max_start = max(0, len(self.emg_times) - self.clip_samples)
        self.window_size_label.setText(
            f"Segmentation window: {self.clip_samples} samples (fixed)"
        )

        region_styles = (
            ((59, 130, 246, 65), (59, 130, 246, 220)),
            ((245, 158, 11, 65), (245, 158, 11, 220)),
        )
        while len(self.label_regions) < self.num_acts:
            brush, pen = region_styles[len(self.label_regions)]
            region = pg.LinearRegionItem(
                movable=False,
                brush=brush,
                pen=pen,
            )
            region.setZValue(-10 + len(self.label_regions))
            self.plot_widget.addItem(region)
            self.label_regions.append(region)

        self.region = self.label_regions[0]

        for zero_based, slider in enumerate(self.label_sliders):
            slider.blockSignals(True)
            slider.setRange(0, max_start)
            if reset_positions:
                # Put Act 1 at the beginning and Act 2 at the end so both
                # colored windows are immediately visible and independent.
                start_idx = 0 if zero_based == 0 else max_start
                slider.setValue(start_idx)
            else:
                start_idx = min(slider.value(), max_start)
                slider.setValue(start_idx)
            slider.blockSignals(False)
            self._update_label_window(zero_based + 1, start_idx)

    def _update_label_window(self, label_idx, start_idx):
        """Move one label window using a sample-index slider value."""
        if not self.label_regions or len(self.emg_times) == 0:
            return

        zero_based = label_idx - 1
        max_start = max(0, len(self.emg_times) - self.clip_samples)
        start_idx = max(0, min(int(start_idx), max_start))
        end_idx = min(start_idx + self.clip_samples, len(self.emg_times))

        t_start = float(self.emg_times[start_idx])
        t_end = float(self.emg_times[end_idx - 1])
        self.label_regions[zero_based].setBounds(
            [float(self.emg_times[0]), float(self.emg_times[-1])]
        )
        self.label_regions[zero_based].setRegion((t_start, t_end))
        self.label_position_labels[zero_based].setText(
            f"samples {start_idx + 1}-{end_idx}  |  {t_start:.3f}-{t_end:.3f} s"
        )

    def _indices_for_label(self, label_idx):
        start_idx = self.label_sliders[label_idx - 1].value()
        end_idx = min(start_idx + self.clip_samples, len(self.emg_times))
        return np.arange(start_idx, end_idx)

    def set_clip_window(self):
        """Compatibility helper: refresh both fixed 500-sample windows."""
        self._configure_label_windows(reset_positions=False)

    def _save_label_outputs(self, act_idx, trial_idx, idx, overwrite=False):
        """Save one explicitly labeled window and all of its resampled forms."""
        sizes = [
            200, 250, 300, 350, 400,
            450, 500, 550, 600, 650,
            700, 750, 800, 900, 1000,
        ]
        destination = (act_idx, trial_idx)

        # Save derived files first and the base ResultClip last. The base file
        # remains the completion marker used by recovery and startup logic.
        if self.clip_samples == self.DEFAULT_CLIP_SAMPLES:
            for size in sizes:
                self._save_clip_to_root(
                    f"ResultClipSizeUp{size}",
                    idx,
                    target_emg_samples=size,
                    destination=destination,
                    overwrite=overwrite,
                )

        return self._save_clip_to_root(
            "ResultClip",
            idx,
            destination=destination,
            overwrite=overwrite,
        )

    def _save_both_label_segments(self):
        """Save Act 1 and Act 2 selections together as one paired trial."""
        if len(self.label_regions) != self.num_acts:
            print("The two label windows are not ready.")
            return

        is_redo = self.redo_output_trial_idx is not None
        if is_redo:
            trial_idx = self.redo_output_trial_idx
            first_act = 1
        else:
            first_act, trial_idx = self._compute_next_destination()

        acts_to_save = range(first_act, self.num_acts + 1)
        starts = [slider.value() for slider in self.label_sliders]
        source_index = self.current_trial_index
        final_save = None

        for act_idx in acts_to_save:
            idx = self._indices_for_label(act_idx)
            if len(idx) != self.clip_samples:
                print(f"Act {act_idx} does not contain {self.clip_samples} samples.")
                return
            final_save = self._save_label_outputs(
                act_idx,
                trial_idx,
                idx,
                overwrite=is_redo,
            )
            if final_save is None:
                print(f"Act {act_idx} was not saved; staying on the current trial.")
                self._update_next_dest_label()
                return

        if final_save and final_save["act_idx"] == self.num_acts:
            if is_redo or first_act == 1:
                self.selection_history[trial_idx] = {
                    "source_index": source_index,
                    "starts": starts,
                }

            if is_redo:
                self.redo_output_trial_idx = None
                print(f"Trial {trial_idx} labels were replaced successfully.")

            self._refresh_completed_trials()
            if not self.load_next_trial():
                self._update_next_dest_label()
        else:
            self._update_next_dest_label()

    def save_both_label_segments(self):
        """Guard the paired save against accidental repeated clicks."""
        if not self.btn_save_both.isEnabled():
            return

        if self.redo_output_trial_idx is not None:
            answer = QMessageBox.question(
                self,
                "Replace saved labels?",
                f"Replace both saved labels for trial "
                f"{self.redo_output_trial_idx}?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        self.btn_save_both.setEnabled(False)
        self.btn_save_both.setText("Saving Both Labels...")
        QApplication.processEvents()
        try:
            self._save_both_label_segments()
        finally:
            button_text = (
                "Replace Both Labels"
                if self.redo_output_trial_idx is not None
                else "Save Both Labels"
            )
            self.btn_save_both.setText(button_text)
            _next_act, next_output_trial = self._compute_next_destination()
            has_pending_source = (
                self._trial_for_output_index(next_output_trial) is not None
            )
            self.btn_save_both.setEnabled(
                self.redo_output_trial_idx is not None or has_pending_source
            )

    def save_clip_segment(self):
        """Compatibility alias for the new paired-label save action."""
        self.save_both_label_segments()


    # --------------- Playback controls -----------------
    def toggle_play_pause(self):
        if self.is_paused:
            self._begin_playback()
        else:
            self._pause_playback()

    def _pause_playback(self):
        self.timer.stop()
        self._stop_audio()
        self.is_paused = True
        self.btn_play_pause.setText("Play")

    def _playback_speed_changed(self, _index=None):
        """Apply a new synchronized video/plot playback speed."""
        speed = float(self.speed_combo.currentData())
        was_playing = not self.is_paused
        self.timer.stop()
        self._stop_audio()
        self.PLAYBACK_SPEED = speed
        self.interval_ms = self._playback_interval_ms()
        self.speed_audio_label.setText("Audio on" if speed == 1.0 else "Audio muted")
        if was_playing:
            self._begin_playback()

    def _playback_seek_started(self):
        self.was_playing_before_seek = not self.is_paused
        self._pause_playback()

    def _playback_seek_finished(self):
        frame_index = self.playback_slider.value()
        should_resume = self.was_playing_before_seek
        self.was_playing_before_seek = False
        self.segment_stop_time = None
        self._seek_playback_frame(frame_index)
        if should_resume:
            self._begin_playback()

    def _seek_playback_frame(self, frame_index):
        """Seek video and both plots to a timeline-slider frame."""
        if self.cap is None:
            return
        frame_count = max(1, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        frame_index = max(0, min(int(frame_index), frame_count - 1))

        self.timer.stop()
        self._stop_audio()
        self.is_paused = True
        self.btn_play_pause.setText("Play")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        if not ret:
            return

        self.frame_idx = frame_index
        current_time_sec = frame_index / self.fps
        self._render_frame_and_signals(frame, current_time_sec)

    def play_label_segment(self, label_idx):
        """Replay one selected 500-sample label window at the chosen speed."""
        idx = self._indices_for_label(label_idx)
        if len(idx) == 0:
            return
        start_time = float(self.emg_times[idx[0]])
        end_time = float(self.emg_times[idx[-1]])
        start_frame = int(round(start_time * self.fps))

        self.segment_stop_time = None
        self._seek_playback_frame(start_frame)
        self.segment_stop_time = end_time
        self._begin_playback()

    def replay(self):
        self.segment_stop_time = None
        self._seek_playback_frame(0)
        self._begin_playback()

    def _render_frame_and_signals(self, frame, current_time_sec):
        """Render one video frame and reveal plot data through the same time."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

        idx = np.searchsorted(self.emg_times, current_time_sec, side="right")
        if idx > 0:
            self.curve_ch1.setData(self.emg_times[:idx], self.emg_data[:idx, 0])
            self.curve_ch2.setData(self.emg_times[:idx], self.emg_data[:idx, 1])
            self.curve_ch3.setData(self.emg_times[:idx], self.emg_data[:idx, 2])
        else:
            self.curve_ch1.clear()
            self.curve_ch2.clear()
            self.curve_ch3.clear()

        audio_idx = np.searchsorted(self.audio_times, current_time_sec, side="right")
        if audio_idx > 0:
            self.curve_audio.setData(
                self.audio_times[:audio_idx],
                self.audio_data[:audio_idx],
            )
        else:
            self.curve_audio.clear()

        slider_frame = int(round(current_time_sec * self.fps))
        self.playback_slider.blockSignals(True)
        self.playback_slider.setValue(slider_frame)
        self.playback_slider.blockSignals(False)
        shown_time = max(0.0, min(current_time_sec, self.playback_duration))
        self.playback_time_label.setText(
            f"{shown_time:.2f} / {self.playback_duration:.2f} s"
        )

    # --------------- Frame update ----------------------
    def _update_frame(self):
        if self.audio_master and self.audio_player.active():
            current_time_sec = self.audio_player.position_s()
        else:
            elapsed = time.perf_counter() - self._play_t0
            current_time_sec = self._play_t_origin + elapsed * self.PLAYBACK_SPEED

        if (
            self.segment_stop_time is not None
            and current_time_sec >= self.segment_stop_time
        ):
            stop_time = self.segment_stop_time
            self.segment_stop_time = None
            self._seek_playback_frame(int(round(stop_time * self.fps)))
            return

        target_frame = max(0, int(current_time_sec * self.fps))
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 0 and target_frame >= frame_count:
            self.timer.stop()
            self._stop_audio()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_idx = 0
            self.is_paused = True
            self.btn_play_pause.setText("Play")
            self.segment_stop_time = None
            self.playback_slider.setValue(0)
            self.playback_time_label.setText(
                f"0.00 / {self.playback_duration:.2f} s"
            )
            return

        next_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        if abs(target_frame - next_frame) > 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self._stop_audio()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_idx = 0
            self.is_paused = True
            self.btn_play_pause.setText("Play")
            self.segment_stop_time = None
            self.playback_slider.setValue(0)
            self.playback_time_label.setText(
                f"0.00 / {self.playback_duration:.2f} s"
            )
            return

        self.frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self._render_frame_and_signals(frame, current_time_sec)

    def closeEvent(self, event):
        self.timer.stop()
        self._stop_audio()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        super().closeEvent(event)


# ----------------------------- MAIN SCRIPT ----------------------------------

def main():
    # Ask for folder path
    folder = input("Enter path to folder (e.g. ...\\finalemg\\set1\\act1): ").strip()

    if not folder:
        print("No folder given.")
        return
    if not os.path.isdir(folder):
        print("Folder does not exist.")
        return

    trials = find_trials_in_folder(folder)
    trials_by_num = {trial["trial_num"]: trial for trial in trials}
    common_trials = sorted(trials_by_num)
    if not common_trials:
        print("No matching trial_X.txt and video_X.avi found.")
        return

    print("Available trials:", common_trials)
    while True:
        try:
            n = int(input("Which trial number do you want to view? "))
        except ValueError:
            print("Please enter a valid integer.")
            continue
        if n not in common_trials:
            print("That trial does not exist. Choose one of:", common_trials)
        else:
            break

    selected_trial = trials_by_num[n]
    emg_path = selected_trial["emg_path"]
    video_path = selected_trial["video_path"]
    audio_path = selected_trial["audio_path"]
    print(f"Using EMG file:   {emg_path}")
    print(f"Using VIDEO file: {video_path}")
    print(f"Using AUDIO file: {audio_path}")

    # Load EMG data
    emg_times, emg_data, button_data = load_emg_file(emg_path)

    # Run Qt app
    app = QApplication(sys.argv)
    viewer = EMGVideoViewer(
        video_path,
        emg_times,
        emg_data,
        button_data,
        emg_path,
        audio_path=audio_path,
        available_trials=trials,
        current_trial_index=trials.index(selected_trial),
    )
    viewer.setWindowTitle(f"Trial {n} - EMG + Video")
    viewer.resize(1100, 950)
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
