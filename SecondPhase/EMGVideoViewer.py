import sys
import os
import glob
import csv
import cv2
import numpy as np
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout,
    QPushButton, QHBoxLayout, QLineEdit, QInputDialog
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

import pyqtgraph as pg


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

    common = sorted(set(txt_nums) & set(vid_nums) & set(audio_nums))
    return [
        {
            "trial_num": n,
            "folder": folder,
            "emg_path": txt_nums[n],
            "video_path": vid_nums[n],
            "audio_path": audio_nums[n],
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

    if not ch1:
        raise ValueError(f"No data found in {emg_path}")

    if saw_new and len(timestamps_ns) == len(ch1):
        t_ns_unwrapped = _unwrap_monotonic_ns(timestamps_ns)
        times_sec = (t_ns_unwrapped - t_ns_unwrapped[0]).astype(np.float64) / 1e9
        if any(button_inline):
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
    ms = int(round((s - int(s)) * 1000))
    total_sec = int(s)
    h = total_sec // 3600
    m = (total_sec % 3600) // 60
    sec = total_sec % 60
    return f"{h:02d}:{m:02d}:{sec:02d}.{ms:03d}"


# ----------------------------- VIEWER CLASS ---------------------------------

class EMGVideoViewer(QWidget):
    PLAYBACK_SPEED = 1.0  # 1.0 = full speed

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

        # For clipping window
        self.clip_samples = None
        self.region = None
        self.button_regions = []
        if len(self.emg_times) > 1:
            self.dt_mean = float(np.mean(np.diff(self.emg_times)))
        else:
            self.dt_mean = 0.0

        # ------------------ UI SETUP ------------------
        layout = QVBoxLayout(self)

        # Video label
        self.video_label = QLabel("Video")
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)

        # EMG plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setLabel('left', 'EMG (V)')
        self.plot_widget.setLabel('bottom', 'Time (s)')
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

        # ---- Clip controls (samples window) ----
        clip_layout = QHBoxLayout()
        self.samples_input = QLineEdit()
        self.samples_input.setPlaceholderText("Samples (e.g. 300)")
        self.btn_set_window = QPushButton("Set Window")
        self.btn_save_clip = QPushButton("Save Clip")

        clip_layout.addWidget(QLabel("Clip samples:"))
        clip_layout.addWidget(self.samples_input)
        clip_layout.addWidget(self.btn_set_window)
        clip_layout.addWidget(self.btn_save_clip)
        layout.addLayout(clip_layout)

        self.btn_set_window.clicked.connect(self.set_clip_window)
        self.btn_save_clip.clicked.connect(self.save_clip_segment)

        # ---- Label showing where the NEXT clip will go ----
        self.next_dest_label = QLabel("")
        layout.addWidget(self.next_dest_label)
        self._update_next_dest_label()

        # ---- Playback buttons ----
        btn_layout = QHBoxLayout()
        self.btn_play_pause = QPushButton("Pause")   # playing at start
        self.btn_replay = QPushButton("Replay")
        btn_layout.addWidget(self.btn_play_pause)
        btn_layout.addWidget(self.btn_replay)
        layout.addLayout(btn_layout)

        self.btn_play_pause.clicked.connect(self.toggle_play_pause)
        self.btn_replay.clicked.connect(self.replay)

        self.btn_load_other = QPushButton("Load Another Trial")
        layout.addWidget(self.btn_load_other)
        self.btn_load_other.clicked.connect(self.load_other_trial)

        # Timer for video/plot updates
        self.timer = QTimer(self)
        self.interval_ms = self._playback_interval_ms()
        self.timer.timeout.connect(self._update_frame)
        self.timer.start(self.interval_ms)

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
        """
        Same logic as before, but parameterized by result_root.
        Count how many trial_*.txt exist under result_root/act1, act2, ...
        Then alternate: act1/trial1, act2/trial1, act1/trial2, ...
        """
        num_acts = self.num_acts
        total_clips = 0

        if os.path.isdir(result_root):
            for act_idx in range(1, num_acts + 1):
                act_dir = os.path.join(result_root, f"act{act_idx}")
                if not os.path.isdir(act_dir):
                    continue
                existing_txt = glob.glob(os.path.join(act_dir, "trial_*.txt"))
                total_clips += len(existing_txt)

        act_idx = (total_clips % num_acts) + 1
        trial_idx = (total_clips // num_acts) + 1
        return act_idx, trial_idx

    def _compute_next_destination(self):
        """
        Backwards-compatible helper for the original ResultClip folder.
        """
        base_dir = os.path.dirname(self.emg_path)
        result_root = os.path.join(base_dir, "ResultClip")
        return self._compute_next_destination_for_root(result_root)

    def _update_next_dest_label(self):
        """Update the GUI text that tells where the next clip will go (ResultClip)."""
        act_idx, trial_idx = self._compute_next_destination()
        self.next_dest_label.setText(
            f"Next clip will be saved to: act{act_idx} / trial_{trial_idx}"
        )

    # ------------------ Core save helper (EMG + video) ------------------
    def _save_clip_to_root(self, root_name, idx):
        """
        Save EMG + video for the given indices `idx` into a root folder
        (e.g. 'ResultClip', 'ResultClipSizeUp700', ...).
        """
        if len(idx) == 0:
            print(f"[{root_name}] No samples to save.")
            return

        base_dir = os.path.dirname(self.emg_path)
        result_root = os.path.join(base_dir, root_name)

        # absolute times for these samples
        times_clip_abs = self.emg_times[idx].copy()
        emg_clip = self.emg_data[idx, :].copy()

        t_clip_start = float(times_clip_abs[0])
        t_clip_end = float(times_clip_abs[-1])

        # re-zero for writing timestamps
        times_clip = times_clip_abs - times_clip_abs[0]

        # Decide act/trial index for THIS root
        act_idx, trial_idx = self._compute_next_destination_for_root(result_root)
        act_dir = os.path.join(result_root, f"act{act_idx}")
        os.makedirs(act_dir, exist_ok=True)

        emg_out_path = os.path.join(act_dir, f"trial_{trial_idx}.txt")
        video_out_path = os.path.join(act_dir, f"video_{trial_idx}.avi")

        # ---------- Save EMG TXT ----------
        with open(emg_out_path, "w") as f:
            f.write("timestamp\tch1\tch2\tch3\n")
            for t_sec, (c1, c2, c3) in zip(times_clip, emg_clip):
                ts_formatted = format_time_from_seconds(t_sec)
                f.write(f"{ts_formatted}\t{c1:.6f}\t{c2:.6f}\t{c3:.6f}\n")

        print(
            f"[{root_name}] Saved EMG clip with {len(times_clip)} samples to: "
            f"{emg_out_path}"
        )

        # ---------- Save VIDEO CLIP ----------
        cap_clip = cv2.VideoCapture(self.video_path)
        if not cap_clip.isOpened():
            print(f"[{root_name}] Could not open video again for clipping.")
            return

        fps = cap_clip.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = self.fps

        width = int(cap_clip.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_clip.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap_clip.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = int(round(t_clip_start * fps))
        end_frame = int(round(t_clip_end * fps))

        start_frame = max(0, min(start_frame, frame_count - 1))
        end_frame = max(start_frame, min(end_frame, frame_count - 1))

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))

        cap_clip.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_idx in range(start_frame, end_frame + 1):
            ret, frame = cap_clip.read()
            if not ret:
                break
            writer.write(frame)

        writer.release()
        cap_clip.release()

        print(
            f"[{root_name}] Saved VIDEO clip frames {start_frame}–{end_frame} to: "
            f"{video_out_path}"
        )

        return {
            "act_idx": act_idx,
            "trial_idx": trial_idx,
            "emg_out_path": emg_out_path,
            "video_out_path": video_out_path,
        }

    # --------------- Clip window logic -----------------
    def _load_trial(self, selected_trial):
        """Reload viewer with a selected trial, reusing this window."""
        self.timer.stop()
        self.is_paused = True
        self.btn_play_pause.setText("Play")

        n = selected_trial["trial_num"]
        new_emg_path = selected_trial["emg_path"]
        new_video_path = selected_trial["video_path"]
        new_audio_path = selected_trial["audio_path"]
        print(f"Loading trial {n}:")
        print("  EMG  :", new_emg_path)
        print("  Video:", new_video_path)
        print("  Audio:", new_audio_path)

        new_times, new_data, new_button = load_emg_file(new_emg_path)

        self.emg_times = new_times
        self.emg_data = new_data
        self.button_data = new_button
        self.emg_path = new_emg_path
        self.audio_path = new_audio_path
        self.audio_times, self.audio_data = load_audio_file(self.audio_path)
        self.folder = os.path.dirname(self.emg_path)

        if len(self.emg_times) > 1:
            self.dt_mean = float(np.mean(np.diff(self.emg_times)))
        else:
            self.dt_mean = 0.0

        self.curve_ch1.clear()
        self.curve_ch2.clear()
        self.curve_ch3.clear()
        self.curve_audio.clear()
        self._clear_button_regions()

        if self.region is not None:
            self.plot_widget.removeItem(self.region)
            self.region = None

        self.plot_widget.setXRange(0, self.emg_times[-1], padding=0)
        self._set_audio_y_range()
        self._build_button_regions()

        if self.cap is not None:
            self.cap.release()

        self.video_path = new_video_path
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print("Could not open new video:", self.video_path)
            return

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30.0

        self.interval_ms = self._playback_interval_ms()
        self.frame_idx = 0

        self.is_paused = False
        self.btn_play_pause.setText("Pause")
        self.timer.start(self.interval_ms)

        if self.clip_samples is not None:
            self.samples_input.setText(str(self.clip_samples))
            self.set_clip_window()

        # Update label in case emg_path changed
        self._update_next_dest_label()
        self.current_trial_index = self._find_trial_index_by_emg_path(self.emg_path)
        self.setWindowTitle(f"Trial {n} - EMG + Video")
        return True

    def load_next_trial(self):
        """Load the next discovered trial after finishing act2."""
        if self.current_trial_index is None:
            self.current_trial_index = self._find_trial_index_by_emg_path(self.emg_path)

        if self.current_trial_index is None:
            print("Could not determine the current trial, so the next trial was not loaded.")
            return False

        next_index = self.current_trial_index + 1
        if next_index >= len(self.available_trials):
            print("No next trial found.")
            return False

        print("Act 2 clip saved. Loading next trial automatically.")
        return self._load_trial(self.available_trials[next_index])

    def load_other_trial(self):
        """Reload viewer with another trial in the same folder, reusing this window."""
        self.timer.stop()
        self.is_paused = True
        self.btn_play_pause.setText("Play")

        folder = self.folder  # same folder as current EMG/Video

        trials = find_trials_in_folder(folder)
        trials_by_num = {trial["trial_num"]: trial for trial in trials}
        common = sorted(trials_by_num)
        if not common:
            print("No more trials found.")
            self.timer.start(self.interval_ms)
            self.is_paused = False
            self.btn_play_pause.setText("Pause")
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
            self.timer.start(self.interval_ms)
            self.is_paused = False
            self.btn_play_pause.setText("Pause")
            return

        self._load_trial(trials_by_num[n])

    def set_clip_window(self):
        """Create or update the highlighted window based on user sample input."""
        text = self.samples_input.text().strip()
        if not text:
            print("No sample count entered.")
            return
        try:
            n = int(text)
        except ValueError:
            print("Invalid sample count.")
            return

        n = max(1, min(n, len(self.emg_times)))
        self.clip_samples = n

        if self.dt_mean <= 0:
            width = 0.0
        else:
            width = self.dt_mean * (n - 1)

        t_start = float(self.emg_times[0])
        t_end = min(t_start + width, float(self.emg_times[-1]))

        if self.region is None:
            self.region = pg.LinearRegionItem(
                values=(t_start, t_end),
                movable=True,
                brush=(50, 50, 200, 50)
            )
            self.region.setZValue(-10)
            self.region.setBounds([float(self.emg_times[0]),
                                   float(self.emg_times[-1])])
            self.plot_widget.addItem(self.region)
        else:
            self.region.setRegion((t_start, t_end))

        print(f"Clip window set: {n} samples, approx width {width:.4f} s")

    def save_clip_segment(self):
        """
        Save EMG samples inside the current window AND matching video clip.

        - Base behavior: saves to ResultClip/actX/trial_Y.*  (unchanged)
        - NEW: if clip_samples == 500, also creates multiple centered clips
            with different lengths and saves them to
            ResultClipSizeUp{N}/actX/trial_Y.*
        """
        if self.region is None:
            print("No clip window defined.")
            return

        t_min, t_max = self.region.getRegion()
        t_min = float(t_min)
        t_max = float(t_max)

        # ---- EMG indices based on clip_samples (base clip) ----
        if self.clip_samples is not None:
            start_idx = int(np.searchsorted(self.emg_times, t_min, side="left"))
            end_idx = start_idx + self.clip_samples
            end_idx = min(end_idx, len(self.emg_times))
            if start_idx >= end_idx:
                print("Selected window has no samples.")
                return
            idx = np.arange(start_idx, end_idx)
        else:
            mask = (self.emg_times >= t_min) & (self.emg_times <= t_max)
            idx = np.where(mask)[0]
            if len(idx) == 0:
                print("Selected window has no samples.")
                return

        # ------------ Save base clip exactly as before ------------
        base_save = self._save_clip_to_root("ResultClip", idx)

        # ------------ NEW: multiple sizes centered on same region ------------
        if self.clip_samples == 500:
            # target sizes in *samples* (200 -> 1000)
            sizes = [
                200, 250, 300, 350, 400,
                450, 500, 550, 600, 650,
                700, 750, 800, 900, 1000
            ]

            # center index in original EMG array (absolute index)
            base_start = idx[0]
            base_end = idx[-1] + 1
            base_center = (base_start + base_end) // 2

            total_samples = len(self.emg_times)

            for n in sizes:
                half = n // 2
                start_n = base_center - half
                end_n = start_n + n

                # clamp to available sample range
                if start_n < 0:
                    start_n = 0
                    end_n = min(n, total_samples)
                if end_n > total_samples:
                    end_n = total_samples
                    start_n = max(0, end_n - n)

                if start_n >= end_n:
                    print(f"[ResultClipSizeUp{n}] window out of bounds, skipping.")
                    continue

                idx_n = np.arange(start_n, end_n)
                root_name = f"ResultClipSizeUp{n}"
                self._save_clip_to_root(root_name, idx_n)

        if base_save and base_save["act_idx"] == self.num_acts:
            loaded_next = self.load_next_trial()
            if not loaded_next:
                self._update_next_dest_label()
        else:
            self._update_next_dest_label()


    # --------------- Playback controls -----------------
    def toggle_play_pause(self):
        if self.is_paused:
            self.timer.start(self.interval_ms)
            self.is_paused = False
            self.btn_play_pause.setText("Pause")
        else:
            self.timer.stop()
            self.is_paused = True
            self.btn_play_pause.setText("Play")

    def replay(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_idx = 0

        self.curve_ch1.clear()
        self.curve_ch2.clear()
        self.curve_ch3.clear()
        self.curve_audio.clear()

        if self.is_paused:
            self.is_paused = False
            self.btn_play_pause.setText("Pause")
        self.timer.start(self.interval_ms)

    # --------------- Frame update ----------------------
    def _update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_idx = 0
            self.is_paused = True
            self.btn_play_pause.setText("Play")
            return

        self.frame_idx += 1
        current_time_sec = self.frame_idx / self.fps

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

        idx = np.searchsorted(self.emg_times, current_time_sec, side="right")

        if idx > 0:
            t = self.emg_times[:idx]
            y1 = self.emg_data[:idx, 0]
            y2 = self.emg_data[:idx, 1]
            y3 = self.emg_data[:idx, 2]

            self.curve_ch1.setData(t, y1)
            self.curve_ch2.setData(t, y2)
            self.curve_ch3.setData(t, y3)

        audio_idx = np.searchsorted(self.audio_times, current_time_sec, side="right")
        if audio_idx > 0:
            self.curve_audio.setData(
                self.audio_times[:audio_idx],
                self.audio_data[:audio_idx],
            )


# ----------------------------- MAIN SCRIPT ----------------------------------

def main():
    trial_logs_root = os.path.join("FirstPhase", "trial_logs")
    if not os.path.isdir(trial_logs_root):
        print(f"Folder does not exist: {trial_logs_root}")
        return


    trials = find_trials_under_root(trial_logs_root)
    if not trials:
        print(f"No matching trial/video/audio files found under {trial_logs_root}")
        return

    # print("Available trials:")
    # for index, trial in enumerate(trials, start=1):
    #     folder_name = os.path.basename(trial["folder"])
    #     print(f"{index}. {folder_name} - trial {trial['trial_num']}")

    # try:
    #     choice_text = input("Choose a trial to view [1]: ").strip()
    # except EOFError:
    #     choice_text = ""

    # if choice_text:
    #     try:
    #         choice = int(choice_text)
    #     except ValueError:
    #         print("Please enter a valid number.")
    #         return
    # else:
    #     choice = 1

    # if choice < 1 or choice > len(trials):
    #     print(f"Choose a number between 1 and {len(trials)}.")
    #     return

    selected_trial_index = 0
    selected_trial = trials[selected_trial_index]
    n = selected_trial["trial_num"]
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
        audio_path,
        available_trials=trials,
        current_trial_index=selected_trial_index,
    )
    viewer.setWindowTitle(f"Trial {n} - EMG + Video")
    viewer.resize(900, 750)
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
