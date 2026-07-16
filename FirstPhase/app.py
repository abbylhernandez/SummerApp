"""TrialLoggerApp — live EMG + video + mic logging GUI (PyQt5 + pyqtgraph)."""

import os
import re
import logging
import threading
import subprocess
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import imageio_ffmpeg
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

from config import (SIMULATE, CAM_FPS, CAM_WIDTH, CAM_HEIGHT,
                     EMG_Y_MIN, EMG_Y_MAX,
                    PREVIEW_POINTS, PLOT_UPDATE_MS, SAVE_ROOT, PACIFIC_TZ,
                    AUDIO_MIN_SPAN)
from camera import CameraCapture
from audio_capture import AudioCapture
from emg_serial import SerialEMGHandler, SimulatedEMGHandler
from theme import THEMES, CH_COLORS, MIC_COLOR, build_stylesheet


class TrialLoggerApp(QtWidgets.QWidget):

    # State machine values
    IDLE    = "idle"
    LOGGING = "logging"
    PAUSED  = "paused"
    ENDED   = "ended"
    PREVIEW = "preview"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("EMG Trial Logger + Live Video")

        self._theme_name = "light"

        # ---- per-object folders ----
        self.object_dirs = {}
        self.object_counts = {}
        self.group_dir = None
        self.current_obj = None

        # ---- trial state ----
        self.trial_num = 1
        self.state = self.IDLE
        self.prev_state = self.IDLE
        self.elapsed_s = 0.0
        self.last_perf = None

        self._buf_lock = threading.Lock()

        # per-trial buffers
        self.tns_buf = []
        self.t_buf = []
        self.c1_buf, self.c2_buf, self.c3_buf = [], [], []
        # preview rolling buffers
        self.pv_t = deque(maxlen=PREVIEW_POINTS)
        self.pv_c1 = deque(maxlen=PREVIEW_POINTS)
        self.pv_c2 = deque(maxlen=PREVIEW_POINTS)
        self.pv_c3 = deque(maxlen=PREVIEW_POINTS)
        self.pv_idx = 0

        self.trial_labels = {}
        self.trial_data = {}
        self.review_key = None
        self._review_video_path = None
        self.auto_enabled = False

        # ---- EMG source ----
        self.serial = SimulatedEMGHandler() if SIMULATE else SerialEMGHandler()
        self.serial.set_callback(self._on_sample_threadsafe)

        self.camera = CameraCapture()
        self.cam_timer = QtCore.QTimer(self)
        self.cam_timer.timeout.connect(self._update_camera)

        self.audio = AudioCapture()

        self._build_ui()
        self._apply_theme(self._theme_name)

        if self.camera.start():
            self.cam_timer.start(int(1000 / CAM_FPS))
        else:
            self.video_label.setText("Camera unavailable")

        if not self.audio.start():
            self.audio_plot.setTitle("Microphone unavailable")

        if not self.serial.start():
            self.status_lbl.setText("Status: serial port unavailable — check SERIAL_PORT")

        self.plot_timer = QtCore.QTimer(self)
        self.plot_timer.timeout.connect(self._refresh_plot)
        self.plot_timer.start(PLOT_UPDATE_MS)

        self._update_controls()

    # ----------------------------------------------------------
    # UI
    # ----------------------------------------------------------
    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # status + theme toggle
        top_row = QtWidgets.QHBoxLayout()
        self.status_lbl = QtWidgets.QLabel("Status: Idle")
        top_row.addWidget(self.status_lbl, stretch=1)
        self.theme_btn = QtWidgets.QPushButton("🌙  Dark")
        self.theme_btn.clicked.connect(self.on_theme_btn)
        top_row.addWidget(self.theme_btn)
        layout.addLayout(top_row)

        # object input
        obj_row = QtWidgets.QHBoxLayout()
        obj_row.addWidget(QtWidgets.QLabel("Object:"))
        self.object_edit = QtWidgets.QLineEdit()
        self.object_edit.setPlaceholderText("e.g. Apple — type to switch the object for the next trial")
        self.object_edit.textChanged.connect(self.on_object_changed)
        obj_row.addWidget(self.object_edit, stretch=1)
        layout.addLayout(obj_row)

        # live camera feed
        self.video_label = QtWidgets.QLabel("Camera starting…")
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setFixedSize(CAM_WIDTH, CAM_HEIGHT)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        layout.addWidget(self.video_label, alignment=QtCore.Qt.AlignHCenter)

        pg.setConfigOptions(antialias=False)

        # sound graph (on top of EMG)
        self.audio_plot = pg.PlotWidget()
        self.audio_plot.setTitle("Microphone")
        self.audio_plot.setLabel("left", "Amplitude")
        self.audio_plot.getPlotItem().hideAxis("bottom")
        self.audio_plot.setMouseEnabled(x=False, y=False)
        self.audio_curve = self.audio_plot.plot([], [], pen=pg.mkPen(MIC_COLOR, width=1))
        layout.addWidget(self.audio_plot, stretch=1)

        # EMG plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel("left", "EMG (V)")
        self.plot_widget.setLabel("bottom", "Time (s)")
        self.plot_widget.setYRange(EMG_Y_MIN, EMG_Y_MAX, padding=0)
        self._legend = self.plot_widget.addLegend(offset=(10, 10))
        self.curve1 = self.plot_widget.plot([], [], pen=pg.mkPen(CH_COLORS[0], width=2), name="ch1")
        self.curve2 = self.plot_widget.plot([], [], pen=pg.mkPen(CH_COLORS[1], width=2), name="ch2")
        self.curve3 = self.plot_widget.plot([], [], pen=pg.mkPen(CH_COLORS[2], width=2), name="ch3")
        layout.addWidget(self.plot_widget, stretch=1)

        # link the time axes so the two graphs pan/zoom together
        self.audio_plot.setXLink(self.plot_widget)

        # completed-trial labels row
        comp_box = QtWidgets.QHBoxLayout()
        comp_box.addWidget(QtWidgets.QLabel("Completed:"))

        # Keep completed-trial buttons from increasing the main window's
        # minimum width. Older trials remain available by scrolling sideways.
        self.labels_widget = QtWidgets.QWidget()
        self.labels_layout = QtWidgets.QHBoxLayout(self.labels_widget)
        self.labels_layout.setContentsMargins(0, 0, 0, 0)
        self.labels_layout.setSpacing(6)
        self.labels_layout.addStretch(1)

        self.labels_scroll = QtWidgets.QScrollArea()
        self.labels_scroll.setWidget(self.labels_widget)
        self.labels_scroll.setWidgetResizable(True)
        self.labels_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.labels_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.labels_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.labels_scroll.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        self.labels_scroll.setFixedHeight(52)
        comp_box.addWidget(self.labels_scroll, stretch=1)

        self.review_video_btn = QtWidgets.QPushButton("▶ Open trial video")
        self.review_video_btn.setVisible(False)
        self.review_video_btn.clicked.connect(self._open_review_video)
        comp_box.addWidget(self.review_video_btn)
        layout.addLayout(comp_box)

        # control buttons
        btn_row = QtWidgets.QHBoxLayout()
        self.trial_btn = QtWidgets.QPushButton("Start Trial 1")
        self.end_btn   = QtWidgets.QPushButton("End Trial")
        self.redo_btn  = QtWidgets.QPushButton("Redo Trial")
        self.next_btn  = QtWidgets.QPushButton("Start Trial 2  ▶")
        self.preview_btn = QtWidgets.QPushButton("Preview")
        self.preview_btn.setCheckable(True)
        self.auto_btn = QtWidgets.QPushButton("Auto: Off")
        self.auto_btn.setCheckable(True)
        self.autoy_btn = QtWidgets.QPushButton("Auto Y")
        self.autoy_btn.setCheckable(True)

        for b in (self.trial_btn, self.end_btn, self.redo_btn, self.next_btn,
                  self.preview_btn, self.auto_btn, self.autoy_btn):
            b.setMinimumHeight(34)
            btn_row.addWidget(b)
        layout.addLayout(btn_row)

        self.trial_btn.clicked.connect(self.on_trial_btn)
        self.end_btn.clicked.connect(self.on_end_btn)
        self.redo_btn.clicked.connect(self.on_redo_btn)
        self.next_btn.clicked.connect(self.on_next_btn)
        self.preview_btn.clicked.connect(self.on_preview_btn)
        self.auto_btn.clicked.connect(self.on_auto_btn)
        self.autoy_btn.clicked.connect(self.on_autoy_btn)

    # ----------------------------------------------------------
    # Theme
    # ----------------------------------------------------------
    def on_theme_btn(self):
        self._apply_theme("dark" if self._theme_name == "light" else "light")

    def _apply_theme(self, name):
        self._theme_name = name
        pal = THEMES[name]
        self.setStyleSheet(build_stylesheet(pal))
        self.status_lbl.setStyleSheet(f"font-weight: bold; color: {pal['text']};")
        self.theme_btn.setText(pal["btn_text"])

        for p in (self.audio_plot, self.plot_widget):
            p.setBackground(pal["plot_bg"])
            pi = p.getPlotItem()
            pi.showGrid(x=True, y=True, alpha=pal["grid"])
            for axname in ("left", "bottom"):
                ax = pi.getAxis(axname)
                ax.setPen(pg.mkPen(pal["axis"]))
                ax.setTextPen(pg.mkPen(pal["axis"]))

        self.audio_plot.setTitle("Microphone", color=pal["axis"])
        self.audio_plot.setLabel("left", "Amplitude", color=pal["axis"])
        self.plot_widget.setLabel("left", "EMG (V)", color=pal["axis"])
        self.plot_widget.setLabel("bottom", "Time (s)", color=pal["axis"])
        try:
            self._legend.setLabelTextColor(pal["axis"])
        except Exception:
            pass

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
            self._reset_to_idle()

    def on_preview_btn(self):
        self._exit_review()
        if self.preview_btn.isChecked():
            self._enter_preview()
        else:
            self._exit_preview()

    def on_auto_btn(self):
        self.auto_enabled = self.auto_btn.isChecked()
        self.auto_btn.setText("Auto: On" if self.auto_enabled else "Auto: Off")

    def on_autoy_btn(self):
        # auto-zoom the EMG y-axis to the signal, or lock back to EMG_Y_MIN..EMG_Y_MAX
        if self.autoy_btn.isChecked():
            self.plot_widget.enableAutoRange(axis="y", enable=True)
        else:
            self.plot_widget.enableAutoRange(axis="y", enable=False)
            self.plot_widget.setYRange(EMG_Y_MIN, EMG_Y_MAX, padding=0)

    def on_object_changed(self, _text):
        if self.state == self.IDLE:
            self.trial_num = self._next_trial_number()
            self._update_controls()

    # ----------------------------------------------------------
    # Object / folder helpers
    # ----------------------------------------------------------
    def _current_object(self):
        obj = self.object_edit.text().strip() or "Object"
        return re.sub(r'[<>:"/\\|?*]', "", obj).strip() or "Object"

    def _next_trial_number(self):
        return self.object_counts.get(self._current_object(), 0) + 1

    def _folder_for_object(self, obj):
        if obj in self.object_dirs:
            return self.object_dirs[obj]
        now = datetime.now(PACIFIC_TZ)
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%I-%M-%S %p")
        base = f"{obj}Data-{date_str} {time_str}"
        path = Path(SAVE_ROOT) / base
        n = 2
        while path.exists():
            path = Path(SAVE_ROOT) / f"{base} ({n})"
            n += 1
        path.mkdir(parents=True, exist_ok=True)
        self.object_dirs[obj] = path
        logging.info("📂 Created folder for %s: %s", obj, path)
        return path

    # ----------------------------------------------------------
    # Trial lifecycle
    # ----------------------------------------------------------
    def _start_logging(self):
        obj = self._current_object()
        self.group_dir = self._folder_for_object(obj)
        self.current_obj = obj
        self.trial_num = self.object_counts.get(obj, 0) + 1

        self.plot_widget.enableAutoRange(axis="x", enable=True)
        self.serial.flush_input()

        self.camera.start_recording(self.group_dir / f"video{self.trial_num}.avi")
        self.audio.start_recording(self.group_dir / f"_audio{self.trial_num}.wav")

        with self._buf_lock:
            self.elapsed_s = 0.0
            self.last_perf = None
            self.tns_buf = []
            self.t_buf = []
            self.c1_buf, self.c2_buf, self.c3_buf = [], [], []
            self.state = self.LOGGING

        logging.info("▶️ Trial %d started.", self.trial_num)
        self._update_controls()

    def _pause_logging(self):
        with self._buf_lock:
            self.state = self.PAUSED
            self.last_perf = None
        self.camera.pause_recording()
        self.audio.pause_recording()
        logging.info("⏸️ Trial %d paused.", self.trial_num)
        self._update_controls()

    def _resume_logging(self):
        with self._buf_lock:
            self.state = self.LOGGING
            self.last_perf = None
        self.camera.resume_recording()
        self.audio.resume_recording()
        logging.info("▶️ Trial %d resumed.", self.trial_num)
        self._update_controls()

    def _write_trial_file(self):
        path = self.group_dir / f"trial{self.trial_num}.txt"
        with open(path, "w", newline="") as f:
            f.write("t_ns,ch1_V,ch2_V,ch3_V\n")
            for t_ns, a, b, c in zip(self.tns_buf, self.c1_buf, self.c2_buf, self.c3_buf):
                f.write(f"{t_ns},{a:.6f},{b:.6f},{c:.6f}\n")
        logging.info("💾 Wrote %s (%d samples).", path, len(self.tns_buf))
        return path

    def _write_video_timestamps(self):
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

    def _write_audio_envelope(self):
        et, ea = self.audio.get_envelope()
        if not et:
            return None
        path = self.group_dir / f"audio{self.trial_num}.csv"
        with open(path, "w", newline="") as f:
            f.write("t_s,amp\n")
            for t_s, amp in zip(et, ea):
                f.write(f"{t_s:.6f},{amp:.6f}\n")
        logging.info("💾 Wrote %s (%d points).", path, len(et))
        return path

    def _mux_audio_into_video(self, trial_num, wav_path):
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
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
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
        with self._buf_lock:
            self.state = self.ENDED
        self.camera.stop_recording()
        wav_path = self.audio.stop_recording()
        self._write_trial_file()
        self._write_video_timestamps()
        self._write_audio_envelope()
        self._mux_audio_into_video(self.trial_num, wav_path)
        self.object_counts[self.current_obj] = self.trial_num

        key = (self.current_obj, self.trial_num)
        et, ea = self.audio.get_envelope()
        self.trial_data[key] = {
            "title": f"{self.current_obj} Trial {self.trial_num}",
            "emg": (list(self.t_buf), list(self.c1_buf), list(self.c2_buf), list(self.c3_buf)),
            "audio": (et, ea),
            "video": self.group_dir / f"video{self.trial_num}.avi",
        }

        btn = QtWidgets.QPushButton(f"{self.current_obj} Trial {self.trial_num} ✓")
        btn.setCursor(QtCore.Qt.PointingHandCursor)
        btn.setToolTip("Click to view this trial's graphs")
        # Insert before the trailing stretch so buttons stay left-aligned.
        self.labels_layout.insertWidget(self.labels_layout.count() - 1, btn)
        self.trial_labels[key] = btn
        btn.clicked.connect(lambda _=False, k=key: self._open_review(k))
        self._restyle_trial_buttons()

        logging.info("⏹️ %s Trial %d ended (%d samples).",
                     self.current_obj, self.trial_num, len(self.t_buf))
        self._update_controls()

        if self.auto_enabled:
            QtCore.QTimer.singleShot(0, self._auto_advance)

    def _auto_advance(self):
        if self.auto_enabled and self.state == self.ENDED:
            self._reset_to_idle()
            self._start_logging()

    # ----------------------------------------------------------
    # Review (in place)
    # ----------------------------------------------------------
    def _open_review(self, key):
        if self.review_key == key:
            self._exit_review()
        else:
            self._enter_review(key)

    def _enter_review(self, key):
        data = self.trial_data.get(key)
        if not data:
            return
        self.review_key = key
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
        self._update_controls()
        self._refresh_plot()

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
                os.startfile(str(self._review_video_path))
        except Exception as e:
            logging.warning("⚠️ Could not open video: %s", e)

    def _redo_trial(self):
        self.camera.stop_recording()
        self.audio.stop_recording()

        doomed = [
            self.group_dir / f"trial{self.trial_num}.txt",
            self.group_dir / f"video{self.trial_num}.avi",
            self.group_dir / f"video{self.trial_num}timestamps.csv",
            self.group_dir / f"audio{self.trial_num}.csv",
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

        self.trial_data.pop((self.current_obj, self.trial_num), None)
        lbl = self.trial_labels.pop((self.current_obj, self.trial_num), None)
        if lbl is not None:
            self.labels_layout.removeWidget(lbl)
            lbl.deleteLater()

        if self.object_counts.get(self.current_obj) == self.trial_num:
            self.object_counts[self.current_obj] = self.trial_num - 1

        redo_obj = self.current_obj
        self._reset_to_idle()
        logging.info("↺ %s Trial %d ready to redo (stopped).", redo_obj, self.trial_num)

    def _reset_to_idle(self):
        with self._buf_lock:
            self.state = self.IDLE
            self.elapsed_s = 0.0
            self.last_perf = None
            self.tns_buf = []
            self.t_buf = []
            self.c1_buf, self.c2_buf, self.c3_buf = [], [], []
        self.current_obj = None
        self.trial_num = self._next_trial_number()
        self._clear_curves()
        self._update_controls()

    # ----------------------------------------------------------
    # Preview mode
    # ----------------------------------------------------------
    def _enter_preview(self):
        with self._buf_lock:
            self.prev_state = self.state
            self.pv_t.clear(); self.pv_c1.clear(); self.pv_c2.clear(); self.pv_c3.clear()
            self.pv_idx = 0
            self.state = self.PREVIEW
        self._clear_curves()
        self._update_controls()

    def _exit_preview(self):
        with self._buf_lock:
            self.state = self.prev_state
        self._clear_curves()
        self._refresh_plot()
        self._update_controls()

    # ----------------------------------------------------------
    # Sample handling (serial thread) + plotting (GUI thread)
    # ----------------------------------------------------------
    def _on_sample_threadsafe(self, c1, c2, c3, t_perf):
        with self._buf_lock:
            if self.state == self.LOGGING:
                dt = 0.0 if self.last_perf is None else (t_perf - self.last_perf)
                self.elapsed_s += dt
                self.last_perf = t_perf
                self.tns_buf.append(int(self.elapsed_s * 1e9))
                self.t_buf.append(self.elapsed_s)
                self.c1_buf.append(c1)
                self.c2_buf.append(c2)
                self.c3_buf.append(c3)
            elif self.state == self.PREVIEW:
                self.pv_t.append(self.pv_idx)
                self.pv_c1.append(c1)
                self.pv_c2.append(c2)
                self.pv_c3.append(c3)
                self.pv_idx += 1

    def _refresh_plot(self):
        if self.review_key is not None:
            return

        if self.state in (self.LOGGING, self.PAUSED, self.ENDED):
            et, ea = self.audio.get_envelope()
            if et:
                self.audio_curve.setData(et, ea)
                span = max(max(ea) * 1.2, AUDIO_MIN_SPAN)
                self.audio_plot.setYRange(0.0, span, padding=0)
            else:
                self.audio_curve.setData([], [])
        else:
            self.audio_curve.setData([], [])

        if self.state in (self.LOGGING, self.PAUSED, self.ENDED):
            with self._buf_lock:
                t = list(self.t_buf)
                a, b, c = list(self.c1_buf), list(self.c2_buf), list(self.c3_buf)
            if t:
                self.curve1.setData(t, a)
                self.curve2.setData(t, b)
                self.curve3.setData(t, c)
        elif self.state == self.PREVIEW:
            with self._buf_lock:
                x = list(self.pv_t)
                a, b, c = list(self.pv_c1), list(self.pv_c2), list(self.pv_c3)
            if x:
                self.curve1.setData(x, a)
                self.curve2.setData(x, b)
                self.curve3.setData(x, c)

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

    # ----------------------------------------------------------
    # Control enable/label state
    # ----------------------------------------------------------
    def _update_controls(self):
        n = self.trial_num
        s = self.state
        obj = self.current_obj if self.current_obj else self._current_object()

        if s == self.IDLE:
            self.trial_btn.setText(f"Start {obj} Trial {n}")
        elif s == self.LOGGING:
            self.trial_btn.setText(f"Pause {obj} Trial {n}")
        elif s == self.PAUSED:
            self.trial_btn.setText(f"Resume {obj} Trial {n}")
        else:
            self.trial_btn.setText(f"Start {obj} Trial {n}")

        self.next_btn.setText(f"Start {obj} Trial {n + 1}  ▶")

        active = s in (self.LOGGING, self.PAUSED)
        self.trial_btn.setEnabled(s in (self.IDLE, self.LOGGING, self.PAUSED))
        self.end_btn.setEnabled(active)
        self.redo_btn.setEnabled(s in (self.LOGGING, self.PAUSED, self.ENDED))
        self.next_btn.setVisible(s == self.ENDED)
        self.preview_btn.setEnabled(s in (self.IDLE, self.ENDED, self.PREVIEW))

        if s == self.PREVIEW:
            self.trial_btn.setEnabled(False)
            self.end_btn.setEnabled(False)
            self.redo_btn.setEnabled(False)
            self.next_btn.setVisible(False)

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
