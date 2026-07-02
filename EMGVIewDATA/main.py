import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

import tkinter as tk
from tkinter import filedialog
import os
import re
import time
import threading

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

# Optional audio playback (extract the video's audio track and play it).
import wave
import tempfile
import subprocess
try:
    import sounddevice as sd
    import imageio_ffmpeg
    AUDIO_OK = True
except Exception:
    AUDIO_OK = False


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
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


# ── Palettes ──────────────────────────────────────────────────────────────────

_LIGHT = dict(
    BG="#dbeeff", PANEL="#f0f8ff", WHITE="#ffffff", TOP_BAR="#1d4ed8",
    TEXT="#1e3a5f", TEXT_MUTED="#4d7aa8", TEXT_FAINT="#93b4ce",
    ACCENT="#2563eb", ACCENT_HVR="#1d4ed8",
    ACCENT2="#0ea5e9", ACCENT2_HVR="#0284c7",
    ACCENT3="#059669", ACCENT3_HVR="#047857",
    BORDER="#bfdbfe",
    VIDEO_BG="#dbeeff", GRAPH_BG="#f8fafc", GRAPH_FIG="#f0f8ff",
    GRAPH_GRID="#bfdbfe", GRAPH_TEXT="#1e3a5f",
    CURSOR_CLR="#1d4ed8", ERROR="#dc2626",
    CH_COLORS={"ch1_V": "#2563eb", "ch2_V": "#059669", "ch3_V": "#d97706"},
    THEME_BTN_TEXT="◑  Dark",
)

_DARK = dict(
    BG="#0f172a", PANEL="#1e293b", WHITE="#1e293b", TOP_BAR="#1e293b",
    TEXT="#e2e8f0", TEXT_MUTED="#94a3b8", TEXT_FAINT="#475569",
    ACCENT="#3b82f6", ACCENT_HVR="#2563eb",
    ACCENT2="#0ea5e9", ACCENT2_HVR="#0284c7",
    ACCENT3="#10b981", ACCENT3_HVR="#059669",
    BORDER="#334155",
    VIDEO_BG="#0f172a", GRAPH_BG="#0f172a", GRAPH_FIG="#1e293b",
    GRAPH_GRID="#334155", GRAPH_TEXT="#94a3b8",
    CURSOR_CLR="#60a5fa", ERROR="#f87171",
    CH_COLORS={"ch1_V": "#60a5fa", "ch2_V": "#34d399", "ch3_V": "#fbbf24"},
    THEME_BTN_TEXT="☀  Light",
)

# Initialise all palette names as module globals from _LIGHT
BG = PANEL = WHITE = TOP_BAR = TEXT = TEXT_MUTED = TEXT_FAINT = ""
ACCENT = ACCENT_HVR = ACCENT2 = ACCENT2_HVR = ACCENT3 = ACCENT3_HVR = ""
BORDER = VIDEO_BG = GRAPH_BG = GRAPH_FIG = GRAPH_GRID = GRAPH_TEXT = ""
CURSOR_CLR = ERROR = THEME_BTN_TEXT = ""
CH_COLORS: dict = {}


def _apply_palette(palette: dict):
    globals().update(palette)


_apply_palette(_LIGHT)


# ── EMG helpers ───────────────────────────────────────────────────────────────

def unwrap_t_ns(t_ns: np.ndarray) -> np.ndarray:
    t = np.asarray(t_ns, dtype=np.int64)
    out = np.empty(len(t), dtype=np.int64)
    out[0] = t[0]
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        if dt < 0:
            dt += 2**32
        out[i] = out[i - 1] + dt
    return out


def load_emg(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["t_ns"] = unwrap_t_ns(df["t_ns"].values)
    df = df.sort_values("t_ns").reset_index(drop=True)
    df["t_s"] = df["t_ns"] / 1e9
    return df


def load_frame_times(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).sort_values("frame_idx").reset_index(drop=True)
    df["t_s"] = df["t_ns"] / 1e9
    return df


def load_audio(path: str) -> pd.DataFrame:
    """Audio envelope written by the recorder: columns 't_s,amp'."""
    df = pd.read_csv(path)
    if "t_s" not in df.columns or "amp" not in df.columns:
        raise ValueError("audio csv must have columns t_s, amp")
    return df.sort_values("t_s").reset_index(drop=True)


def scan_trials(folder: str) -> list:
    nums = []
    for f in os.listdir(folder):
        m = re.fullmatch(r"[Tt]rial(\d+)(?:\.\w+)?", f)
        if m and m.group(1) not in nums:
            nums.append(m.group(1))
    return sorted(nums, key=lambda x: int(x))


# ── App ───────────────────────────────────────────────────────────────────────

class EMGViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EMG Viewer")
        self.geometry("1920x1080")
        self.resizable(True, True)
        self._dark = False
        self.configure(bg=BG)

        self.container = tk.Frame(self, bg=BG)
        self.container.pack(fill="both", expand=True)
        self.show_home()

    # ── Screen transitions ────────────────────────────────────────────────────

    def show_home(self):
        for w in self.container.winfo_children():
            w.destroy()
        self.configure(bg=BG)
        self.container.configure(bg=BG)
        HomeFrame(self.container, self).pack(fill="both", expand=True)

    def show_viewer(self, trial_path, video_path, timestamps_path, trial_number,
                    folder=None, all_trials=None, saved_state=None):
        for w in self.container.winfo_children():
            w.destroy()
        self.configure(bg=BG)
        self.container.configure(bg=BG)
        ViewerFrame(
            self.container, self,
            trial_path, video_path, timestamps_path, trial_number,
            folder=folder, all_trials=all_trials or [trial_number],
            saved_state=saved_state,
        ).pack(fill="both", expand=True)

    # ── Theme toggle ──────────────────────────────────────────────────────────

    def toggle_theme(self, viewer_ctx=None):
        self._dark = not self._dark
        _apply_palette(_DARK if self._dark else _LIGHT)

        # If we're in the viewer, rebuild it and restore position
        if viewer_ctx:
            self.show_viewer(
                viewer_ctx["trial_path"],
                viewer_ctx["video_path"],
                viewer_ctx["timestamps_path"],
                viewer_ctx["trial_number"],
                folder=viewer_ctx["folder"],
                all_trials=viewer_ctx["all_trials"],
                saved_state=viewer_ctx["saved_state"],
            )
        else:
            self.show_home()


# ── Home ──────────────────────────────────────────────────────────────────────

class HomeFrame(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self.app = app

        # Theme toggle button — top-right corner
        tk.Button(
            self,
            text=THEME_BTN_TEXT,
            font=("Helvetica", 10),
            fg=TEXT, bg=PANEL,
            activebackground=BORDER, activeforeground=TEXT,
            relief="flat", padx=10, pady=4, cursor="hand2",
            highlightbackground=BORDER, highlightthickness=1,
            command=self.app.toggle_theme,
        ).place(relx=1.0, rely=0.0, anchor="ne", x=-16, y=12)

        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)
        self.rowconfigure(2, weight=1)
        self.columnconfigure(0, weight=1)

        # Card
        card = tk.Frame(self, bg=WHITE, padx=64, pady=52,
                        highlightbackground=BORDER, highlightthickness=1)
        card.grid(row=1, column=0)

        tk.Frame(card, bg=ACCENT, height=5, width=340).pack()

        tk.Label(
            card, text="EMG Viewer",
            font=("Helvetica", 44, "bold"),
            fg=TEXT, bg=WHITE,
        ).pack(pady=(28, 4))

        tk.Frame(card, bg=BORDER, height=1, width=340).pack(pady=28)

        tk.Label(
            card, text="Select the folder containing your Trial files",
            font=("Helvetica", 11),
            fg=TEXT_MUTED, bg=WHITE,
        ).pack(pady=(0, 18))

        tk.Button(
            card, text="  Open Folder  ",
            font=("Helvetica", 13, "bold"),
            fg=WHITE, bg=ACCENT,
            activebackground=ACCENT_HVR, activeforeground=WHITE,
            relief="flat", padx=20, pady=10, cursor="hand2",
            command=self._select_folder,
        ).pack()

        self._status = tk.Label(
            card, text="",
            font=("Helvetica", 10),
            fg=ERROR, bg=WHITE,
            wraplength=420,
        )
        self._status.pack(pady=(18, 0))

        tk.Label(
            self,
            text="Expected filenames:  Trial1.csv  +  Video1.mp4  in the same folder",
            font=("Helvetica", 9),
            fg=TEXT_FAINT, bg=BG,
        ).grid(row=2, column=0, pady=(0, 20))

    def _select_folder(self):
        folder = filedialog.askdirectory(title="Select Trial Folder")
        if not folder:
            return

        all_trials = scan_trials(folder)
        if not all_trials:
            self._status.config(
                text="No Trial files found in this folder. "
                     "Expected files named Trial1, Trial2, etc."
            )
            return

        n = all_trials[0]
        video_path = self._find_file(
            folder, f"video{n}", [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"],
        )
        if video_path is None:
            self._status.config(
                text=f'Found {len(all_trials)} trial(s) but no matching video for Trial{n}.'
            )
            return

        timestamps_path = self._find_file(folder, f"video{n}timestamps", [".csv"])
        trial_path = self._find_file(folder, f"trial{n}", [".csv", ".txt", ".dat"])

        self._status.config(text="")
        self.app.show_viewer(trial_path, video_path, timestamps_path, n,
                             folder=folder, all_trials=all_trials)

    def _find_file(self, folder, stem_lower, extensions):
        for ext in extensions:
            for prefix in (stem_lower, stem_lower[0].upper() + stem_lower[1:]):
                candidate = os.path.join(folder, prefix + ext)
                if os.path.isfile(candidate):
                    return candidate
        return None


# ── Viewer ────────────────────────────────────────────────────────────────────

class ViewerFrame(tk.Frame):
    def __init__(self, parent, app, trial_path, video_path, timestamps_path, trial_number,
                 folder=None, all_trials=None, saved_state=None):
        super().__init__(parent, bg=BG)
        self.app = app
        self.trial_path = trial_path
        self.video_path = video_path
        self.timestamps_path = timestamps_path
        self.trial_number = trial_number
        self.folder = folder
        self.all_trials = all_trials or [trial_number]
        self._saved_state = saved_state or {}

        self._cap = None
        self._video_running = False
        self._paused = self._saved_state.get("paused", False)
        self._speed = self._saved_state.get("speed", 1.0)
        self._after_id = None
        self._current_frame_idx = self._saved_state.get("frame_idx", 0)
        self._frames_df = None
        self._cursor = None
        self._ax_audio = None
        self._cursor_audio = None
        self._fig = None
        self._bg = None
        self._box_w = 0
        self._box_h = 0

        # audio track (extracted from the video) for playback
        self._audio_data = None
        self._audio_rate = None
        self._audio = None              # _AudioStreamPlayer
        self._audio_master = False      # True when audio drives the clock (1x)

        self._build_ui()
        self._load_emg()
        self.after(250, self._start_video)

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Top bar
        bar = tk.Frame(self, bg=TOP_BAR, pady=8)
        bar.pack(fill="x")

        tk.Button(
            bar, text="← Back",
            font=("Helvetica", 11),
            fg=WHITE, bg=ACCENT,
            activebackground=ACCENT_HVR, activeforeground=WHITE,
            relief="flat", padx=12, pady=4, cursor="hand2",
            command=self._on_back,
        ).pack(side="left", padx=12)

        self._title_label = tk.Label(
            bar, text=f"Trial {self.trial_number}",
            font=("Helvetica", 15, "bold"),
            fg=WHITE, bg=TOP_BAR,
        )
        self._title_label.pack(side="left", padx=8)

        # Theme toggle button — right side of top bar
        tk.Button(
            bar,
            text=THEME_BTN_TEXT,
            font=("Helvetica", 10),
            fg=WHITE, bg=ACCENT,
            activebackground=ACCENT_HVR, activeforeground=WHITE,
            relief="flat", padx=12, pady=4, cursor="hand2",
            command=self._on_theme_toggle,
        ).pack(side="right", padx=12)

        # Trial navigation
        curr_idx = self.all_trials.index(self.trial_number) if self.trial_number in self.all_trials else 0

        self._prev_btn = tk.Button(
            bar, text="◀",
            font=("Helvetica", 11),
            fg=WHITE, bg=ACCENT,
            activebackground=ACCENT_HVR, activeforeground=WHITE,
            relief="flat", padx=8, pady=4, cursor="hand2",
            command=self._go_prev,
            state="normal" if curr_idx > 0 else "disabled",
        )
        self._prev_btn.pack(side="left", padx=(16, 2))

        self._trial_var = tk.StringVar(value=f"Trial {self.trial_number}")
        trial_menu = tk.OptionMenu(
            bar, self._trial_var,
            *[f"Trial {n}" for n in self.all_trials],
            command=lambda val: self._go_to_trial(val.split()[-1]),
        )
        trial_menu.config(
            font=("Helvetica", 11), fg=WHITE, bg=ACCENT,
            activebackground=ACCENT_HVR, activeforeground=WHITE,
            relief="flat", padx=6, pady=3, cursor="hand2",
            highlightthickness=0,
        )
        trial_menu["menu"].config(
            font=("Helvetica", 11), fg=TEXT, bg=WHITE,
            activebackground=BORDER, activeforeground=TEXT,
        )
        trial_menu.pack(side="left", padx=2)

        self._next_btn = tk.Button(
            bar, text="▶",
            font=("Helvetica", 11),
            fg=WHITE, bg=ACCENT,
            activebackground=ACCENT_HVR, activeforeground=WHITE,
            relief="flat", padx=8, pady=4, cursor="hand2",
            command=self._go_next,
            state="normal" if curr_idx < len(self.all_trials) - 1 else "disabled",
        )
        self._next_btn.pack(side="left", padx=(2, 0))

        # Outer grid: col 0 = content panels, col 1 = persistent checkbox sidebar
        self._outer = tk.Frame(self, bg=BG)
        self._outer.pack(fill="both", expand=True)
        self._outer.rowconfigure(0, weight=1, minsize=300)
        self._outer.rowconfigure(1, weight=1, minsize=300)
        self._outer.columnconfigure(0, weight=1)
        self._outer.columnconfigure(1, weight=0, minsize=200)
        outer = self._outer

        self._outer.bind("<Configure>", self._on_outer_configure)

        # ── Persistent checkbox sidebar ───────────────────────────────────────
        cb_frame = tk.Frame(outer, bg=PANEL,
                            highlightbackground=BORDER, highlightthickness=1)
        cb_frame.grid(row=0, column=1, rowspan=2, sticky="ns", padx=(0, 12), pady=10)

        self._var_both  = tk.BooleanVar(value=True)
        self._var_video = tk.BooleanVar(value=True)
        self._var_emg   = tk.BooleanVar(value=True)

        tk.Label(cb_frame, text="Display",
                 font=("Helvetica", 10, "bold"),
                 fg=TEXT, bg=PANEL).pack(anchor="w", pady=(16, 6), padx=14)

        tk.Frame(cb_frame, bg=BORDER, height=1).pack(fill="x", padx=14, pady=(0, 6))

        for text, var, color, cmd in [
            ("Enable Both",  self._var_both,  TEXT,    self._on_cb_both),
            ("Enable Video", self._var_video, ACCENT,  self._on_cb_video),
            ("Enable EMG",   self._var_emg,   ACCENT3, self._on_cb_emg),
        ]:
            tk.Checkbutton(
                cb_frame, text=text, variable=var,
                font=("Helvetica", 10), fg=color, bg=PANEL,
                selectcolor=BORDER, activebackground=PANEL,
                activeforeground=color, command=cmd,
            ).pack(anchor="w", padx=14, pady=5)

        # ── Video panel ───────────────────────────────────────────────────────
        vid_wrap = tk.Frame(outer, bg=PANEL,
                            highlightbackground=BORDER, highlightthickness=1)
        vid_wrap.grid(row=0, column=0, sticky="nsew", padx=(12, 4), pady=(10, 4))

        tk.Label(
            vid_wrap, text=f"Video {self.trial_number}",
            font=("Helvetica", 11, "bold"),
            fg=ACCENT, bg=PANEL, anchor="w",
        ).pack(fill="x", padx=12, pady=(8, 2))

        tk.Frame(vid_wrap, bg=BORDER, height=1).pack(fill="x", padx=12)

        self.video_label = tk.Label(vid_wrap, bg=VIDEO_BG)
        self.video_label.pack(fill="both", expand=True, padx=10, pady=(6, 8))
        self.video_label.bind("<Configure>",
                              lambda e: self._on_video_resize(e.width, e.height))

        # ── EMG graph panel ───────────────────────────────────────────────────
        self._graph_wrap = tk.Frame(outer, bg=PANEL,
                                    highlightbackground=BORDER, highlightthickness=1)
        self._graph_wrap.grid(row=1, column=0, sticky="nsew", padx=(12, 4), pady=(4, 10))
        self._graph_wrap.grid_propagate(False)
        graph_wrap = self._graph_wrap

        graph_header = tk.Frame(graph_wrap, bg=PANEL)
        graph_header.pack(fill="x", padx=12, pady=(6, 2))

        tk.Label(
            graph_header, text=f"EMG — Trial {self.trial_number}",
            font=("Helvetica", 11, "bold"),
            fg=ACCENT3, bg=PANEL, anchor="w",
        ).pack(side="left")

        tk.Button(
            graph_header, text="↺ Reset",
            font=("Helvetica", 10),
            fg=WHITE, bg=ACCENT3,
            activebackground=ACCENT3_HVR, activeforeground=WHITE,
            relief="flat", padx=10, pady=2, cursor="hand2",
            command=self._reset_video,
        ).pack(side="right", padx=(4, 0))

        # Speed selector
        tk.Label(
            graph_header, text="Speed:",
            font=("Helvetica", 10), fg=TEXT_MUTED, bg=PANEL,
        ).pack(side="right", padx=(8, 2))

        speed_labels = ["0.05×", "0.1×", "0.15×", "0.25×", "0.33×", "0.5×", "0.75×", "1×", "1.5×", "2×"]
        speed_map = {
            "0.05×": 0.05, "0.1×": 0.1, "0.15×": 0.15,
            "0.25×": 0.25, "0.33×": 0.33, "0.5×": 0.5,
            "0.75×": 0.75, "1×": 1.0, "1.5×": 1.5, "2×": 2.0,
        }
        # Pick the label that matches the restored speed, fallback to "1×"
        rev_map = {v: k for k, v in speed_map.items()}
        default_speed_label = rev_map.get(self._speed, "1×")

        self._speed_var = tk.StringVar(value=default_speed_label)

        def _on_speed_change(*_):
            self._speed = speed_map[self._speed_var.get()]
            # restart the clock/audio for the new speed if currently playing
            self._stop_audio()
            if self._video_running and not self._paused:
                self._begin_playback()

        speed_menu = tk.OptionMenu(graph_header, self._speed_var, *speed_labels,
                                   command=lambda _: _on_speed_change())
        speed_menu.config(
            font=("Helvetica", 10), fg=TEXT, bg=PANEL,
            activebackground=BORDER, activeforeground=TEXT,
            relief="flat", padx=4, pady=2, cursor="hand2",
            highlightthickness=1, highlightbackground=BORDER,
        )
        speed_menu["menu"].config(
            font=("Helvetica", 10), fg=TEXT, bg=WHITE,
            activebackground=BORDER, activeforeground=TEXT,
        )
        speed_menu.pack(side="right")

        pause_label = "▶ Play" if self._paused else "⏸ Pause"
        self._pause_btn = tk.Button(
            graph_header, text=pause_label,
            font=("Helvetica", 10),
            fg=WHITE, bg=ACCENT2,
            activebackground=ACCENT2_HVR, activeforeground=WHITE,
            relief="flat", padx=10, pady=2, cursor="hand2",
            command=self._toggle_pause,
        )
        self._pause_btn.pack(side="right", padx=4)

        tk.Frame(graph_wrap, bg=BORDER, height=1).pack(fill="x", padx=12)

        self.graph_frame = tk.Frame(graph_wrap, bg=GRAPH_BG)
        self.graph_frame.pack(fill="both", expand=True, padx=10, pady=(4, 8))

        self._error_label = tk.Label(
            self.graph_frame, text="",
            font=("Helvetica", 10), fg=ERROR, bg=GRAPH_BG,
        )

    # ── Theme toggle ──────────────────────────────────────────────────────────

    def _on_theme_toggle(self):
        saved = {
            "frame_idx": self._current_frame_idx,
            "paused": self._paused,
            "speed": self._speed,
        }
        self._stop()
        self.app.toggle_theme(viewer_ctx={
            "trial_path": self.trial_path,
            "video_path": self.video_path,
            "timestamps_path": self.timestamps_path,
            "trial_number": self.trial_number,
            "folder": self.folder,
            "all_trials": self.all_trials,
            "saved_state": saved,
        })

    # ── EMG ───────────────────────────────────────────────────────────────────

    def _load_emg(self):
        try:
            emg = load_emg(self.trial_path)
        except Exception as e:
            self._error_label.config(text=f"Could not load EMG data:\n{e}")
            self._error_label.pack(expand=True)
            return

        present = [c for c in CH_COLORS if c in emg.columns]
        if not present:
            self._error_label.config(
                text="No channel columns found (expected ch1_V, ch2_V, ch3_V)."
            )
            self._error_label.pack(expand=True)
            return

        # optional audio envelope (audio<N>.csv) saved alongside the trial
        self._ax_audio = None
        self._cursor_audio = None
        audio_df = None
        audio_path = self._find_file_local(f"audio{self.trial_number}", [".csv"]) \
            if self.folder else None
        if audio_path and os.path.isfile(audio_path):
            try:
                audio_df = load_audio(audio_path)
            except Exception:
                audio_df = None

        if audio_df is not None:
            fig = Figure(figsize=(10, 5.4), facecolor=GRAPH_FIG, tight_layout=True)
            # EMG gets more room than the sound graph (65 / 35)
            gs = fig.add_gridspec(2, 1, height_ratios=[35, 65])
            ax_audio = fig.add_subplot(gs[0], facecolor=GRAPH_BG)              # sound (top)
            ax = fig.add_subplot(gs[1], facecolor=GRAPH_BG, sharex=ax_audio)   # EMG (bottom)
            self._ax_audio = ax_audio

            ax_audio.plot(audio_df["t_s"], audio_df["amp"],
                          color="#9B59B6", linewidth=0.7)
            ax_audio.set_ylabel("Amplitude", color=GRAPH_TEXT, fontsize=9)
            ax_audio.set_title("Microphone", color=GRAPH_TEXT, fontsize=9)
            ax_audio.tick_params(colors=GRAPH_TEXT, labelsize=8)
            for spine in ax_audio.spines.values():
                spine.set_edgecolor(BORDER)
            ax_audio.grid(True, alpha=0.6, color=GRAPH_GRID)
            ax_audio.set_ylim(0, max(audio_df["amp"].max() * 1.2, 0.02))
            for lbl in ax_audio.get_xticklabels():   # x labels only on the EMG axis
                lbl.set_visible(False)
        else:
            fig = Figure(figsize=(10, 2.8), facecolor=GRAPH_FIG, tight_layout=True)
            ax = fig.add_subplot(111, facecolor=GRAPH_BG)

        lines = []
        for col in present:
            label = col.replace("_V", "").replace("ch", "Channel ")
            (ln,) = ax.plot(
                emg["t_s"], emg[col],
                label=label, linewidth=0.5,
                color=CH_COLORS[col],
            )
            lines.append(ln)

        ax.set_xlabel("Time (s)", color=GRAPH_TEXT, fontsize=9)
        ax.set_ylabel("Voltage (V)", color=GRAPH_TEXT, fontsize=9)
        ax.tick_params(colors=GRAPH_TEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.grid(True, alpha=0.6, color=GRAPH_GRID)

        legend = ax.legend(
            loc="upper right",
            fontsize=8,
            facecolor=PANEL,
            edgecolor=BORDER,
            labelcolor=GRAPH_TEXT,
        )

        leg_map = {}
        for leg_ln, data_ln in zip(legend.get_lines(), lines):
            leg_ln.set_picker(5)
            leg_map[leg_ln] = data_ln

        def _on_pick(event):
            if event.artist not in leg_map:
                return
            data_ln = leg_map[event.artist]
            data_ln.set_visible(not data_ln.get_visible())
            for leg_ln, dl in zip(legend.get_lines(), lines):
                leg_ln.set_alpha(1.0 if dl.get_visible() else 0.25)
            canvas.draw_idle()

        fig.canvas.mpl_connect("pick_event", _on_pick)

        t0 = emg["t_s"].iloc[0]
        self._emg_t0 = t0
        # animated=True keeps the cursors out of the cached background so we can
        # blit just the cursor each frame (full redraws are far too slow)
        self._cursor = ax.axvline(t0, color=CURSOR_CLR, linewidth=1.5,
                                  zorder=5, animated=True)
        self._ax = ax
        if self._ax_audio is not None:
            self._cursor_audio = self._ax_audio.axvline(
                t0, color=CURSOR_CLR, linewidth=1.5, zorder=5, animated=True)

        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()

        toolbar = NavigationToolbar2Tk(canvas, self.graph_frame, pack_toolbar=False)
        toolbar.config(bg=PANEL)
        for child in toolbar.winfo_children():
            try:
                child.config(bg=PANEL, relief="flat")
            except Exception:
                pass
        toolbar.update()
        toolbar.pack(side="bottom", fill="x", padx=8, pady=(0, 4))

        canvas.get_tk_widget().pack(fill="both", expand=True)
        self._canvas = canvas
        self._fig = fig
        self._bg = None

        # cache the static background after every full draw, then paint the
        # cursors on top — lets _advance_cursor blit only the cursor (fast)
        def _on_full_draw(_evt=None):
            self._bg = canvas.copy_from_bbox(fig.bbox)
            ax.draw_artist(self._cursor)
            if self._ax_audio is not None and self._cursor_audio is not None:
                self._ax_audio.draw_artist(self._cursor_audio)
            canvas.blit(fig.bbox)

        canvas.mpl_connect("draw_event", _on_full_draw)

        fig.canvas.mpl_connect("button_press_event", self._on_graph_click)

        def _on_scroll(event):
            axc = event.inaxes
            if axc not in (ax, self._ax_audio):
                return
            factor = 0.85 if event.button == "up" else 1.15
            xc, yc = event.xdata, event.ydata
            x0, x1 = axc.get_xlim()
            y0, y1 = axc.get_ylim()
            # x is shared, so this zooms both graphs together; y is per-axis
            axc.set_xlim(xc + (x0 - xc) * factor, xc + (x1 - xc) * factor)
            axc.set_ylim(yc + (y0 - yc) * factor, yc + (y1 - yc) * factor)
            canvas.draw_idle()

        fig.canvas.mpl_connect("scroll_event", _on_scroll)

        if self.timestamps_path and os.path.isfile(self.timestamps_path):
            try:
                self._frames_df = load_frame_times(self.timestamps_path)
            except Exception:
                self._frames_df = None

    # ── Trial navigation ──────────────────────────────────────────────────────

    def _go_prev(self):
        idx = self.all_trials.index(self.trial_number)
        if idx > 0:
            self._go_to_trial(self.all_trials[idx - 1])

    def _go_next(self):
        idx = self.all_trials.index(self.trial_number)
        if idx < len(self.all_trials) - 1:
            self._go_to_trial(self.all_trials[idx + 1])

    def _go_to_trial(self, n):
        if not self.folder:
            return
        trial_path = self._find_file_local(f"trial{n}")
        if trial_path is None:
            return
        video_path = self._find_file_local(
            f"video{n}", [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"]
        )
        if video_path is None:
            return
        timestamps_path = self._find_file_local(f"video{n}timestamps", [".csv"])
        self._stop()
        self.app.show_viewer(trial_path, video_path, timestamps_path, n,
                             folder=self.folder, all_trials=self.all_trials)

    def _find_file_local(self, stem_lower, extensions=None):
        if extensions is None:
            extensions = [".csv", ".txt", ".dat", ""]
        for ext in extensions:
            for prefix in (stem_lower, stem_lower[0].upper() + stem_lower[1:]):
                path = os.path.join(self.folder, prefix + ext)
                if os.path.isfile(path):
                    return path
        return None

    def _stop(self):
        self._video_running = False
        self._stop_audio()
        if self._after_id:
            self.after_cancel(self._after_id)
        if self._cap:
            self._cap.release()
            self._cap = None

    # ── Checkbox callbacks ────────────────────────────────────────────────────

    def _on_cb_both(self):
        val = self._var_both.get()
        self._var_video.set(val)
        self._var_emg.set(val)
        self._apply_visibility()

    def _on_cb_video(self):
        self._var_both.set(self._var_video.get() and self._var_emg.get())
        self._apply_visibility()

    def _on_cb_emg(self):
        self._var_both.set(self._var_video.get() and self._var_emg.get())
        self._apply_visibility()

    def _apply_visibility(self):
        show_video = self._var_video.get()
        show_emg   = self._var_emg.get()

        if show_video:
            self.video_label.pack(fill="both", expand=True, padx=10, pady=(6, 8))
        else:
            self.video_label.pack_forget()

        if show_emg:
            self._graph_wrap.grid(row=1, column=0, sticky="nsew",
                                  padx=12, pady=(4, 10))
        else:
            self._graph_wrap.grid_remove()

        if show_video and show_emg:
            self._outer.rowconfigure(0, weight=1, minsize=300)
            self._outer.rowconfigure(1, weight=1, minsize=300)
        elif show_video:
            self._outer.rowconfigure(0, weight=1, minsize=0)
            self._outer.rowconfigure(1, weight=0, minsize=0)
        elif show_emg:
            self._outer.rowconfigure(0, weight=0, minsize=160)
            self._outer.rowconfigure(1, weight=1, minsize=0)
        else:
            self._outer.rowconfigure(0, weight=1, minsize=0)
            self._outer.rowconfigure(1, weight=0, minsize=0)

    # ── Graph click seek ──────────────────────────────────────────────────────

    def _on_graph_click(self, event):
        if event.inaxes not in (self._ax, self._ax_audio) or event.xdata is None:
            return
        t_clicked = event.xdata

        if self._cursor is not None:
            self._cursor.set_xdata([t_clicked, t_clicked])
        if self._cursor_audio is not None:
            self._cursor_audio.set_xdata([t_clicked, t_clicked])
        self._blit_cursor()

        if self._frames_df is not None:
            idx = int((self._frames_df["t_s"] - t_clicked).abs().idxmin())
        elif self._cap is not None:
            fps = self._cap.get(cv2.CAP_PROP_FPS) or 30
            rel_t = t_clicked - getattr(self, "_emg_t0", 0)
            idx = max(0, int(rel_t * fps))
        else:
            return

        self._seek_to_frame(idx)

    def _seek_to_frame(self, frame_idx):
        if self._cap is None:
            return
        if self._after_id:
            self.after_cancel(self._after_id)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        self._current_frame_idx = frame_idx

        ret, frame = self._cap.read()
        if ret and self._box_w >= 50 and self._box_h >= 50:
            fh, fw = frame.shape[:2]
            scale = min(self._box_w / fw, self._box_h / fh)
            frame = cv2.resize(frame, (int(fw * scale), int(fh * scale)),
                               interpolation=cv2.INTER_AREA)
            img = ImageTk.PhotoImage(
                Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.video_label.config(image=img)
            self.video_label.image = img
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        self._stop_audio()
        if not self._paused:
            self._begin_playback()

    # ── Video playback ────────────────────────────────────────────────────────

    def _on_outer_configure(self, event):
        if not (self._var_video.get() and self._var_emg.get()):
            return
        h = max(400, event.height)
        self._outer.rowconfigure(0, weight=1, minsize=int(h * 0.35))
        self._outer.rowconfigure(1, weight=1, minsize=int(h * 0.65))

    def _on_video_resize(self, w, h):
        self._box_w = w
        self._box_h = h

    # ── Audio ──────────────────────────────────────────────────────────────────

    def _extract_audio(self):
        """Pull the video's audio track into memory (mono int16) via ffmpeg."""
        if not AUDIO_OK or not self.video_path:
            return
        try:
            ff = imageio_ffmpeg.get_ffmpeg_exe()
            tmp = tempfile.mktemp(suffix=".wav")
            cmd = [ff, "-y", "-i", self.video_path,
                   "-vn", "-ac", "1", "-ar", "44100", "-f", "wav", tmp]
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, creationflags=flags)
            wf = wave.open(tmp, "rb")
            self._audio_rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            wf.close()
            os.remove(tmp)
            self._audio_data = np.frombuffer(frames, dtype=np.int16)
            self._audio = _AudioStreamPlayer(self._audio_data, self._audio_rate)
        except Exception:
            self._audio_data = None
            self._audio_rate = None
            self._audio = None

    def _current_t_s(self):
        idx = self._current_frame_idx
        if self._frames_df is not None and 0 <= idx < len(self._frames_df):
            return float(self._frames_df.loc[idx, "t_s"])
        fps = (self._cap.get(cv2.CAP_PROP_FPS) or 30) if self._cap else 30
        return idx / fps

    def _start_audio(self, t_s):
        # play only at 1x (other speeds would change pitch / break sync)
        if self._audio is None or abs(self._speed - 1.0) > 1e-6:
            return
        try:
            self._audio.start(int(t_s * self._audio_rate))
        except Exception:
            pass

    def _stop_audio(self):
        if self._audio is not None:
            self._audio.stop()

    def _start_video(self):
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            self.video_label.config(
                text="Could not open video.", fg=ERROR, font=("Helvetica", 11)
            )
            return

        # pull the audio track out of the video once
        self._extract_audio()

        # Restore position if coming from a theme toggle
        start_idx = self._saved_state.get("frame_idx", 0)
        if start_idx > 0:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
            self._current_frame_idx = start_idx

        if self._paused:
            # Show the frame but don't start playing
            ret, frame = self._cap.read()
            if ret and self._box_w >= 50 and self._box_h >= 50:
                fh, fw = frame.shape[:2]
                scale = min(self._box_w / fw, self._box_h / fh)
                frame = cv2.resize(frame, (int(fw * scale), int(fh * scale)),
                                   interpolation=cv2.INTER_AREA)
                img = ImageTk.PhotoImage(
                    Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                self.video_label.config(image=img)
                self.video_label.image = img
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
            return

        self._begin_playback()

    def _show_frame(self, frame):
        if self._box_w < 50 or self._box_h < 50:
            return
        fh, fw = frame.shape[:2]
        scale = min(self._box_w / fw, self._box_h / fh)
        frame = cv2.resize(frame, (int(fw * scale), int(fh * scale)),
                           interpolation=cv2.INTER_AREA)
        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self.video_label.config(image=img)
        self.video_label.image = img

    def _begin_playback(self):
        """Start playing from the current position. At 1x the AUDIO is the master
        clock (video/cursor follow the audio's true playback position, so they
        stay in sync despite output latency). At other speeds a wall clock is
        used and audio is muted. Frames are skipped if drawing falls behind."""
        if self._cap is None:
            return
        if self._after_id:
            self.after_cancel(self._after_id)
        self._video_running = True
        self._play_t0 = time.perf_counter()
        self._play_t_origin = self._current_t_s()
        self._audio_master = (abs(self._speed - 1.0) < 1e-6 and self._audio is not None)
        self._start_audio(self._play_t_origin)
        self._play_tick()

    def _play_tick(self):
        if not self._video_running or self._cap is None or self._paused:
            return
        if self._box_w < 50 or self._box_h < 50:
            self._after_id = self.after(30, self._play_tick)
            return

        # where we should be on the timeline right now
        if self._audio_master and self._audio is not None and self._audio.active():
            target_t = self._audio.position_s()        # follow the real audio
        else:
            elapsed = time.perf_counter() - self._play_t0
            target_t = self._play_t_origin + elapsed * self._speed

        # the frame index that corresponds to target_t
        if self._frames_df is not None and len(self._frames_df):
            ts = self._frames_df["t_s"].values
            target_frame = int(np.searchsorted(ts, target_t, side="right")) - 1
            target_frame = max(0, target_frame)
            last_frame = len(ts) - 1
        else:
            fps = self._cap.get(cv2.CAP_PROP_FPS) or 30
            target_frame = max(0, int(target_t * fps))
            last_frame = None

        # read forward to the target frame (dropping intermediate frames keeps
        # timing correct even if decode/draw is slow)
        frame = None
        while self._current_frame_idx <= target_frame:
            ret, f = self._cap.read()
            if not ret:
                self._video_running = False
                self._stop_audio()
                return
            frame = f
            self._current_frame_idx += 1

        if frame is not None:
            self._show_frame(frame)
            self._advance_cursor(self._current_frame_idx - 1)

        if last_frame is not None and self._current_frame_idx > last_frame:
            self._video_running = False
            self._stop_audio()
            return

        self._after_id = self.after(15, self._play_tick)

    def _advance_cursor(self, idx):
        if self._cursor is None or self._frames_df is None:
            return
        df = self._frames_df
        if idx >= len(df):
            return
        t_s = float(df.loc[idx, "t_s"])
        self._cursor.set_xdata([t_s, t_s])
        if self._cursor_audio is not None:
            self._cursor_audio.set_xdata([t_s, t_s])
        self._blit_cursor()

    def _blit_cursor(self):
        """Repaint only the cursor(s) via blitting — avoids full figure redraws."""
        c = getattr(self, "_canvas", None)
        if c is None:
            return
        if self._bg is None:
            c.draw()          # populates the cached background via draw_event
            return
        c.restore_region(self._bg)
        self._ax.draw_artist(self._cursor)
        if self._ax_audio is not None and self._cursor_audio is not None:
            self._ax_audio.draw_artist(self._cursor_audio)
        c.blit(self._fig.bbox)

    def _toggle_pause(self):
        self._paused = not self._paused
        if self._paused:
            self._pause_btn.config(text="▶ Play")
            self._stop_audio()
        else:
            self._pause_btn.config(text="⏸ Pause")
            if not self._video_running:
                self._reset_video()
                return
            self._begin_playback()

    def _reset_video(self):
        if self._cap is None:
            return
        if self._after_id:
            self.after_cancel(self._after_id)
        self._paused = False
        self._pause_btn.config(text="⏸ Pause")
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._current_frame_idx = 0
        self._begin_playback()

    def _on_back(self):
        self._stop()
        self.app.show_home()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = EMGViewer()
    app.mainloop()
