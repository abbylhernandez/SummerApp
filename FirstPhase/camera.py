"""Threaded USB camera capture with per-trial video recording (Windows DirectShow)."""

import time
import queue
import logging
import threading

import cv2

from config import CAM_INDEX, CAM_PROBE_MAX, CAM_WIDTH, CAM_HEIGHT, CAM_FPS


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
        """Choose the OpenCV camera backend for the current OS. Windows = DirectShow, Linux = V4L2, Mac = AVFoundation."""
        import platform
        os_name = platform.system()
        if os_name == "Windows":
            return cv2.CAP_DSHOW
        elif os_name == "Linux":
            return cv2.CAP_V4L2
        elif os_name == "Darwin":
            return cv2.CAP_AVFOUNDATION
        else:
            logging.warning("⚠️ Unknown OS '%s'; using default OpenCV camera backend.", os_name)
            return 0


    def _detect_indices(self, limit=CAM_PROBE_MAX):
        """Return indices that actually deliver frames (DirectShow)."""
        found = []
        backend = self._backend_for_current_os() #detect the backend for the current OS

        for i in range(limit):
            try: 
                cap = cv2.VideoCapture(i, backend) 
            
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        found.append(i)
                cap.release()
            except Exception:
                continue # ignore any exceptions and keep probing
        return found

    def start(self):
        # resolve the camera index (auto-pick the external one when not set)
        if self.cam_index is None:
            found = self._detect_indices()
            logging.info("📷 Cameras detected at indices: %s", found)
            self.cam_index = max(found) if found else 0
            logging.info("📷 Using camera index %s", self.cam_index)

        backend = self._backend_for_current_os()

        try:
            #use the detected backend for the current OS to open the camera
            self.cap = cv2.VideoCapture(self.cam_index, backend)
        except Exception: 
            #fallback to default backend if the specified one fails
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
