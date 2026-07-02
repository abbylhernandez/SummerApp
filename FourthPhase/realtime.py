import sys
import logging
from datetime import datetime
from pathlib import Path
import threading
import queue
import os
from collections import deque
import time

import serial
import cv2
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg


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

        # Serial
        self.serial = SerialEMGHandler(port=port, baudrate=baudrate)
        self.serial.set_sample_callback(self.on_sample)

        # Camera
        self.camera = CameraCapture(cam_index=0, width=640, height=360, fps=30)
        self.cam_timer = QtCore.QTimer(self)
        self.cam_timer.timeout.connect(self.update_camera_frame)

        # Start camera ONCE at app startup
        if self.camera.start():
            self.cam_timer.start(30)
        else:
            QtWidgets.QMessageBox.warning(self, "Camera Error", "Could not start camera.")

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
        self.pred_label = QtWidgets.QLabel("Last prediction: -")
        self.button_label = QtWidgets.QLabel("Last button: -")
        self.emg_label = QtWidgets.QLabel("Last EMG: -")
        self.plot_title = QtWidgets.QLabel("EMG Plot (Very Low Resolution)")
        self.video_label = QtWidgets.QLabel("Camera starting...")
        self.video_label.setFixedSize(640, 360)
        self.video_label.setStyleSheet("background-color: black; color: white;")

        # Low-cost EMG plot (downsampled + timer-driven redraw)
        pg.setConfigOptions(antialias=False)
        self.emg_plot = pg.PlotWidget()
        self.emg_plot.setFixedHeight(170)
        self.emg_plot.setBackground((20, 20, 20))
        self.emg_plot.showGrid(x=True, y=True, alpha=0.2)
        self.emg_plot.setLabel("left", "V")
        self.emg_plot.setLabel("bottom", "Low-res index")
        self.emg_plot.setYRange(0.0, 3.0, padding=0)
        self.emg_plot.getPlotItem().setDownsampling(auto=False, ds=1, mode='peak')
        self.emg_plot.getPlotItem().setClipToView(True)

        self.plot_curve1 = self.emg_plot.plot([], [], pen=pg.mkPen((0, 255, 255), width=1))
        self.plot_curve2 = self.emg_plot.plot([], [], pen=pg.mkPen((255, 165, 0), width=1))
        self.plot_curve3 = self.emg_plot.plot([], [], pen=pg.mkPen((0, 255, 0), width=1))

        # Layout
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)

        label_layout = QtWidgets.QVBoxLayout()
        label_layout.addWidget(self.pred_label)
        label_layout.addWidget(self.button_label)
        label_layout.addWidget(self.emg_label)
        label_layout.addWidget(self.plot_title)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(btn_layout)
        main_layout.addLayout(label_layout)
        main_layout.addWidget(self.emg_plot)
        main_layout.addWidget(self.video_label)
        self.setLayout(main_layout)

        # Connections
        self.start_btn.clicked.connect(self.handle_start)
        self.stop_btn.clicked.connect(self.handle_stop)

        self.emg_sig.connect(self.update_emg_label)
        self.plot_sig.connect(self.on_plot_sample)
        self.pred_sig.connect(self.update_pred_labels)

        self.plot_timer = QtCore.QTimer(self)
        self.plot_timer.timeout.connect(self.refresh_plot)
        self.plot_timer.start(self.PLOT_UPDATE_MS)

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

        self.pred_label.setText("Last prediction: -")
        self.button_label.setText("Last button: -")
        self.emg_label.setText("Last EMG: -")

        logging.info(f"✅ Started new trial {trial_idx} in {self.current_set_dir}")

    def handle_stop(self):
        if not self.collecting:
            return

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
