"""EMG sources: real serial handler and a simulated (no-hardware) handler.

Both fire callback(ch1_v, ch2_v, ch3_v, t_perf) on a background thread, where
t_perf is perf_counter() captured at the moment of arrival.
"""

import time
import logging
import threading

import serial

from config import (SERIAL_PORT, BAUD_RATE, START_CMD, STOP_CMD,
                    ADC_RES, ADC_MIN, ADC_MAX, VREF, SIM_LEVELS, SIM_RATE_HZ)


class SerialEMGHandler:
    """Reads comma-separated lines from the MCU and converts the first three
    integer fields to voltages (extra trailing fields are ignored)."""

    def __init__(self, port=SERIAL_PORT, baudrate=BAUD_RATE):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.thread = None
        self.running = False
        self.callback = None

    def set_callback(self, func):
        self.callback = func

    def flush_input(self):
        """Drop any buffered serial data so a trial starts on current samples."""
        try:
            if self.ser and self.ser.is_open:
                self.ser.reset_input_buffer()
        except Exception as e:
            logging.warning("⚠️ Could not flush serial input: %s", e)

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
        # Drain the entire OS serial buffer each pass and parse all complete
        # lines, rather than one slow readline() at a time. This keeps the
        # buffer near-empty so a sample's arrival time ≈ its real time.
        buf = bytearray()
        while self.running and self.ser is not None:
            try:
                n = self.ser.in_waiting
                chunk = self.ser.read(n if n > 0 else 1)
                if not chunk:
                    continue

                t_perf = time.perf_counter()
                buf.extend(chunk)

                while True:
                    nl = buf.find(b"\n")
                    if nl < 0:
                        break
                    line = buf[:nl].decode("utf-8", errors="ignore").strip()
                    del buf[:nl + 1]

                    if not line or line.count(",") < 2:
                        continue
                    parts = line.split(",")
                    try:
                        raws = [int(parts[i]) for i in range(3)]
                    except ValueError:
                        continue
                    if any(v < ADC_MIN or v > ADC_MAX for v in raws):
                        continue

                    volts = [(v / ADC_RES) * VREF for v in raws]
                    if self.callback:
                        self.callback(volts[0], volts[1], volts[2], t_perf)

                if len(buf) > 65536:
                    del buf[:-256]

            except (serial.SerialException, OSError) as e:
                logging.error("❌ Serial connection lost (%s). Stopping reader.", e)
                self.running = False
                break
            except Exception as e:
                logging.warning("⚠️ Skipping bad serial data: %s", e)
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


class SimulatedEMGHandler:
    """Drop-in replacement for SerialEMGHandler used for testing without an EMG
    device. Fires flat values at SIM_RATE_HZ so each channel is a straight line."""

    def __init__(self, levels=SIM_LEVELS, rate_hz=SIM_RATE_HZ):
        self.levels = levels
        self.period = 1.0 / float(rate_hz)
        self.callback = None
        self.running = False
        self.thread = None

    def set_callback(self, func):
        self.callback = func

    def flush_input(self):
        pass

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
                self.callback(c1, c2, c3, time.perf_counter())
            time.sleep(self.period)

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        logging.info("🧪 Simulated EMG source stopped.")
