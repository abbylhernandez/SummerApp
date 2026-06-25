"""Configuration constants for the EMG Trial Logger (edit to match hardware)."""

from zoneinfo import ZoneInfo

# Set SIMULATE = True to test the GUI WITHOUT any EMG hardware connected.
# It feeds flat (straight-line) values for all three channels.
SIMULATE        = False
SIM_RATE_HZ     = 1000              # fake sample rate while simulating
SIM_LEVELS      = (0.5, 1.5, 2.5)   # straight-line value (V) for ch1, ch2, ch3

SERIAL_PORT     = "COM3"            # USB serial port of the EMG microcontroller
BAUD_RATE       = 500000
START_CMD       = b"z"              # 'z' = stream raw ADC (3 ch); 'c' adds prediction RX
STOP_CMD        = b"v"              # byte sent to MCU to stop streaming

CAM_INDEX       = 1                 # 1 = Logi C270 USB cam (0 = laptop built-in); None = auto
CAM_PROBE_MAX   = 4                 # how many indices to probe when auto-detecting
CAM_WIDTH       = 640
CAM_HEIGHT      = 360
CAM_FPS         = 30

AUDIO_DEVICE    = None              # mic input device (None = system default)
AUDIO_RATE      = 44100             # sample rate (Hz)
AUDIO_BLOCK     = 1024              # samples per audio callback block
AUDIO_ENV_CHUNK = 441               # samples per envelope point (~10 ms at 44.1 kHz)
AUDIO_MIN_SPAN  = 0.02              # smallest height so silence doesn't over-zoom

VREF            = 3.0               # ADC reference voltage
EMG_Y_MIN       = -0.5              # fixed EMG plot y-axis when Auto Y is off
EMG_Y_MAX       = 3.5
ADC_RES         = 4095.0            # 12-bit ADC full scale
ADC_MIN         = 0
ADC_MAX         = 4095

PREVIEW_POINTS  = 1500              # rolling window length for preview mode
PLOT_UPDATE_MS  = 33                # graph redraw interval

SAVE_ROOT       = "trial_logs"      # where session folders are created
PACIFIC_TZ      = ZoneInfo("America/Los_Angeles")  # Pacific time (PST/PDT)
