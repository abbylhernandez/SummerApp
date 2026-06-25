"""Backwards-compatible shim. The app is now split across modules:

    config.py        - settings/constants
    camera.py        - CameraCapture
    audio_capture.py - AudioCapture
    emg_serial.py    - SerialEMGHandler / SimulatedEMGHandler
    theme.py         - light/dark palettes
    app.py           - TrialLoggerApp (the GUI)
    main.py          - entry point

Run `python main.py` (or this file) to launch.
"""

from main import main

if __name__ == "__main__":
    main()
