# EMG Trial Logger — System Flowchart

> Run with: `python main.py`  
> Modules: `main.py` → `app.py` + `config.py` + `emg_serial.py` + `camera.py` + `audio_capture.py` + `theme.py`

---

## Master diagram — full system at a glance

```mermaid
flowchart TB
    classDef user fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    classDef gui fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef thread fill:#e0e7ff,stroke:#4f46e5,color:#312e81
    classDef disk fill:#d1fae5,stroke:#059669,color:#064e3b
    classDef hw fill:#fce7f3,stroke:#db2777,color:#831843

    START([▶ python main.py]) --> BOOT[Boot PyQt5 GUI<br/>TrialLoggerApp]
    BOOT --> INIT[Start 3 background sources<br/>+ 2 display timers]

  subgraph HW["Hardware / inputs"]
        MCU{{EMG microcontroller<br/>USB serial COM3}}:::hw
        CAMHW{{USB webcam}}:::hw
        MICHW{{Microphone}}:::hw
        SIM{{Or SIMULATE=True<br/>fake flat EMG}}:::hw
    end

  subgraph THREADS["Background threads — always running"]
        direction TB
        EMGT["emg_serial.py<br/>Read lines → parse 3 ADC values<br/>→ volts → callback"]:::thread
        CAMT["camera.py<br/>Capture frames → live preview queue<br/>→ write AVI when recording"]:::thread
        MICT["audio_capture.py<br/>Mic blocks → WAV when recording<br/>→ amplitude envelope"]:::thread
    end

  subgraph MAIN["Main thread — GUI + state machine"]
        direction TB
        UI["Widgets: object name, buttons,<br/>video pane, mic plot, EMG plot"]:::gui
        STATE{{Trial state}}:::gui
        SAMPLE["_on_sample_threadsafe<br/>buffer EMG if LOGGING<br/>or roll preview window"]:::gui
        REFRESH["Timers: plot ~33 ms, camera ~30 fps"]:::gui
    end

  subgraph USERFLOW["User trial workflow"]
        direction LR
        U1["① Type object name"]:::user
        U2["② Start trial"]:::user
        U3["③ Pause / Resume optional"]:::user
        U4["④ End trial"]:::user
        U5["⑤ Next trial or Review"]:::user
        U1 --> U2 --> U3 --> U4 --> U5
    end

  subgraph SAVE["On End trial — write to disk"]
        direction TB
        F1["trialN.txt<br/>t_ns, ch1, ch2, ch3"]:::disk
        F2["videoN.avi + timestamps.csv"]:::disk
        F3["audioN.csv + mux audio into video"]:::disk
        FOLDER[("trial_logs/ObjectData-date/")]:::disk
        F1 & F2 & F3 --> FOLDER
    end

    MCU --> EMGT
    SIM --> EMGT
    CAMHW --> CAMT
    MICHW --> MICT

    INIT --> EMGT & CAMT & MICT
    INIT --> UI

    EMGT -->|"ch1, ch2, ch3, timestamp"| SAMPLE
    SAMPLE --> REFRESH
    CAMT --> REFRESH
    MICT --> REFRESH
    REFRESH --> UI

    UI --> STATE
    USERFLOW --> STATE
    STATE -->|"LOGGING / PAUSED / ENDED"| SAVE

    U5 -->|click ✓ button| REVIEW[Review mode<br/>frozen graphs + open video]:::gui
    REVIEW --> UI
```

---

## Trial state machine

```mermaid
stateDiagram-v2
    direction LR

    [*] --> Idle: App opens

    Idle --> Logging: Start trial
    Logging --> Paused: Pause
    Paused --> Logging: Resume
    Logging --> Ended: End trial
    Paused --> Ended: End trial

    Ended --> Idle: Start next trial
    Ended --> Logging: Auto mode ON

    Logging --> Idle: Redo
    Paused --> Idle: Redo
    Ended --> Idle: Redo

    Idle --> Preview: Preview ON
    Ended --> Preview: Preview ON
    Preview --> Idle: Preview OFF

    state Logging {
        [*] --> CapEMG: Buffer EMG samples
        CapEMG --> CapAV: Record video + mic
        note right of CapEMG: Serial keeps streaming.\nSamples appended with\nt_ns timestamps.
    }

    state Paused {
        [*] --> Hold: EMG not buffered
        note right of Hold: Video + audio recording\npaused. Clock excludes gap.
    }

    state Ended {
        [*] --> Save: Write all files
        Save --> Badge: Show green ✓ button
    }

    state Preview {
        [*] --> Live: Rolling window only
        note right of Live: Live EMG on screen.\nNothing saved to disk.
    }
```

---

## Data paths — how each signal reaches the screen

```mermaid
flowchart LR
    classDef src fill:#fce7f3,stroke:#db2777
    classDef proc fill:#e0e7ff,stroke:#4f46e5
    classDef ui fill:#fef3c7,stroke:#d97706

    subgraph EMGpath["EMG path"]
        S1["Serial line<br/>2048,1536,1024"]:::src
        S2["ADC → volts<br/>v = raw/4095 × 3.0V"]:::proc
        S3["3 curves on EMG plot"]:::ui
        S1 --> S2 --> S3
    end

    subgraph VIDpath["Video path"]
        V1["OpenCV frame"]:::src
        V2["Resize + RGB"]:::proc
        V3["QLabel live preview<br/>+ AVI if recording"]:::ui
        V1 --> V2 --> V3
    end

    subgraph AUDpath["Audio path"]
        A1["Mic float samples"]:::src
        A2["Peak per ~10 ms chunk"]:::proc
        A3["Mic amplitude plot<br/>+ WAV if recording"]:::ui
        A1 --> A2 --> A3
    end

    subgraph SYNC["Time alignment"]
        T["All use perf_counter()<br/>Pause excludes gaps<br/>from trial clocks"]:::proc
    end

    S2 -.-> T
    V2 -.-> T
    A2 -.-> T
```

---

## Button map — what each control does

```mermaid
flowchart TD
    classDef btn fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    classDef act fill:#d1fae5,stroke:#059669,color:#064e3b

    ROOT["Control bar"]:::btn

    ROOT --> B1["Start / Pause / Resume"]:::btn
    ROOT --> B2["End Trial"]:::btn
    ROOT --> B3["Redo Trial"]:::btn
    ROOT --> B4["Start Next ▶"]:::btn
    ROOT --> B5["Preview"]:::btn
    ROOT --> B6["Auto On/Off"]:::btn
    ROOT --> B7["Auto Y"]:::btn
    ROOT --> B8["Completed ✓ buttons"]:::btn

    B1 -->|Idle| A1["Create object folder<br/>Start video + audio record<br/>Clear EMG buffers"]:::act
    B1 -->|Logging| A2["Pause recording"]:::act
    B1 -->|Paused| A3["Resume recording"]:::act

    B2 --> A4["Stop AV, save txt/csv/avi<br/>Mux audio into video"]:::act

    B3 --> A5["Delete this trial's files<br/>Remove ✓ button<br/>Stay on same trial #"]:::act

    B4 --> A6["Reset to Idle<br/>Ready for trial N+1"]:::act

    B5 -->|ON| A7["Live rolling EMG window<br/>No disk writes"]:::act
    B5 -->|OFF| A8["Return to previous state"]:::act

    B6 --> A9["After End → auto-start<br/>next trial"]:::act

    B7 --> A10["Lock or auto-scale<br/>EMG Y axis"]:::act

    B8 --> A11["Freeze graphs<br/>Open trial video"]:::act
```

---

## End-to-end sequence — one complete trial

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant GUI as TrialLoggerApp
    participant EMG as emg_serial
    participant Cam as camera
    participant Mic as audio_capture
    participant Disk as trial_logs/

    User->>GUI: Type object "Apple"
    User->>GUI: Press Start Trial 1

    GUI->>Disk: Create AppleData-{date}/
    GUI->>Cam: start_recording(video1.avi)
    GUI->>Mic: start_recording(_audio1.wav)
    GUI->>EMG: flush_input()
    Note over GUI: state = LOGGING

    loop Every EMG sample (~1000 Hz)
        EMG->>GUI: callback(ch1, ch2, ch3, t)
        GUI->>GUI: append to buffers
    end

    loop Every 33 ms
        GUI->>GUI: refresh EMG + mic plots
    end

    loop Every 33 ms
        Cam->>GUI: latest RGB frame
        GUI->>GUI: show on video label
    end

    opt User pauses
        User->>GUI: Pause
        GUI->>Cam: pause_recording()
        GUI->>Mic: pause_recording()
        Note over GUI: state = PAUSED
    end

    User->>GUI: End Trial

    GUI->>Cam: stop_recording()
    GUI->>Mic: stop_recording()
    GUI->>Disk: trial1.txt
    GUI->>Disk: video1timestamps.csv
    GUI->>Disk: audio1.csv
    GUI->>Disk: ffmpeg mux → video1.avi
    GUI->>GUI: Add "Apple Trial 1 ✓" button
    Note over GUI: state = ENDED

    opt Review
        User->>GUI: Click ✓ button
        GUI->>GUI: Show saved EMG + audio graphs
        User->>GUI: Open trial video
    end

    User->>GUI: Start Next Trial 2
    Note over GUI: state = IDLE → LOGGING
```

---

## Quick reference

| Piece | File | What it does |
|-------|------|----------------|
| Entry | `main.py` | Starts Qt event loop |
| Settings | `config.py` | Ports, camera, ADC scale, save folder |
| GUI + logic | `app.py` | State machine, plots, save, review |
| EMG input | `emg_serial.py` | Serial MCU or simulated source |
| Video | `camera.py` | Webcam preview + AVI per trial |
| Audio | `audio_capture.py` | Mic WAV + envelope plot |
| Look | `theme.py` | Light / dark colors |

**Output folder example:**
```
trial_logs/
  AppleData-2026-06-22 09-53-33 AM/
    trial1.txt
    video1.avi
    video1timestamps.csv
    audio1.csv
    trial2.txt
    ...
```
