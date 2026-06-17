# SummerApp
Summer 2026 EMG Application

Planned Changes:
FirstPhase
- preview w/o saving a trial
- add sound recording
- add ability to redo a certain trial (deletes trial u just saves and records a new one in its place)
- save trials to folder with naming scheme "SetDataMM-SS" (or some timestamp)

Labelling
- run program without requiring path of file as the terminal input
- Save Clip should save the current trial and automatically increment to the next trial #; but keep Load Another Trial button as an option to be able to navigate to another one to edit
- EMG graph navigation; bind the graph view to the begin/end time and +1.5V/-1.5V y-axis, make it impossible to navigate out of bounds

We Are Going Home
- change absolute path to the folder containing trials in main()
- change how to access weights (float Wg_init[24], float Cg_init[2], float xstd_init[12], float xmean_init[12]); in order to quickly copy to 

New Program: Realtime Testing (replaces the tera term step)
- a python GUI that combines live emg graph, predictions (1s and 2s via UART) and video and voice (from video camera) (similar to FirstPhase,py)
- "EMGliveviewer" records EMG prediction and video and synchronizes them, (but theres a better way to implement it)
