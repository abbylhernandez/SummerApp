import ctypes
import os
import array
from pathlib import Path
import logging
import shutil
import random
import re
from collections import defaultdict

# -------------------------------------------------
#  LOGGER SETUP (module-level)
# -------------------------------------------------
logger = logging.getLogger(__name__)


class trainclass:
    def __init__(self):
        # Create Report File
        with open("ReportM.txt", "w") as f:
            f.write("Report File\n")

        # Folders
        self.data_set_location = None
        self.testing_folder = None
        self.train_folder = None
        self.last_train_accuracy = None

        # === Classifier and Signal Parameters ===
        self.num_class = 2
        self.channel = 3
        self.wl = 100
        self.winc = 50
        self.tdfeatureN = 4

        # === Feature Extraction Settings ===
        self.deadzone_zc = 0.025
        self.deadzone_turn = 0.015
        self.scale_zc = 15
        self.scale_mav = 2


        # === Derived Parameters ===
        self.feature_dim = self.tdfeatureN * self.channel
        self.trial_per_class = 2
        self.data_per_trial = 2000
        self.win_per_trial = (
            self.data_per_trial // self.winc
            - (self.wl // self.winc - 1)
        )
        total_windows = self.num_class * self.trial_per_class * self.win_per_trial

        logger.debug(
            "Initial derived params: feature_dim=%d, win_per_trial=%d, "
            "total_windows=%d",
            self.feature_dim, self.win_per_trial, total_windows
        )

        # === Data Storage ===
        self.traindata = array.array('f', [0.0] * (self.channel * self.data_per_trial))
        self.testdata = array.array('f', [0.0] * (self.channel * self.data_per_trial))
        self.trainclass = array.array('i', [0] * total_windows)
        self.featurematrix = array.array('f', [0.0] * (total_windows * self.feature_dim))

        # Normalization and LDA Parameters
        self.xmean = array.array('f', [0.0] * self.feature_dim)
        self.xstd = array.array('f', [1.0] * self.feature_dim)
        self.Wg = array.array('f', [0.0] * (self.num_class * self.feature_dim))
        self.Cg = array.array('f', [0.0] * self.num_class)

        self.trial_feature_ranges = {}

        # === Load DLL ===
        dll_path = os.path.abspath("libfunctions.dll")
        try:
            self.lib = ctypes.CDLL(dll_path)
            logger.info("Loaded DLL: %s", dll_path)
        except OSError as e:
            logger.warning(
                "Could not load DLL at %s. Running without C functions. Error: %s",
                dll_path, e
            )
            self.lib = None

        # === Function Signatures (only if DLL loaded) ===
        if self.lib is not None:

            self.lib.tdfeats.argtypes = [
                ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_float, ctypes.c_float,
                ctypes.c_int, ctypes.c_int,
                ctypes.c_int
            ]

            self.lib.feature_normalization.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int, ctypes.c_int
            ]


            self.lib.LDA_train.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int
            ]

            self.lib.LDA_train_accuracy.argtypes = self.lib.LDA_train.argtypes
            self.lib.LDA_train_accuracy.restype = ctypes.c_float

            self.lib.LDA_test.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int,
                ctypes.c_float, ctypes.c_float,
                ctypes.c_int, ctypes.c_int, ctypes.c_int
            ]
            self.lib.LDA_test.restype = ctypes.c_int


    # -------------------------------------------------
    # Trial parsing helpers
    # -------------------------------------------------
    def _split_trial_fields(self, line: str):
        """
        Split one trial line into fields.
        Supports comma-separated new format and whitespace old formats.
        """
        if "," in line:
            parts = [p.strip() for p in line.split(",")]
            # tolerate trailing comma without inventing an empty column
            while parts and parts[-1] == "":
                parts.pop()
            return parts
        return line.split()

    def _infer_num_channels_from_header_line(self, header_line: str) -> int:
        """
        Infer channel count from header line.
        Handles headers like:
            timestamp ch1 ch2 ch3
            timestamp ch1 ch2 ch3 button
            t_ns,ch1_V,ch2_V,ch3_V
            t_ns,ch1_V,ch2_V,ch3_V,button
        """
        cols = self._split_trial_fields(header_line)
        if len(cols) < 2:
            raise ValueError(f"Invalid header: {header_line}")

        cols_lc = [c.lower() for c in cols]
        ch_cols = [c for c in cols_lc if c.startswith("ch")]
        if ch_cols:
            return len(ch_cols)

        # Fallback: first column is time, remaining are channels (+ optional button)
        num_channels = len(cols) - 1
        if cols_lc[-1] in ("button", "btn"):
            num_channels -= 1
        return max(num_channels, 0)

    def _parse_trial_channel_values(
        self,
        line: str,
        expected_channels: int,
        trial_path: Path,
        line_number: int,
    ):
        """
        Parse one data row and return exactly expected_channels floats.
        """
        parts = self._split_trial_fields(line)
        if not parts:
            raise ValueError(f"{trial_path}:{line_number}: empty data line")

        if "," in line:
            # New CSV format: t_ns,ch1,ch2,ch3[,button]
            start_idx = 1
        else:
            # Old text formats:
            #   HH:MM:SS.mmm ch1 ch2 ch3
            #   YYYY-MM-DD HH:MM:SS.mmm ch1 ch2 ch3
            start_idx = 1
            if len(parts) >= expected_channels + 2 and "-" in parts[0] and ":" in parts[1]:
                start_idx = 2

        end_idx = start_idx + expected_channels
        if len(parts) < end_idx:
            raise ValueError(
                f"{trial_path}:{line_number}: expected at least {end_idx} fields, got {len(parts)}"
            )

        vals = []
        for tok in parts[start_idx:end_idx]:
            try:
                vals.append(float(tok))
            except ValueError as e:
                raise ValueError(
                    f"{trial_path}:{line_number}: cannot parse EMG value '{tok}'"
                ) from e
        return vals

    # -------------------------------------------------
    # TRIAL INFO: old + new EMG formats
    # -------------------------------------------------
    def trial_data_info(self, trial_path: Path):
        """
        Given a single trial file in either old/new EMG format,
        Return (num_samples, num_channels).
        """
        logger.debug("Reading trial data info from %s", trial_path)

        with trial_path.open('r') as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            logger.error("Trial file %s is empty", trial_path)
            raise ValueError(f"Trial file {trial_path} is empty")

        header = lines[0]
        try:
            num_channels = self._infer_num_channels_from_header_line(header)
        except ValueError:
            logger.error(
                "Header in %s invalid: %s", trial_path, header
            )
            raise ValueError(
                f"Header in {trial_path} doesn't look like a supported EMG header: {header}"
            )
        if num_channels <= 0:
            raise ValueError(f"Header in {trial_path} has no channel columns: {header}")
        data_lines = lines[1:]
        num_samples = len(data_lines)

        logger.debug(
            "Trial %s -> num_samples=%d, num_channels=%d",
            trial_path, num_samples, num_channels
        )

        return num_samples, num_channels

    # -------------------------------------------------
    # SCAN SET FOLDER: labels/actions and trials
    # -------------------------------------------------
    def label_data_info(self):
        """
        Inspect the set folder and return:
        num_trials_per_label, num_labels, num_samples, num_channels
        """
        if self.train_folder is None:
            logger.error("train Folder is not set before calling label_data_info()")
            raise ValueError("train Folder is not set")

        set_dir = Path(self.train_folder)
        logger.info("Scanning train Folder : %s", set_dir)

        act_dirs = [
            d for d in set_dir.iterdir()
            if d.is_dir() and d.name.lower().startswith("act")
        ]
        if not act_dirs:
            logger.error("No act* folders found in %s", set_dir)
            raise FileNotFoundError(f"No act* folders found in {set_dir}")

        act_dirs.sort()
        num_labels = len(act_dirs)
        logger.info("Found %d label folders (actions): %s",
                    num_labels, [d.name for d in act_dirs])

        expected_num_trials = None
        expected_num_samples = None
        expected_num_channels = None

        for act_dir in act_dirs:
            trial_files = sorted(act_dir.glob("trial_*.txt"))
            if not trial_files:
                logger.error("No trial_*.txt files in %s", act_dir)
                raise FileNotFoundError(f"No trial_*.txt files in {act_dir}")

            num_trials_here = len(trial_files)
            logger.debug("Action %s has %d trials", act_dir.name, num_trials_here)

            if expected_num_trials is None:
                expected_num_trials = num_trials_here
            elif num_trials_here != expected_num_trials:
                logger.error(
                    "Action folder %s has %d trials but expected %d",
                    act_dir, num_trials_here, expected_num_trials
                )
                raise ValueError(
                    f"Action folder {act_dir} has {num_trials_here} trials, "
                    f"but expected {expected_num_trials}."
                )

            for trial_path in trial_files:
                num_samples, num_channels = self.trial_data_info(trial_path)

                if expected_num_samples is None:
                    expected_num_samples = num_samples
                    expected_num_channels = num_channels
                    logger.debug(
                        "Reference samples/channels set from %s: samples=%d, channels=%d",
                        trial_path, num_samples, num_channels
                    )
                else:
                    if num_samples != expected_num_samples:
                        logger.error(
                            "Inconsistent samples in %s: %d vs expected %d",
                            trial_path, num_samples, expected_num_samples
                        )
                        raise ValueError(
                            f"Trial {trial_path} has {num_samples} samples, "
                            f"expected {expected_num_samples}."
                        )
                    if num_channels != expected_num_channels:
                        logger.error(
                            "Inconsistent channels in %s: %d vs expected %d",
                            trial_path, num_channels, expected_num_channels
                        )
                        raise ValueError(
                            f"Trial {trial_path} has {num_channels} channels, "
                            f"expected {expected_num_channels}."
                        )

        logger.info(
            "Dataset summary: num_labels=%d, trials_per_label=%d, "
            "samples_per_trial=%d, channels=%d",
            num_labels, expected_num_trials, expected_num_samples, expected_num_channels
        )

        return expected_num_trials, num_labels, expected_num_samples, expected_num_channels

    # -------------------------------------------------
    # TOP-LEVEL: SET DATA INFO
    # -------------------------------------------------
    def set_data_info(self):
        """
        High-level helper:
        - uses label_data_info()
        - updates internal settings
        - returns a dict with the results
        """
        num_trials, num_labels, num_samples, num_channels = self.label_data_info()
        self.update_setting(num_labels, num_trials, num_samples, num_channels)

        info = {
            "num_labels": num_labels,
            "num_trials_per_label": num_trials,
            "num_samples_per_trial": num_samples,
            "num_channels": num_channels,
        }
        logger.info("set_data_info(): %s", info)
        return info

    def update_setting(self, num_labels, num_trials, num_samples, num_channels):
        """
        Update internal EMG parameters based on the dataset.
        """
        logger.info(
            "Updating settings from dataset: labels=%d, trials=%d, samples=%d, channels=%d",
            num_labels, num_trials, num_samples, num_channels
        )

        # Basic parameters from dataset (TRAIN set in your case)
        self.num_class = num_labels
        self.trial_per_class = num_trials
        self.data_per_trial = num_samples
        self.channel = num_channels

        # Recompute derived quantities
        self.feature_dim = self.tdfeatureN * self.channel
        self.win_per_trial = (
            self.data_per_trial // self.winc
            - (self.wl // self.winc - 1)
        )

        # ---- Derived "expected" values for the training set ----
        windows_per_trial = self.win_per_trial
        total_windows_per_label = windows_per_trial * self.trial_per_class
        total_windows_train_set = total_windows_per_label * self.num_class

        logger.debug(
            "Derived quantities: feature_dim=%d, win_per_trial=%d, "
            "windows_per_label=%d, total_windows_train=%d",
            self.feature_dim,
            windows_per_trial,
            total_windows_per_label,
            total_windows_train_set
        )

        # ---- Add to Report.txt ----
        self.add_report("\n=== Training Set ENTERED SETTINGS  ===")
        self.add_report(f"  Number of labels (classes): {self.num_class}")
        self.add_report(f"  Trials per label: {self.trial_per_class}")
        self.add_report(f"  Samples per trial: {self.data_per_trial}")
        self.add_report(f"  Channels: {self.channel}")
        self.add_report(f"  TD features per channel: {self.tdfeatureN}")
        self.add_report(f"  Feature dimension (per window): {self.feature_dim}")
        self.add_report(f"  Window length (WL): {self.wl}")
        self.add_report(f"  Window increment (WINC): {self.winc}")
        self.add_report(f"  Windows per trial: {windows_per_trial}")
        self.add_report(f"  Total windows per label: {total_windows_per_label}")
        self.add_report(f"  Total windows in training set: {total_windows_train_set}")
        self.add_report("")  # blank line separator


    def split_training_testing(self, type: str, percentage: int):
        """
        Split trials into Train and Test folders based on the given percentage and method.

        Parameters
        ----------
        type : str
            "random"    -> randomly shuffle then split by percentage
            "50/50"     -> first N% go to train, rest go to test
            "alternate" -> alternate (train/test/train...), then adjust to match percentage
        percentage : int
            Percent of files to assign to training set (0–100).
        """
        if not (0 < percentage < 100):
            raise ValueError("percentage must be between 1 and 99")

        if self.data_set_location is None:
            raise ValueError("data_set_location is not set")

        set_dir = Path(self.data_set_location)

        # -------------------------------
        # Reset Train/ and Test/
        # -------------------------------
        train_root = set_dir / "Train"
        test_root = set_dir / "Test"

        if train_root.exists():
            shutil.rmtree(train_root)
        if test_root.exists():
            shutil.rmtree(test_root)

        train_root.mkdir()
        test_root.mkdir()

        self.train_folder = train_root
        self.testing_folder = test_root

        # find actions (act1, act2...)
        act_dirs = [
            d for d in set_dir.iterdir()
            if d.is_dir() and d.name.lower().startswith("act")
        ]

        for act_dir in sorted(act_dirs):
            act_name = act_dir.name

            train_act_dir = train_root / act_name
            test_act_dir = test_root / act_name
            train_act_dir.mkdir()
            test_act_dir.mkdir()

            # trial files
            trial_files = sorted(act_dir.glob("trial_*.txt"))
            if not trial_files:
                continue

            total = len(trial_files)
            num_train = round((percentage / 100) * total)
            files = trial_files[:]

            # ---- Strategy: random ----
            if type.lower() == "random":
                random.shuffle(files)
                train_files = files[:num_train]
                test_files = files[num_train:]

            elif type.lower() == "first_half":
                # match C code exactly
                train_files = files[: total//2]
                test_files  = files[total//2:]

            # ---- Strategy: 50/50 ----
            elif type.lower() == "50/50":
                train_files = files[:num_train]
                test_files = files[num_train:]

            # ---- Strategy: alternate ----
            elif type.lower() == "alternate":
                train_files = []
                test_files = []
                flag = True
                for f in files:
                    (train_files if flag else test_files).append(f)
                    flag = not flag

                # adjust percentage
                if len(train_files) > num_train:
                    excess = len(train_files) - num_train
                    test_files.extend(train_files[-excess:])
                    train_files = train_files[:-excess]
                elif len(train_files) < num_train:
                    needed = num_train - len(train_files)
                    train_files.extend(test_files[:needed])
                    test_files = test_files[needed:]

            else:
                raise ValueError(f"Unknown split type: {type}")

            # ---- Copy files ----
            for f in train_files:
                shutil.copy2(f, train_act_dir / f.name)

            for f in test_files:
                shutil.copy2(f, test_act_dir / f.name)

            # -------------------------------
            # Logging to console + Report.txt
            # -------------------------------
            summary_lines = [
                "",
                f"[{act_name}] Split summary:",
                f"  Total trials: {total}",
                f"  Train: {len(train_files)}",
                f"  Test:  {len(test_files)}",
                f"  Method: {type}, Percentage: {percentage}%",
            ]

            for line in summary_lines:
                self.add_report(line)


    def save_feature_matrix(self, num_rows: int, filename: str):
        """
        Save the current feature matrix to a text file in the SAME format as C:

            for (i = 0; i < num_rows; i++) {
                for (j = 0; j < FEATURE_DIM; j++) {
                    fprintf(file, "%f ", Feature_matrix[j + i*FEATURE_DIM]);
                }
                fprintf(file, "\\n");
            }

        num_rows  = number of windows actually filled (e.g. feat_idx)
        filename  = output txt file path
        """
        if num_rows <= 0:
            #print("⚠️ save_feature_matrix: num_rows <= 0, nothing to save.")
            return

        if len(self.featurematrix) < num_rows * self.feature_dim:
            raise ValueError(
                f"save_feature_matrix: buffer too small: "
                f"len(featurematrix)={len(self.featurematrix)}, "
                f"expected at least {num_rows * self.feature_dim}"
            )

        with open(filename, "w") as f:
            for i in range(num_rows):
                offset = i * self.feature_dim
                row = self.featurematrix[offset: offset + self.feature_dim]

                # Match C's "%f " (6 decimal places)
                line = " ".join(f"{val:.6f}" for val in row)
                f.write(line + "\n")

        #print(f"✅ Saved feature matrix ({num_rows} rows) to {filename}")


    def add_report(self, line):
        with open("Report.txt", "a") as f:
            f.write(line + "\n")

    def load_trial_data(self, trial_path: Path, label: str = "traindata"):
        """
        Load EMG data from old/new trial text formats.

        into either self.traindata or self.testdata, using interleaved layout:
            [s0_ch1, s0_ch2, ..., s0_chC,  s1_ch1, s1_ch2, ..., s1_chC,  ...]

        label: "traindata" or "testdata"
        """
        if label == "traindata":
            buffer = self.traindata
        elif label == "testdata":
            buffer = self.testdata
        else:
            raise ValueError("label must be 'traindata' or 'testdata'")

        trial_path = Path(trial_path)

        with trial_path.open("r") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            raise ValueError(f"Trial file {trial_path} is empty")

        header = lines[0]
        num_channels_in_file = self._infer_num_channels_from_header_line(header)

        if num_channels_in_file != self.channel:
            raise ValueError(
                f"{trial_path}: header has {num_channels_in_file} channels, "
                f"but self.channel={self.channel}"
            )

        data_lines = lines[1:]
        num_samples = len(data_lines)

        if num_samples != self.data_per_trial:
            raise ValueError(
                f"{trial_path}: has {num_samples} samples, expected {self.data_per_trial}"
            )

        values = []
        for line_no, line in enumerate(data_lines, start=2):
            ch_vals = self._parse_trial_channel_values(
                line=line,
                expected_channels=self.channel,
                trial_path=trial_path,
                line_number=line_no,
            )
            values.extend(ch_vals)

        if len(values) != self.channel * self.data_per_trial:
            raise ValueError(
                f"{trial_path}: parsed {len(values)} floats, "
                f"expected {self.channel * self.data_per_trial}"
            )

        # zero out buffer, then fill
        for i in range(len(buffer)):
            buffer[i] = 0.0
        buffer[:len(values)] = array.array("f", values)

        logger.debug(
            "Loaded %s into %s: %d samples x %d channels",
            trial_path, label, num_samples, self.channel
        )
    def generate_windows(self):
        """
        Generates overlapping windows from self.traindata using:
            - window length: self.wl
            - window increment: self.winc

        Returns:
            List[array('f')] where each window is size (wl * channel) floats.
        """
        windows = []
        total_floats = self.channel * self.data_per_trial

        for m in range(self.win_per_trial):
            start = m * self.winc * self.channel
            end = start + self.wl * self.channel
            if end > total_floats:
                break
            window = self.traindata[start:end]  # slice of array('f')
            windows.append(window)

        logger.debug(
            "Generated %d windows (wl=%d, winc=%d) from one trial",
            len(windows), self.wl, self.winc
        )
        return windows
    def extract_features(self, windows, class_idx: int = 0, start_idx: int = 0):
        """
        Extract features from windows and store into self.featurematrix and self.trainclass.

        windows    : list of array('f') windows
        class_idx  : zero-based class index (0..num_class-1)
        start_idx  : starting row index in featurematrix/trainclass
        """
        feat_ptr = ctypes.cast(
            self.featurematrix.buffer_info()[0],
            ctypes.POINTER(ctypes.c_float)
        )
        label_ptr = ctypes.cast(
            self.trainclass.buffer_info()[0],
            ctypes.POINTER(ctypes.c_int)
        )

        for m, window in enumerate(windows):
            row_idx = start_idx + m
            window_ptr = ctypes.cast(
                window.buffer_info()[0],
                ctypes.POINTER(ctypes.c_float)
            )

            self.lib.tdfeats(
                window_ptr,
                ctypes.c_int(self.wl),
                ctypes.c_int(self.channel),
                ctypes.c_int(row_idx),    # Nframe = row index
                feat_ptr,
                ctypes.c_float(self.deadzone_zc),
                ctypes.c_float(self.deadzone_turn),
                ctypes.c_int(self.scale_mav),
                ctypes.c_int(self.scale_zc),
                ctypes.c_int(self.tdfeatureN)
            )

            label_ptr[row_idx] = class_idx + 1  # labels 1..CLASS

    def train_model(self):
        """
        Full training pipeline using Option 1 (window arrays):

        - Assumes split_training_testing() already created Train/actX/trial_*.txt
        - Assumes set_data_info() already called (training set info)
        - For each trial in Train/, loads data, generates windows,
        calls tdfeats on each window, and fills featurematrix + trainclass
        - Normalizes features and trains LDA
        - Writes detailed info into Report.txt
        """
        if self.train_folder is None:
            raise ValueError("train_folder is not set. Run split_training_testing() first.")

        # Recompute (or confirm) derived sizes
        windows_per_trial = self.win_per_trial
        total_windows_per_label = windows_per_trial * self.trial_per_class
        total_windows_train_set = total_windows_per_label * self.num_class

        # Allocate / resize buffers based on current dataset
        self.featurematrix = array.array(
            "f", [0.0] * (total_windows_train_set * self.feature_dim)
        )
        self.trainclass = array.array("i", [0] * total_windows_train_set)

        # Resize normalization + model arrays to exact dims
        self.xmean = array.array("f", [0.0] * self.feature_dim)
        self.xstd = array.array("f", [1.0] * self.feature_dim)
        self.Wg = array.array("f", [0.0] * (self.num_class * self.feature_dim))
        self.Cg = array.array("f", [0.0] * self.num_class)

        self.trial_feature_ranges = {}

        self.add_report("********************************************************")
        self.add_report("*******      Training Session Starts      **************")
        self.add_report("********************************************************")
        self.add_report(f"Train folder: {self.train_folder}")
        self.add_report(
            f"Expected windows: per_trial={windows_per_trial}, "
            f"per_label={total_windows_per_label}, total={total_windows_train_set}"
        )
        self.add_report("")

        logger.info("Starting training over Train folder: %s", self.train_folder)

        feat_idx = 0         # current global feature row index
        trial_count = 0

        # discover classes from Train/act*
        act_dirs = [
            d for d in Path(self.train_folder).iterdir()
            if d.is_dir() and d.name.lower().startswith("act")
        ]
        act_dirs.sort()

        for class_idx, act_dir in enumerate(act_dirs):
            act_name = act_dir.name
            trial_files = sorted(act_dir.glob("trial_*.txt"))
            if not trial_files:
                logger.warning("No trial_*.txt files found in %s", act_dir)
                continue

            #self.add_report(f"--- Label {class_idx + 1} ({act_name}) ---")

            for t_idx, trial_path in enumerate(trial_files, start=1):
                trial_count += 1
                logger.info("Processing trial: %s", trial_path)

                # 1) Load into self.traindata
                self.load_trial_data(trial_path, label="traindata")

                # 2) Generate sliding windows
                windows = self.generate_windows()
                start_row = feat_idx

                # 3) Extract features for these windows
                self.extract_features(windows, class_idx=class_idx, start_idx=feat_idx)
                feat_idx += len(windows)
                end_row = feat_idx - 1

                # Record range for this trial
                trial_key = f"{act_name}/{trial_path.name}"
                self.trial_feature_ranges[trial_key] = (start_row, feat_idx)

                # 4) Report per-trial info
                #self.add_report(f"Trial: {trial_key}")
                #self.add_report(
                #    f"  Windows in this trial: {len(windows)} "
                #    f"(feature rows {start_row}..{end_row})"
                #)

                for ch in range(self.channel):
                    col_start = ch * self.tdfeatureN
                    col_end = (ch + 1) * self.tdfeatureN - 1
                    #self.add_report(
                    #    f"  Channel {ch + 1} feature columns: {col_start}..{col_end}"
                    #)

                #self.add_report("")

        # ----- Summary + sanity check -----
        self.add_report(
            f"Total training trials processed: {trial_count}"
        )
        self.add_report(
            f"Total feature rows filled: {feat_idx} (expected {total_windows_train_set})"
        )

        if feat_idx != total_windows_train_set:
            msg = (
                f"WARNING: feature rows used={feat_idx}, "
                f"expected={total_windows_train_set}"
            )
            logger.warning(msg)
            self.add_report(msg)

        self.add_report("")
        self.add_report("Starting feature normalization and LDA training...")

        # ----- Save raw feature matrix BEFORE normalization (C-style) -----
        self.save_feature_matrix(
            num_rows=feat_idx,
            filename="FeatureMatrix_python_before_norm.txt"
        )

        # ----- Feature normalization -----
        feat_ptr = ctypes.cast(
            self.featurematrix.buffer_info()[0],
            ctypes.POINTER(ctypes.c_float)
        )
        xmean_ptr = ctypes.cast(
            self.xmean.buffer_info()[0],
            ctypes.POINTER(ctypes.c_float)
        )
        xstd_ptr = ctypes.cast(
            self.xstd.buffer_info()[0],
            ctypes.POINTER(ctypes.c_float)
        )
        class_ptr = ctypes.cast(
            self.trainclass.buffer_info()[0],
            ctypes.POINTER(ctypes.c_int)
        )

        self.lib.feature_normalization(
            feat_ptr,
            xmean_ptr,
            xstd_ptr,
            ctypes.c_int(feat_idx),          # num_samples (rows)
            ctypes.c_int(self.feature_dim),  # feature_dim (cols)
        )
        logger.info("Feature normalization complete.")
        self.add_report("Feature normalization complete.")

        # ----- LDA train -----
        Wg_ptr = ctypes.cast(
            self.Wg.buffer_info()[0],
            ctypes.POINTER(ctypes.c_float)
        )
        Cg_ptr = ctypes.cast(
            self.Cg.buffer_info()[0],
            ctypes.POINTER(ctypes.c_float)
        )

        self.lib.LDA_train(
            feat_ptr,
            class_ptr,
            Wg_ptr,
            Cg_ptr,
            ctypes.c_int(self.feature_dim),
            ctypes.c_int(self.num_class),
            ctypes.c_int(self.win_per_trial),
            ctypes.c_int(self.trial_per_class),
        )
        logger.info("LDA training complete.")
        self.add_report("LDA training complete.")

        # ----- LDA training accuracy -----
        accuracy = self.lib.LDA_train_accuracy(
            feat_ptr,
            class_ptr,
            Wg_ptr,
            Cg_ptr,
            ctypes.c_int(self.feature_dim),
            ctypes.c_int(self.num_class),
            ctypes.c_int(self.win_per_trial),
            ctypes.c_int(self.trial_per_class),
        )
        acc_percent = float(accuracy) * 100.0
        self.last_train_accuracy = acc_percent
        logger.info("Training accuracy = %.2f%%", acc_percent)
        self.add_report(f"Training accuracy = {acc_percent:.2f}%")
        self.add_report("")
        self.add_report("=== Normalization Parameters (xmean, xstd) ===")
        for j in range(self.feature_dim):
            self.add_report(
                f"xmean[{j}]: {self.xmean[j]:.4g}\t xstd[{j}]: {self.xstd[j]:.4g}"
            )

        self.add_report("")

        # ----- Dump LDA weights (Wg, Cg) in same layout as C -----
        self.add_report("=== LDA Weights (Wg) and Biases (Cg) ===")
        self.add_report(
            f"Wg shape: {self.num_class} x {self.feature_dim} "
            f"(stored column-major: Wg[i + j * num_class])"
        )

        for i in range(self.num_class):
            self.add_report(f"Wg[{i}]:")
            row_vals = []
            # C layout: Wg[i + j * CLASS], so replicate that indexing
            for j in range(self.feature_dim):
                idx = i + j * self.num_class
                row_vals.append(f"{self.Wg[idx]:.4g}")
            # One line per class with all feature weights
            self.add_report("  " + "\t".join(row_vals))

            # Corresponding bias term
            self.add_report(f"Cg[{i}]: {self.Cg[i]:.4g}")
            self.add_report("")

        self.add_report("=== End of LDA parameter report ===")
        self.add_report("")

    def save_lda_weights_cfile(
        self,
        out_path="my_lda_weights_for_microcontroller.c",
        meta=None,
    ):
        """
        Save LDA parameters (Wg, Cg, xmean, xstd) into a C file
        compatible with STM32 firmware.

        Adds metadata at the top as C comments (dataset, split, accuracy, etc).
        """
        import math
        from datetime import datetime

        num_class = int(self.num_class)
        feat_dim  = int(self.feature_dim)

        # ---------------- SAFETY CHECKS ----------------
        if len(self.Wg) != num_class * feat_dim:
            raise ValueError("Wg size mismatch")
        if len(self.Cg) != num_class:
            raise ValueError("Cg size mismatch")
        if len(self.xmean) != feat_dim or len(self.xstd) != feat_dim:
            raise ValueError("xmean/xstd size mismatch")

        def _check(arr, name):
            for v in arr:
                if not math.isfinite(float(v)):
                    raise ValueError(f"{name} contains NaN or Inf")

        _check(self.Wg, "Wg")
        _check(self.Cg, "Cg")
        _check(self.xmean, "xmean")
        _check(self.xstd, "xstd")

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        meta = dict(meta or {})
        meta.setdefault("generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        meta.setdefault("num_class", num_class)
        meta.setdefault("feature_dim", feat_dim)

        # ---------------- WRITE FILE ----------------
        with open(out_path, "w") as f:
            f.write("// ============================================================\n")
            f.write("// Auto-generated LDA model for STM32\n")
            for k, v in meta.items():
                f.write(f"// {k}: {v}\n")
            f.write("// ============================================================\n\n")

            # -------- Wg_init (column-major, EXACT MATCH) --------
            f.write(f"float Wg_init[{num_class * feat_dim}] = {{\n")
            for j in range(feat_dim):
                line = "    "
                for i in range(num_class):
                    idx = i + j * num_class
                    line += f"{self.Wg[idx]:.6f}, "
                f.write(line.rstrip() + "\n")
            f.write("};\n\n")

            # -------- Cg_init --------
            f.write(f"float Cg_init[{num_class}] = {{\n    ")
            f.write(", ".join(f"{self.Cg[i]:.6f}" for i in range(num_class)))
            f.write("\n};\n\n")

            # -------- xstd_init --------
            f.write(f"float xstd_init[{feat_dim}] = {{\n    ")
            f.write(", ".join(f"{self.xstd[i]:.6f}" for i in range(feat_dim)))
            f.write("\n};\n\n")

            # -------- xmean_init --------
            f.write(f"float xmean_init[{feat_dim}] = {{\n    ")
            f.write(", ".join(f"{self.xmean[i]:.6f}" for i in range(feat_dim)))
            f.write("\n};\n\n")

        print(f"✅ Saved LDA weights to: {out_path}")


    def test_model(self):
        """
        Mirror the C testing phase:

        - Uses self.testing_folder created by split_training_testing()
        - For each actX in Test/, for each trial_*.txt:
            * load_trial_data(..., label="testdata")
            * slide windows with WINC, WL over raw EMG
            * call LDA_test on each window (C code does feature extraction + norm)
        - Computes overall window-level accuracy.
        """
        if self.testing_folder is None:
            raise ValueError("testing_folder is not set. Run split_training_testing() first.")

        test_root = Path(self.testing_folder)

        # ---- Pointers to model + norm params (already filled by train_model) ----
        Wg_ptr = ctypes.cast(
            self.Wg.buffer_info()[0],
            ctypes.POINTER(ctypes.c_float)
        )
        Cg_ptr = ctypes.cast(
            self.Cg.buffer_info()[0],
            ctypes.POINTER(ctypes.c_float)
        )
        xmean_ptr = ctypes.cast(
            self.xmean.buffer_info()[0],
            ctypes.POINTER(ctypes.c_float)
        )
        xstd_ptr = ctypes.cast(
            self.xstd.buffer_info()[0],
            ctypes.POINTER(ctypes.c_float)
        )

        # Raw address of self.testdata (array('f')) buffer
        base_addr = self.testdata.buffer_info()[0]
        elem_size = ctypes.sizeof(ctypes.c_float)

        total_win_num = 0
        num_correct = 0

        self.add_report("********************************************************")
        self.add_report("*******        Testing Session Starts        ***********")
        self.add_report("********************************************************")
        self.add_report(f"Test folder: {self.testing_folder}")
        self.add_report("")

        logger.info("Starting testing over Test folder: %s", self.testing_folder)

        # Discover classes (act1, act2, ...) in Test/
        act_dirs = [
            d for d in test_root.iterdir()
            if d.is_dir() and d.name.lower().startswith("act")
        ]
        act_dirs.sort()

        for class_idx, act_dir in enumerate(act_dirs):
            act_name = act_dir.name
            trial_files = sorted(act_dir.glob("trial_*.txt"))
            if not trial_files:
                logger.warning("No trial_*.txt files found in %s", act_dir)
                continue

            self.add_report(f"--- TEST Label {class_idx + 1} ({act_name}) ---")

            for t_idx, trial_path in enumerate(trial_files, start=1):
                self.add_report(f"Trial: {act_name}/{trial_path.name}")
                logger.info("[TEST] Processing trial: %s", trial_path)

                # 1) Load into self.testdata (NOT traindata)
                self.load_trial_data(trial_path, label="testdata")

                m = 0
                wins_this_trial = 0

                while m < self.win_per_trial:
                    start_float_index = m * self.winc * self.channel
                    end_float_index = start_float_index + self.wl * self.channel

                    # Safety check – don't walk off the end of the buffer
                    if end_float_index > len(self.testdata):
                        break

                    # Compute address of the start of this window
                    offset_addr = base_addr + start_float_index * elem_size
                    window_ptr = ctypes.cast(offset_addr, ctypes.POINTER(ctypes.c_float))

                    # 3) Call LDA_test (C will call tdfeats + apply normalization)
                    decision = self.lib.LDA_test(
                        window_ptr,                 # float* data (raw EMG)
                        Wg_ptr,
                        Cg_ptr,
                        xmean_ptr,
                        xstd_ptr,
                        ctypes.c_int(self.wl),
                        ctypes.c_int(self.channel),
                        ctypes.c_int(self.feature_dim),
                        ctypes.c_int(self.num_class),
                        ctypes.c_float(self.deadzone_zc),
                        ctypes.c_float(self.deadzone_turn),
                        ctypes.c_int(self.scale_mav),
                        ctypes.c_int(self.scale_zc),
                        ctypes.c_int(self.tdfeatureN),
                    )

                    if int(decision) == (class_idx + 1):
                        num_correct += 1

                    total_win_num += 1
                    wins_this_trial += 1
                    m += 1

                # Per-trial summary
                #self.add_report(
                #    f"  Windows tested in this trial: {wins_this_trial} "
                #    f"(win_per_trial={self.win_per_trial})"
                #)
                #self.add_report("")

        # ---- Final accuracy ----
        if total_win_num > 0:
            test_accuracy = float(num_correct) / float(total_win_num)
        else:
            test_accuracy = 0.0

        self.add_report("=== Testing Summary ===")
        self.add_report(f"Total test windows: {total_win_num}")
        self.add_report(f"Correct decisions: {num_correct}")
        self.add_report(f"Testing accuracy (window-level): {test_accuracy:.4f}")
        self.add_report("")

        logger.info(
            "Testing complete. Windows=%d, correct=%d, accuracy=%.4f",
            total_win_num, num_correct, test_accuracy
        )


        return test_accuracy


    def _compute_worst_trials_per_label(self, top_k=1):
        """
        Compute per-trial accuracy using the current trained LDA model and return
        the worst 'top_k' trials per label (e.g., per act1/act2).

        Returns
        -------
        worst_by_label : dict
            { 'act1': ['act1/trial_18.txt', ...], 'act2': [...], ... }
        all_trial_results : list of (trial_key, accuracy)
            Sorted ascending by accuracy.
        """
        import numpy as np

        # Basic model checks before scoring trials.
        if not hasattr(self, "Wg") or not hasattr(self, "Cg"):
            raise RuntimeError("Wg/Cg not found. Run train_model() first.")

        if not hasattr(self, "trial_feature_ranges") or not self.trial_feature_ranges:
            raise RuntimeError("trial_feature_ranges not found. "
                               "Make sure features were extracted with trial info.")

        if not hasattr(self, "featurematrix") or len(self.featurematrix) == 0:
            raise RuntimeError("featurematrix is empty. Run train_model() first.")

        # Convert featurematrix to (num_samples, feature_dim)
        feat = np.array(self.featurematrix, dtype=np.float32)
        if feat.size % self.feature_dim != 0:
            raise ValueError(
                f"featurematrix length {feat.size} is not divisible by feature_dim {self.feature_dim}"
            )
        num_samples = feat.size // self.feature_dim
        feat = feat.reshape((num_samples, self.feature_dim))

        # Labels (1..num_class)
        labels = np.array(self.trainclass, dtype=np.int32)
        if labels.size != num_samples:
            raise ValueError(
                f"trainclass length {labels.size} does not match num_samples {num_samples}"
            )

        # LDA params
        Wg_vec = np.array(self.Wg, dtype=np.float32)
        Cg_vec = np.array(self.Cg, dtype=np.float32)
        Wg_mat = Wg_vec.reshape((self.num_class, self.feature_dim), order="F")

        def classify_row(x_row):
            scores_row = Wg_mat @ x_row + Cg_vec
            return int(np.argmax(scores_row)) + 1   # 1..num_class

        trial_results = []

        # Per-trial accuracy
        for trial_key, (start_row, end_row) in self.trial_feature_ranges.items():
            s = int(start_row)
            e = int(end_row)
            if s < 0 or e > num_samples or s >= e:
                self.add_report(f"WARNING: invalid range for trial {trial_key}: ({s}, {e})")
                continue

            trial_feat = feat[s:e, :]
            trial_labels = labels[s:e]

            preds = [classify_row(trial_feat[i, :]) for i in range(trial_feat.shape[0])]
            preds = np.array(preds, dtype=np.int32)

            total = trial_labels.size
            correct = int((preds == trial_labels).sum())
            acc = correct / total if total > 0 else 0.0

            trial_results.append((trial_key, acc))

        # Sort by accuracy ascending
        trial_results.sort(key=lambda x: x[1])

        # Pick worst 'top_k' per label (act1, act2, ...)
        worst_by_label = {}
        for trial_key, acc in trial_results:
            label_name = trial_key.split("/")[0]  # 'act1', 'act2', ...
            bucket = worst_by_label.setdefault(label_name, [])
            if len(bucket) < top_k:
                bucket.append(trial_key)

        return worst_by_label, trial_results
    
    

    def multipletrain_prune(
        self,
        root_folder,
        prune_trials=1,
        split_type="alternate",
        percentage=60,
        accuracy_report_name=None,
        make_plot=True,
        max_sample_size=1100,
    ):
        """
        Compare original vs pruned models across all clip sizes <= max_sample_size.

        Steps for each ResultClipSizeUpXXXX set:

        1) On the ORIGINAL set:
            - split Train/Test, train LDA, test LDA
            - record baseline train/test accuracy
            - compute per-trial accuracy on TRAIN windows
            - choose the 'prune_trials' worst trials per label (act1, act2, ...)

        2) Create a fresh copy of the original set under:
                root_folder / f"prune_<prune_trials>" / ResultClipSizeUpXXXX
            and REMOVE those worst trials (actX/trial_YY.txt) from the copy only.

        3) On the PRUNED copy:
            - split Train/Test again, train LDA, test LDA
            - record pruned train/test accuracy

        4) Save a text report + a plot of Test(original) vs Test(pruned)
            versus sample size (only sizes <= max_sample_size).

        Notes
        -----
        * The ORIGINAL datasets under root_folder are never modified
        (only Train/Test folders are recreated, as usual).
        * Each prune_trials value gets its own subfolder:
            root_folder / f"prune_<prune_trials>"
        """
        root = Path(root_folder)
        if not root.exists():
            raise FileNotFoundError(f"{root} does not exist")

        # ---------- helper to read size from "ResultClipSizeUpXXXX" ----------
        def size_key(p: Path) -> int:
            digits = "".join(ch for ch in p.name if ch.isdigit())
            return int(digits) if digits else 0

        def trialnum_from_path(p) -> int | None:
            """
            Extract trial number from strings like:
            'act1/trial_03.txt' or 'act2\\trial_18.txt'
            Returns int trial number, or None if not found.
            """
            s = str(p).replace("\\", "/")
            m = re.search(r"trial_(\d+)\.txt$", s)
            return int(m.group(1)) if m else None

        def object_id_from_trial(trial_num: int, group_size: int = 10) -> int:
            """
            Object mapping:
            1..10   -> object 1
            11..20  -> object 2
            ...
            """
            return ((trial_num - 1) // group_size) + 1

        def summarize_object_prune(total_by_obj: dict, removed_relpaths, group_size: int = 10):
            """
            total_by_obj: dict like {object_id: total_trials_before_deletion}
            removed_relpaths: list like ["act1/trial_03.txt", ...]
            """
            removed_by_obj = defaultdict(list)

            for rel in removed_relpaths:
                tn = trialnum_from_path(rel)
                if tn is None:
                    continue
                obj = object_id_from_trial(tn, group_size=group_size)
                removed_by_obj[obj].append(tn)

            summary = {}
            all_objs = set(total_by_obj.keys()) | set(removed_by_obj.keys())

            for obj in sorted(all_objs):
                total = int(total_by_obj.get(obj, 0))
                removed_list = sorted(removed_by_obj.get(obj, []))
                removed = len(removed_list)
                pct = (removed / total * 100.0) if total > 0 else 0.0

                summary[obj] = {
                    "total_trials": total,
                    "removed_trials": removed,
                    "percent_removed": pct,
                    "removed_list": removed_list,
                }

            return summary

        # All ResultClipSizeUp* sets, but keep only <= max_sample_size
        dataset_dirs = [
            d for d in root.iterdir()
            if d.is_dir()
            and d.name.lower().startswith("resultclipsizeup")
            and size_key(d) <= max_sample_size
        ]
        if not dataset_dirs:
            raise FileNotFoundError(
                f"No 'ResultClipSizeUp*' folders with size <= {max_sample_size} found in {root}"
            )

        dataset_dirs.sort(key=size_key)

        # ---------- outputs we will fill ----------
        sample_sizes = []
        base_train_accs = []   # original train
        base_test_accs = []    # original test
        pruned_train_accs = [] # pruned train
        pruned_test_accs = []  # pruned test

        # ---------- set up prune root + report path ----------
        prune_root = root / f"prune_{prune_trials}"
        prune_root.mkdir(exist_ok=True)

        if accuracy_report_name is None:
            accuracy_report_name = f"AccuracyReport_Train{percentage}_Prune{prune_trials}.txt"

        acc_report_path = prune_root / accuracy_report_name
        if acc_report_path.exists():
            acc_report_path.unlink()

        def wr(line=""):
            with acc_report_path.open("a", encoding="utf-8") as f:
                f.write(str(line) + "\n")

        wr("============================================================")
        wr(f"PRUNE REPORT  (Train {percentage}% / Test {100-percentage}%)")
        wr(f"Prune trials per label   : {prune_trials}")
        wr(f"Root folder (original)   : {root}")
        wr(f"Prune root (output sets) : {prune_root}")
        wr(f"Only sample sizes <= {max_sample_size} are used.")
        wr("============================================================")
        wr("SampleSize\tBaseTrain(%)\tBaseTest(%)\tPrunedTrain(%)\tPrunedTest(%)")
        wr("")

        # ==========================================================
        # LOOP OVER EACH SAMPLE-SIZE DATASET (<= max_sample_size)
        # ==========================================================
        for ds in dataset_dirs:
            size = size_key(ds)
            sample_sizes.append(size)

            # ------------------------------------------------------
            # 1) BASELINE on ORIGINAL dataset "ds"
            # ------------------------------------------------------
            self.add_report("")
            self.add_report("################################################")
            self.add_report(f"### BASELINE DATASET: {ds.name}  ({ds})")
            self.add_report("################################################")

            self.data_set_location = ds
            self.split_training_testing(type=split_type, percentage=percentage)
            self.set_data_info()
            self.train_model()
            base_test = self.test_model()

            base_test_pct = base_test * 100.0
            base_test_accs.append(base_test_pct)

            if hasattr(self, "last_train_accuracy") and self.last_train_accuracy is not None:
                base_train_pct = float(self.last_train_accuracy)
            else:
                base_train_pct = float("nan")
            base_train_accs.append(base_train_pct)

            self.add_report(
                f"[PRUNE] BASELINE size={size}: Train={base_train_pct:.2f}%, Test={base_test_pct:.2f}%"
            )

            # ============================
            # SAVE BASELINE WEIGHTS
            # ============================
            base_c_out = ds / f"LDA_BASE_size{size}_Train{percentage}_{split_type}.c"
            base_meta = {
                "mode": "multipletrain_prune:baseline",
                "dataset": str(ds),
                "sample_size": size,
                "split_type": split_type,
                "train_percent": percentage,
                "test_percent": 100 - percentage,
                "train_accuracy": f"{base_train_pct:.2f}%",
                "test_accuracy": f"{base_test_pct:.2f}%",
                "prune_trials_per_label": prune_trials,
            }
            try:
                self.save_lda_weights_cfile(out_path=str(base_c_out), meta=base_meta)
            except TypeError:
                self.save_lda_weights_cfile(out_path=str(base_c_out))

            # ------------------------------------------------------
            # 2) Find worst 'prune_trials' training trials per label
            # ------------------------------------------------------
            worst_by_label, all_trial_results = self._compute_worst_trials_per_label(top_k=prune_trials)

            self.add_report("[PRUNE] Per-trial accuracies (ascending):")
            for trial_key, acc in all_trial_results:
                self.add_report(f"  {trial_key}: {acc*100:.2f}%")

            self.add_report("")
            self.add_report(
                f"[PRUNE] Marking up to {prune_trials} worst trials per label for removal from the COPY."
            )
            for label_name, trial_list in worst_by_label.items():
                self.add_report(f"  Label {label_name}: " + ", ".join(trial_list))

            # ------------------------------------------------------
            # 3) Copy dataset -> adjusted_ds
            # ------------------------------------------------------
            adjusted_ds = prune_root / ds.name
            if adjusted_ds.exists():
                shutil.rmtree(adjusted_ds)
            shutil.copytree(ds, adjusted_ds)

            # --- snapshot totals BEFORE deletion (for correct percentages) ---
            predelete_total_by_label_obj = {}
            for label_name in worst_by_label.keys():
                label_train_folder = adjusted_ds / "Train" / label_name
                total_by_obj = defaultdict(int)

                if label_train_folder.exists():
                    for f in label_train_folder.rglob("trial_*.txt"):
                        tn = trialnum_from_path(f)
                        if tn is None:
                            continue
                        obj = object_id_from_trial(tn, group_size=10)
                        total_by_obj[obj] += 1

                predelete_total_by_label_obj[label_name] = dict(total_by_obj)

            # --- delete worst trials from copy only ---
            removed_by_label = {k: [] for k in worst_by_label.keys()}

            for label_name, trial_list in worst_by_label.items():
                for trial_key in trial_list:
                    rel = Path(trial_key)  # e.g. "act1/trial_03.txt"
                    raw_file = adjusted_ds / "Train" / rel

                    if raw_file.exists():
                        raw_file.unlink()
                        removed_by_label[label_name].append(str(rel))
                        self.add_report(f"[PRUNE] Removed from copy: {raw_file}")
                    else:
                        self.add_report(f"[PRUNE] WARNING: expected file not found in copy: {raw_file}")

            # ------------------------------------------------------
            # Object-level cut summary
            # ------------------------------------------------------
            self.add_report("")
            self.add_report("[PRUNE] Object-level cut summary (1-10=obj1, 11-20=obj2, ...):")
            wr("# Object-level cut summary (1-10=obj1, 11-20=obj2, ...):")

            for label_name, removed_relpaths in removed_by_label.items():
                label_train_folder = adjusted_ds / "Train" / label_name
                if not label_train_folder.exists():
                    msg = f"#   {label_name}: (Train folder not found: {label_train_folder})"
                    self.add_report(msg)
                    wr(msg)
                    continue

                obj_summary = summarize_object_prune(
                    total_by_obj=predelete_total_by_label_obj.get(label_name, {}),
                    removed_relpaths=removed_relpaths,
                    group_size=10,
                )

                self.add_report(f"  Label {label_name}:")
                wr(f"#   Label {label_name}:")

                for obj_id, s in obj_summary.items():
                    if s["total_trials"] == 0:
                        continue

                    line = (
                        f"    Object {obj_id}: removed {s['removed_trials']}/{s['total_trials']} "
                        f"= {s['percent_removed']:.1f}% (trials removed: {s['removed_list']})"
                    )
                    self.add_report(line)
                    wr("# " + line)

            # ------------------------------------------------------
            # 4) Train & test on PRUNED COPY (reuse copied Train/Test)
            # ------------------------------------------------------
            self.add_report("")
            self.add_report("################################################")
            self.add_report(f"###  PRUNED DATASET (copy): {adjusted_ds.name}  ({adjusted_ds})")
            self.add_report("################################################")

            self.data_set_location = adjusted_ds
            self.train_folder = adjusted_ds / "Train"
            self.testing_folder = adjusted_ds / "Test"

            self.set_data_info()
            self.train_model()
            pruned_test = self.test_model()

            pruned_test_pct = pruned_test * 100.0
            pruned_test_accs.append(pruned_test_pct)

            if hasattr(self, "last_train_accuracy") and self.last_train_accuracy is not None:
                pruned_train_pct = float(self.last_train_accuracy)
            else:
                pruned_train_pct = float("nan")
            pruned_train_accs.append(pruned_train_pct)

            self.add_report(
                f"[PRUNE] PRUNED size={size}: Train={pruned_train_pct:.2f}%, Test={pruned_test_pct:.2f}%"
            )

            # ============================
            # SAVE PRUNED WEIGHTS
            # ============================
            pruned_c_out = adjusted_ds / (
                f"LDA_PRUNED_size{size}_Train{percentage}_{split_type}_Prune{prune_trials}.c"
            )
            pruned_meta = {
                "mode": "multipletrain_prune:pruned",
                "dataset": str(adjusted_ds),
                "source_dataset": str(ds),
                "sample_size": size,
                "split_type": split_type,
                "train_percent": percentage,
                "test_percent": 100 - percentage,
                "train_accuracy": f"{pruned_train_pct:.2f}%",
                "test_accuracy": f"{pruned_test_pct:.2f}%",
                "prune_trials_per_label": prune_trials,
                "removed_trials": removed_by_label,
            }
            try:
                self.save_lda_weights_cfile(out_path=str(pruned_c_out), meta=pruned_meta)
            except TypeError:
                self.save_lda_weights_cfile(out_path=str(pruned_c_out))

            # ------------------------------------------------------
            # 5) Write summary row for this sample size into report file
            # ------------------------------------------------------
            wr(
                f"{size}\t"
                f"{base_train_pct:.2f}\t{base_test_pct:.2f}\t"
                f"{pruned_train_pct:.2f}\t{pruned_test_pct:.2f}"
            )
            wr(f"# Dataset: {ds.name}")
            wr("# Removed trials from copy:")
            for label_name, rel_list in removed_by_label.items():
                if rel_list:
                    wr(f"#   {label_name}: " + ", ".join(rel_list))
                else:
                    wr(f"#   {label_name}: (none removed)")
            wr("")

        # ==========================================================
        # 6) Plot Train/Test (original vs pruned) vs sample size
        # ==========================================================
        if make_plot and sample_sizes:
            import numpy as np
            import matplotlib.pyplot as plt

            sample_sizes_arr = np.array(sample_sizes, dtype=np.int32)
            base_train_arr = np.array(base_train_accs, dtype=np.float32)
            base_test_arr = np.array(base_test_accs, dtype=np.float32)
            pruned_train_arr = np.array(pruned_train_accs, dtype=np.float32)
            pruned_test_arr = np.array(pruned_test_accs, dtype=np.float32)

            plt.figure(figsize=(7, 4))

            # Original model
            plt.plot(sample_sizes_arr, base_train_arr, marker="o", linestyle="--", label="Train (original)")
            plt.plot(sample_sizes_arr, base_test_arr, marker="o", linestyle="--", label="Test (original)")

            # Pruned model
            plt.plot(sample_sizes_arr, pruned_train_arr, marker="s", linestyle="-", label="Train (pruned)")
            plt.plot(sample_sizes_arr, pruned_test_arr, marker="s", linestyle="-", label="Test (pruned)")

            plt.xlabel("Sample Size")
            plt.ylabel("Accuracy (%)")
            plt.title(
                f"Accuracy vs. Sample Size (Train {percentage}% / Test {100 - percentage}%, prune={prune_trials})"
            )
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend()
            plt.tight_layout()

            plot_path = prune_root / f"Accuracy_vs_SampleSize_Train{percentage}_Prune{prune_trials}.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()

            wr(f"# Saved accuracy plot: {plot_path}")

        wr("")
        wr("=== End of prune run ===")


if __name__ == "__main__":
    train_obj = trainclass()
    root_tabledata = r"C:\Users\mtino\OneDrive\Desktop\ResearchCode\Application5\set1final"

    for prune in range(1, 15):  # 1..14
        print(f"\n\n=== Running prune_trials={prune} ===\n")
        train_obj.multipletrain_prune(
            root_folder=root_tabledata,
            prune_trials=prune,
            split_type="alternate",
            percentage=80,
            accuracy_report_name=f"AccuracyReport_Train90_Pruned{prune}.txt",
            make_plot=True,
            max_sample_size=1100
        )
