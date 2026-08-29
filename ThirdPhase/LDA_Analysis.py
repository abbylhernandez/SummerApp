"""Analysis and reporting methods for the EMG LDA training pipeline.

The methods live in a mixin so the core training class can remain focused on
data preparation, model fitting, and evaluation.
"""


class AnalysisMethods:
    def run_all_analyses(self):
        """Run every training, testing, and train-vs-test analysis in order.

        Call this after ``train_model()`` and after ``testing_folder`` has been
        set. A mapping of method names to their return values is returned.
        Most methods communicate their results through files in
        ``Report analysis/`` and therefore return ``None``.
        """
        methods = (
            self.train_analysis,
            self.train_analysis2,
            self.train_analysis3,
            self.train_analysis4,
            self.test_analysis,
            self.test_analysis2,
            self.test_analysis3,
            self.train_test_trial_distance_analysis,
            self.compare_train_test_features,
            self.train_test_confusion,
            self.confusion_matrix_train_vs_test_trials,
        )

        results = {}
        for method in methods:
            results[method.__name__] = method()
        return results

    def train_analysis(self):
        """
        Analyze training data and save results into:
            ./Report analysis/

        Includes:
        1) Per-feature statistics & histograms
        2) Per-trial feature heatmaps
        3) LDA projection histograms
        4) Per-trial accuracy using trained LDA
        5) Feature correlation matrix
        6) Key discriminative features summary

        Call AFTER train_model(), so that:
        - self.featurematrix
        - self.trainclass
        - self.Wg, self.Cg
        - self.xmean, self.xstd
        - self.trial_feature_ranges
        are all populated.

        NOTE: self.trainclass is expected to use 1-based labels: 1..num_class
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt

        if self.train_folder is None:
            raise ValueError("train_folder is not set. Run split_training_testing() and train_model() first.")

        if not hasattr(self, "featurematrix") or len(self.featurematrix) == 0:
            raise ValueError("featurematrix is empty. Run train_model() before train_analysis().")

        # -------------------------------------------------------
        # 0) Setup: create analysis folder + report file
        # -------------------------------------------------------
        report_dir = os.getcwd()  # current working directory
        analysis_dir = os.path.join(report_dir, "Report analysis")
        os.makedirs(analysis_dir, exist_ok=True)

        report_path = os.path.join(analysis_dir, "TrainAnalysis_Report.txt")

        # Clear existing report (start fresh)
        if os.path.exists(report_path):
            os.remove(report_path)

        def wr(line=""):
            """Helper to write a line into the analysis report."""
            with open(report_path, "a", encoding="utf-8") as f:
                f.write(str(line) + "\n")

        wr("********************************************************")
        wr("************  TRAINING DATA ANALYSIS  *****************")
        wr("********************************************************")
        wr(f"Train folder: {self.train_folder}")
        wr(f"Analysis folder: {analysis_dir}")
        wr("")

        # -------------------------------------------------------
        # 1) Convert featurematrix and labels to NumPy
        # -------------------------------------------------------
        feat = np.array(self.featurematrix, dtype=np.float32)
        if feat.size % self.feature_dim != 0:
            raise ValueError(
                f"featurematrix length {feat.size} is not divisible by feature_dim {self.feature_dim}"
            )
        num_samples = feat.size // self.feature_dim
        feat = feat.reshape((num_samples, self.feature_dim))

        labels = np.array(self.trainclass, dtype=np.int32)  # labels are 1..num_class
        if labels.size != num_samples:
            raise ValueError(
                f"trainclass length {labels.size} does not match num_samples {num_samples}"
            )

        unique_labels = np.unique(labels)

        wr(f"Total samples (windows): {num_samples}")
        wr(f"Number of features per sample: {self.feature_dim}")
        wr(f"Number of classes (configured): {self.num_class}")
        wr(f"Unique labels present in training: {unique_labels.tolist()}")
        wr("")

        # Try to use human-readable feature names if available
        tdfeature_names = getattr(self, "tdfeature_names", None)
        if tdfeature_names is None or len(tdfeature_names) < self.tdfeatureN:
            # Fallback generic naming if not defined
            tdfeature_names = [f"F{i+1}" for i in range(self.tdfeatureN)]

        # -------------------------------------------------------
        # 2) Per-feature statistics & histograms
        # -------------------------------------------------------
        wr("=== Per-feature Statistics (Global + Per Class) ===")

        # We'll also compute a discriminative score for each feature
        # based on between-class mean spread / global std
        discriminative_scores = np.zeros(self.feature_dim, dtype=np.float32)
        class_means_per_feature = np.zeros((self.num_class, self.feature_dim), dtype=np.float32)

        for j in range(self.feature_dim):
            col = feat[:, j]
            global_mean = float(col.mean())
            global_std = float(col.std(ddof=1)) if col.size > 1 else 0.0

            wr(f"Feature {j}: global_mean={global_mean:.4f}, global_std={global_std:.4f}")

            class_means = []
            for lbl in unique_labels:
                c_idx = int(lbl) - 1   # map label 1..C -> index 0..C-1
                c_mask = (labels == lbl)
                if np.any(c_mask):
                    cm = float(col[c_mask].mean())
                    class_means.append(cm)
                    if 0 <= c_idx < self.num_class:
                        class_means_per_feature[c_idx, j] = cm
                    wr(f"  Class {lbl}: mean={cm:.4f}, n={c_mask.sum()}")
                else:
                    wr(f"  Class {lbl}: no samples")
                    class_means.append(global_mean)

            # Discriminative score: spread of class means / global std
            spread = max(class_means) - min(class_means)
            if global_std > 1e-6:
                discriminative_scores[j] = spread / global_std
            else:
                discriminative_scores[j] = 0.0

            # Save histogram for this feature (per-class overlaid)
            plt.figure()
            for lbl in unique_labels:
                mask = (labels == lbl)
                if np.any(mask):
                    plt.hist(
                        col[mask],
                        bins=40,
                        alpha=0.5,
                        label=f"Class {lbl}",
                        density=True,
                    )
            plt.title(f"Feature {j} distribution by class")
            plt.xlabel(f"Feature {j} value")
            plt.ylabel("Density")
            plt.legend()
            fig_path = os.path.join(analysis_dir, f"Feature_{j:02d}_hist.png")
            plt.tight_layout()
            plt.savefig(fig_path)
            plt.close()

            wr(f"  Saved histogram: {fig_path}")
            wr("")

        wr("")
        wr("=== End of Per-feature Statistics ===")
        wr("")

        # -------------------------------------------------------
        # 3) Per-trial feature heatmaps (using trial_feature_ranges)
        # -------------------------------------------------------
        wr("=== Per-trial Feature Heatmaps ===")

        if not hasattr(self, "trial_feature_ranges") or not self.trial_feature_ranges:
            wr("No trial_feature_ranges found. Skipping heatmaps.")
        else:
            for trial_key, (start_row, end_row) in self.trial_feature_ranges.items():
                # end_row is assumed to be exclusive (as set in train_model)
                s = int(start_row)
                e = int(end_row)
                if s < 0 or e > num_samples or s >= e:
                    wr(f"WARNING: invalid range for trial {trial_key}: ({s}, {e})")
                    continue

                trial_feat = feat[s:e, :]  # shape: (num_windows, feature_dim)

                # sanitize filename
                safe_key = trial_key.replace("/", "_").replace("\\", "_")
                fig_path = os.path.join(analysis_dir, f"Heatmap_{safe_key}.png")

                plt.figure(figsize=(10, 4))
                plt.imshow(trial_feat, aspect="auto", interpolation="nearest")
                plt.colorbar(label="Feature value")
                plt.title(f"Feature Heatmap: {trial_key}")
                plt.xlabel("Feature index")
                plt.ylabel("Window index")
                plt.tight_layout()
                plt.savefig(fig_path)
                plt.close()

                wr(f"Trial: {trial_key}")
                wr(f"  Windows: {e - s}, feature rows: {s}..{e - 1}")
                wr(f"  Saved heatmap: {fig_path}")
                wr("")

        wr("")
        wr("=== End of Heatmaps ===")
        wr("")

        # -------------------------------------------------------
        # 4) LDA projection histograms
        #     Use trained Wg, Cg to see class separation
        # -------------------------------------------------------
        wr("=== LDA Projection Histograms ===")

        if not hasattr(self, "Wg") or not hasattr(self, "Cg"):
            wr("Wg/Cg not found. Skipping LDA projection analysis.")
        else:
            Wg_vec = np.array(self.Wg, dtype=np.float32)
            Cg_vec = np.array(self.Cg, dtype=np.float32)

            if Wg_vec.size != self.num_class * self.feature_dim:
                wr(f"WARNING: Wg size {Wg_vec.size} does not match num_class*feature_dim.")
            else:
                # Recall: C layout: Wg[i + j * CLASS]
                # This corresponds to Fortran ordering (CLASS x FEATURE_DIM)
                Wg_mat = Wg_vec.reshape((self.num_class, self.feature_dim), order="F")

                # Features in feat are already normalized (after feature_normalization in train_model),
                # so we use them directly.
                scores = feat @ Wg_mat.T + Cg_vec  # shape: (num_samples, num_class)

                # For each discriminant dimension (0..num_class-1),
                # plot score distribution per true label.
                for d in range(self.num_class):
                    plt.figure()
                    for lbl in unique_labels:
                        mask = (labels == lbl)
                        if np.any(mask):
                            plt.hist(
                                scores[mask, d],
                                bins=40,
                                alpha=0.5,
                                density=True,
                                label=f"True class {lbl}",
                            )
                    plt.title(f"LDA Score Distribution: Discriminant {d}")
                    plt.xlabel("LDA score")
                    plt.ylabel("Density")
                    plt.legend()
                    fig_path = os.path.join(analysis_dir, f"LDA_Discriminant_{d}.png")
                    plt.tight_layout()
                    plt.savefig(fig_path)
                    plt.close()

                    wr(f"LDA discriminant {d} score histogram saved: {fig_path}")

                wr("")
                wr("LDA projection analysis complete.")
        wr("")
        wr("=== End of LDA Projection Analysis ===")
        wr("")

        # -------------------------------------------------------
        # 5) Per-trial accuracy using final LDA model
        #     (Train-on-all, test-on-each-trial to find weak trials)
        # -------------------------------------------------------
        wr("=== Per-trial Accuracy (using trained LDA) ===")

        if not hasattr(self, "Wg") or not hasattr(self, "Cg"):
            wr("Wg/Cg not found. Skipping per-trial accuracy.")
        elif not hasattr(self, "trial_feature_ranges") or not self.trial_feature_ranges:
            wr("trial_feature_ranges not found. Skipping per-trial accuracy.")
        else:
            Wg_vec = np.array(self.Wg, dtype=np.float32)
            Cg_vec = np.array(self.Cg, dtype=np.float32)
            Wg_mat = Wg_vec.reshape((self.num_class, self.feature_dim), order="F")

            def classify_row(x_row):
                # x_row: shape (feature_dim,)
                # Return 1-based predicted class label to match self.trainclass
                scores_row = Wg_mat @ x_row + Cg_vec
                return int(np.argmax(scores_row)) + 1  # 1..num_class

            trial_results = []

            for trial_key, (start_row, end_row) in self.trial_feature_ranges.items():
                s = int(start_row)
                e = int(end_row)
                if s < 0 or e > num_samples or s >= e:
                    wr(f"WARNING: invalid range for trial {trial_key}: ({s}, {e})")
                    continue

                trial_feat = feat[s:e, :]
                trial_labels = labels[s:e]

                preds = []
                for i in range(trial_feat.shape[0]):
                    preds.append(classify_row(trial_feat[i, :]))
                preds = np.array(preds, dtype=np.int32)

                correct = (preds == trial_labels).sum()
                total = trial_labels.size
                acc = correct / total if total > 0 else 0.0

                trial_results.append((trial_key, acc, correct, total))

            # Sort trials by accuracy (ascending) to highlight worst ones
            trial_results.sort(key=lambda x: x[1])

            wr("Per-trial accuracy (sorted by accuracy ascending):")
            for trial_key, acc, correct, total in trial_results:
                wr(f"  {trial_key}: accuracy={acc*100:.2f}%  ({correct}/{total})")

            wr("")
            if trial_results:
                worst = trial_results[0]
                wr(f"Worst trial: {worst[0]}  accuracy={worst[1]*100:.2f}%  "
                f"({worst[2]}/{worst[3]})")
            wr("")
        wr("=== End of Per-trial Accuracy ===")
        wr("")

        # -------------------------------------------------------
        # 6) Feature correlation matrix
        # -------------------------------------------------------
        wr("=== Feature Correlation Matrix ===")

        if num_samples <= 1:
            wr("Not enough samples for correlation.")
        else:
            corr = np.corrcoef(feat, rowvar=False)  # shape: (feature_dim, feature_dim)

            fig_path = os.path.join(analysis_dir, "Feature_Correlation_Matrix.png")
            plt.figure(figsize=(8, 6))
            im = plt.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm", interpolation="nearest")
            plt.colorbar(im, label="Correlation")
            plt.title("Feature Correlation Matrix")
            plt.xlabel("Feature index")
            plt.ylabel("Feature index")
            plt.tight_layout()
            plt.savefig(fig_path)
            plt.close()

            wr(f"Feature correlation matrix saved: {fig_path}")
        wr("")
        wr("=== End of Feature Correlation Analysis ===")
        wr("")

        # -------------------------------------------------------
        # 7) Key discriminative features summary
        # -------------------------------------------------------
        wr("=== Key Discriminative Features (Ranking) ===")
        wr("Score definition: (max class mean - min class mean) / global_std")
        wr("Higher score = better separation between classes.\n")

        # sort feature indices by score descending
        idx_sorted = np.argsort(-discriminative_scores)
        top_k = min(20, self.feature_dim)  # top 20 or fewer
        wr(f"Top {top_k} features by discriminative score:\n")

        for rank in range(top_k):
            j = int(idx_sorted[rank])
            score = float(discriminative_scores[j])

            # map j -> channel and TD feature index
            ch_idx = j // self.tdfeatureN
            feat_idx = j % self.tdfeatureN

            ch_name = f"Ch{ch_idx + 1}"
            feat_name = tdfeature_names[feat_idx] if feat_idx < len(tdfeature_names) else f"F{feat_idx+1}"

            means_line = " | ".join(
                [
                    f"class {lbl}: mean={class_means_per_feature[int(lbl)-1, j]:.4f}"
                    for lbl in unique_labels
                ]
            )

            wr(
                f"Rank {rank+1:02d}: Feature {j} "
                f"({ch_name}-{feat_name}), score={score:.4f}\n"
                f"    {means_line}"
            )

        wr("")
        wr("=== End of Key Feature Ranking ===")
        wr("")
        wr("************  END OF TRAINING ANALYSIS  ***************")
        wr("********************************************************")

    def train_analysis2(self):
        """
        Extra training analysis with confusion matrices and per-trial prediction
        distributions. Results are saved into:

            ./Report analysis/TrainAnalysis2_Report.txt
            ./Report analysis/ConfusionMatrix_*.png

        Requirements (call AFTER train_model()):
        - self.featurematrix  : normalized training features
        - self.trainclass     : 1-based labels (1..num_class)
        - self.Wg, self.Cg    : trained LDA parameters
        - self.trial_feature_ranges : mapping trial_key -> (start_row, end_row)
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt

        # ---------------------------------------------------
        # 0) Basic checks
        # ---------------------------------------------------
        if self.train_folder is None:
            raise ValueError(
                "train_folder is not set. Run split_training_testing() and train_model() first."
            )

        if not hasattr(self, "featurematrix") or len(self.featurematrix) == 0:
            raise ValueError(
                "featurematrix is empty. Run train_model() before train_analysis2()."
            )

        if not hasattr(self, "Wg") or not hasattr(self, "Cg"):
            raise ValueError(
                "Wg/Cg not found. Run train_model() before train_analysis2()."
            )

        # ---------------------------------------------------
        # 1) Setup: analysis folder + report file
        # ---------------------------------------------------
        cwd = os.getcwd()
        analysis_dir = os.path.join(cwd, "Report analysis")
        os.makedirs(analysis_dir, exist_ok=True)

        report_path = os.path.join(analysis_dir, "TrainAnalysis2_Report.txt")

        # Start fresh each time
        if os.path.exists(report_path):
            os.remove(report_path)

        def wr(line=""):
            with open(report_path, "a", encoding="utf-8") as f:
                f.write(str(line) + "\n")

        wr("********************************************************")
        wr("************  TRAINING DATA ANALYSIS 2  ****************")
        wr("********************************************************")
        wr(f"Train folder: {self.train_folder}")
        wr(f"Analysis folder: {analysis_dir}")
        wr("")

        # ---------------------------------------------------
        # 2) Convert buffers to NumPy and build valid mask
        # ---------------------------------------------------
        feat = np.array(self.featurematrix, dtype=np.float32)
        if feat.size % self.feature_dim != 0:
            raise ValueError(
                f"featurematrix length {feat.size} is not divisible by feature_dim {self.feature_dim}"
            )
        num_samples = feat.size // self.feature_dim
        feat = feat.reshape((num_samples, self.feature_dim))

        labels = np.array(self.trainclass, dtype=np.int32)

        # Use only rows with valid labels 1..num_class
        valid_mask = (labels >= 1) & (labels <= self.num_class)
        feat_valid = feat[valid_mask]
        labels_valid = labels[valid_mask]
        num_valid = feat_valid.shape[0]

        wr(f"Total rows in featurematrix: {num_samples}")
        wr(f"Valid labeled rows (1..{self.num_class}): {num_valid}")
        wr("")

        if num_valid == 0:
            wr("No valid labeled rows found. Aborting confusion analysis.")
            wr("************  END OF TRAINING ANALYSIS 2  *************")
            return

        unique_labels = np.unique(labels_valid)
        wr(f"Unique labels in valid set: {unique_labels.tolist()}")
        wr("")

        # ---------------------------------------------------
        # 3) Build LDA weight matrix and classify all rows
        # ---------------------------------------------------
        Wg_vec = np.array(self.Wg, dtype=np.float32)
        Cg_vec = np.array(self.Cg, dtype=np.float32)

        if Wg_vec.size != self.num_class * self.feature_dim:
            wr(
                f"WARNING: Wg size {Wg_vec.size} != num_class * feature_dim "
                f"({self.num_class * self.feature_dim}). Cannot continue."
            )
            wr("************  END OF TRAINING ANALYSIS 2  *************")
            return

        # C layout: Wg[i + j*CLASS], reshape Fortran-style
        Wg_mat = Wg_vec.reshape((self.num_class, self.feature_dim), order="F")

        # feat_valid is already normalized (train_model called feature_normalization)
        scores_all = feat_valid @ Wg_mat.T + Cg_vec  # (N_valid, num_class)
        preds_all = np.argmax(scores_all, axis=1) + 1  # 1..num_class

        # ---------------------------------------------------
        # 4) Overall confusion matrix (counts + normalized)
        # ---------------------------------------------------
        wr("=== Overall Confusion Matrix (Training Set, window-level) ===")

        C = np.zeros((self.num_class, self.num_class), dtype=np.int32)
        for t, p in zip(labels_valid, preds_all):
            C[t - 1, p - 1] += 1

        total_windows = C.sum()
        wr(f"Total valid training windows: {total_windows}")
        wr("Confusion matrix (rows = true, cols = predicted):")
        for r in range(self.num_class):
            wr(f"True class {r+1}: " + "  ".join(f"{C[r, c]:5d}" for c in range(self.num_class)))
        wr("")

        # Normalized by true-class row (so each row sums to 1)
        C_norm = C.astype(np.float32)
        row_sums = C_norm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        C_norm /= row_sums

        wr("Row-normalized confusion matrix (fraction per true class):")
        for r in range(self.num_class):
            wr(
                f"True class {r+1}: "
                + "  ".join(f"{C_norm[r, c]:0.3f}" for c in range(self.num_class))
            )
        wr("")

        # ---- Save confusion matrix images ----
        # Counts
        fig_path_counts = os.path.join(analysis_dir, "ConfusionMatrix_Train_Counts.png")
        plt.figure(figsize=(5, 4))
        im = plt.imshow(C, interpolation="nearest")
        plt.title("Training Confusion Matrix (Counts)")
        plt.xlabel("Predicted class")
        plt.ylabel("True class")
        plt.colorbar(im, label="Count")

        # put counts as text
        for i in range(self.num_class):
            for j in range(self.num_class):
                plt.text(
                    j, i, str(C[i, j]),
                    ha="center", va="center"
                )

        plt.tight_layout()
        plt.savefig(fig_path_counts)
        plt.close()
        wr(f"Saved confusion matrix (counts): {fig_path_counts}")

        # Normalized
        fig_path_norm = os.path.join(analysis_dir, "ConfusionMatrix_Train_Normalized.png")
        plt.figure(figsize=(5, 4))
        im = plt.imshow(C_norm, vmin=0.0, vmax=1.0, interpolation="nearest")
        plt.title("Training Confusion Matrix (Row-normalized)")
        plt.xlabel("Predicted class")
        plt.ylabel("True class")
        plt.colorbar(im, label="Fraction")

        for i in range(self.num_class):
            for j in range(self.num_class):
                plt.text(
                    j, i, f"{C_norm[i, j]:.2f}",
                    ha="center", va="center"
                )

        plt.tight_layout()
        plt.savefig(fig_path_norm)
        plt.close()
        wr(f"Saved confusion matrix (row-normalized): {fig_path_norm}")
        wr("")

        # ---------------------------------------------------
        # 5) Per-class precision / recall / F1
        # ---------------------------------------------------
        wr("=== Per-class Metrics (from confusion matrix) ===")
        wr("C[true, pred]   TP = C[c,c],")
        wr("FN = sum_row(c) - TP,  FP = sum_col(c) - TP,  TN = total - TP - FN - FP")
        wr("")

        for c in range(self.num_class):
            TP = C[c, c]
            FN = C[c, :].sum() - TP
            FP = C[:, c].sum() - TP
            TN = total_windows - TP - FN - FP

            prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            wr(f"Class {c+1}:")
            wr(f"  TP={TP}, FP={FP}, FN={FN}, TN={TN}")
            wr(f"  Precision = {prec*100:6.2f}%")
            wr(f"  Recall    = {rec*100:6.2f}%")
            wr(f"  F1 score  = {f1*100:6.2f}%")
            wr("")

        overall_acc = np.trace(C) / total_windows if total_windows > 0 else 0.0
        wr(f"Overall window-level training accuracy (from confusion matrix) = {overall_acc*100:.2f}%")
        wr("")

        # ---------------------------------------------------
        # 6) Per-trial prediction distributions
        # ---------------------------------------------------
        wr("=== Per-trial Prediction Distributions (window-level) ===")

        if not hasattr(self, "trial_feature_ranges") or not self.trial_feature_ranges:
            wr("trial_feature_ranges not found. Skipping per-trial distributions.")
        else:
            # We will use the FULL feat / labels / preds (not masked), to match ranges
            feat_full = feat      # (num_samples, feature_dim)
            labels_full = labels  # (num_samples,)

            # Helper to classify a single feature row (for safety)
            def classify_row(x_row):
                s_row = Wg_mat @ x_row + Cg_vec  # shape (num_class,)
                return int(np.argmax(s_row)) + 1

            # If you want to reuse preds_all, we need a mapping from valid indices back
            # to global indices; simpler (and consistent with trial ranges) is to
            # recompute predictions per trial.
            trial_summary = []

            for trial_key, (start_row, end_row) in self.trial_feature_ranges.items():
                s = int(start_row)
                e = int(end_row)
                if s < 0 or e > feat_full.shape[0] or s >= e:
                    wr(f"WARNING: invalid range for trial {trial_key}: ({s}, {e})")
                    continue

                X_trial = feat_full[s:e, :]
                y_true_trial = labels_full[s:e]

                # ignore unlabeled rows in this slice (if any)
                mask_trial_valid = (y_true_trial >= 1) & (y_true_trial <= self.num_class)
                if not np.any(mask_trial_valid):
                    wr(f"Trial {trial_key}: no valid labeled rows in this range. Skipping.")
                    continue

                X_trial = X_trial[mask_trial_valid]
                y_true_trial = y_true_trial[mask_trial_valid]

                preds_trial = np.array(
                    [classify_row(X_trial[i, :]) for i in range(X_trial.shape[0])],
                    dtype=np.int32
                )

                correct = (preds_trial == y_true_trial).sum()
                total = y_true_trial.size
                acc = correct / total if total > 0 else 0.0

                # True class (assume mostly single-class trial)
                # We just take the majority of y_true_trial
                vals, counts = np.unique(y_true_trial, return_counts=True)
                majority_idx = np.argmax(counts)
                majority_class = int(vals[majority_idx])

                # Prediction histogram for this trial
                pred_counts = np.zeros(self.num_class, dtype=np.int32)
                for p in preds_trial:
                    if 1 <= p <= self.num_class:
                        pred_counts[p - 1] += 1

                trial_summary.append((trial_key, majority_class, acc, correct, total, pred_counts))

                # --- Save per-trial prediction distribution plot ---
                safe_key = trial_key.replace("/", "_").replace("\\", "_")
                fig_path_trial = os.path.join(
                    analysis_dir, f"TrialPredDist_{safe_key}.png"
                )

                plt.figure(figsize=(5, 3))
                x = np.arange(1, self.num_class + 1)
                plt.bar(x, pred_counts)
                plt.xticks(x)
                plt.xlabel("Predicted class")
                plt.ylabel("Window count")
                plt.title(
                    f"{trial_key}\ntrue≈{majority_class}, acc={acc*100:.1f}% "
                    f"({correct}/{total})"
                )
                plt.tight_layout()
                plt.savefig(fig_path_trial)
                plt.close()

                wr(f"Trial: {trial_key}")
                wr(f"  Approx true class: {majority_class}")
                wr(f"  Windows (valid): {total}")
                wr(f"  Accuracy: {acc*100:.2f}%  ({correct}/{total})")
                wr(
                    "  Predicted counts per class: "
                    + ", ".join(
                        f"class {c+1}: {pred_counts[c]}" for c in range(self.num_class)
                    )
                )
                wr(f"  Saved prediction distribution plot: {fig_path_trial}")
                wr("")

            # Sort by accuracy to highlight worst trials
            if trial_summary:
                trial_summary.sort(key=lambda x: x[2])  # sort by acc ascending
                wr("Per-trial accuracy sorted ascending:")
                for trial_key, maj_class, acc, correct, total, pred_counts in trial_summary:
                    wr(
                        f"  {trial_key}: true≈{maj_class}, "
                        f"acc={acc*100:.2f}% ({correct}/{total}), "
                        "pred counts=["
                        + ", ".join(f"{pc}" for pc in pred_counts)
                        + "]"
                    )
                wr("")

                worst = trial_summary[0]
                wr(
                    f"Worst trial: {worst[0]}  "
                    f"(true≈{worst[1]}, acc={worst[2]*100:.2f}%  "
                    f"({worst[3]}/{worst[4]}), "
                    "pred counts=["
                    + ", ".join(str(pc) for pc in worst[5])
                    + "])"
                )
                wr("")

        wr("************  END OF TRAINING ANALYSIS 2  *************")
        wr("********************************************************")

    def train_analysis3(self):
        """
        Extended training analysis with:
          - LOO (leave-one-trial-out) LDA training & per-trial generalization
          - Mahalanobis distance statistics per trial
          - LDA margin (confidence) statistics per trial

        Outputs:
          ./Report analysis/TrainAnalysis3_Report.txt
          ./Report analysis/LOTO_Trial_Accuracy.png
          ./Report analysis/Mahalanobis_MeanDistance_PerTrial.png
          ./Report analysis/LDA_Margin_Mean_PerTrial.png

        Assumes train_model() has already run, so that:
          - self.featurematrix     : normalized training features (flattened)
          - self.trainclass        : labels aligned with feature rows
          - self.num_class         : number of classes
          - self.feature_dim       : features per window
          - self.trial_feature_ranges : dict {trial_key: (start_row, end_row)}
          - self.Wg, self.Cg       : (optional) trained LDA parameters, for margin
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt

        # ------------------------------
        # 0) Basic checks
        # ------------------------------
        if self.train_folder is None:
            raise ValueError("train_folder is not set. Run split_training_testing() and train_model() first.")

        if not hasattr(self, "featurematrix") or len(self.featurematrix) == 0:
            raise ValueError("featurematrix is empty. Run train_model() before train_analysis3().")

        # ------------------------------
        # 1) Setup report folder / file
        # ------------------------------
        cwd = os.getcwd()
        analysis_dir = os.path.join(cwd, "Report analysis")
        os.makedirs(analysis_dir, exist_ok=True)

        report_path = os.path.join(analysis_dir, "TrainAnalysis3_Report.txt")
        if os.path.exists(report_path):
            os.remove(report_path)

        def wr(line=""):
            with open(report_path, "a", encoding="utf-8") as f:
                f.write(str(line) + "\n")

        wr("********************************************************")
        wr("************  TRAINING DATA ANALYSIS 3  ****************")
        wr("********************************************************")
        wr(f"Train folder: {self.train_folder}")
        wr(f"Analysis folder: {analysis_dir}")
        wr("")

        # ------------------------------
        # 2) Convert buffers to NumPy
        # ------------------------------
        feat = np.array(self.featurematrix, dtype=np.float32)
        if feat.size % self.feature_dim != 0:
            raise ValueError(
                f"featurematrix length {feat.size} is not divisible by feature_dim {self.feature_dim}"
            )
        num_samples = feat.size // self.feature_dim
        feat = feat.reshape((num_samples, self.feature_dim))

        labels = np.array(self.trainclass, dtype=np.int32)
        if labels.size != num_samples:
            raise ValueError(
                f"trainclass length {labels.size} does not match num_samples {num_samples}"
            )

        # Only keep rows with valid labels (1..num_class)
        valid_mask = (labels >= 1) & (labels <= self.num_class)
        valid_idx = np.where(valid_mask)[0]
        X_full = feat[valid_mask]
        y_full = labels[valid_mask]
        N_valid = X_full.shape[0]

        wr(f"Total windows (rows) in featurematrix: {num_samples}")
        wr(f"Valid labeled windows (1..{self.num_class}): {N_valid}")
        wr(f"Feature dimension: {self.feature_dim}")
        wr("")

        if N_valid == 0:
            wr("No valid labeled windows found. Aborting train_analysis3.")
            wr("************  END OF TRAINING ANALYSIS 3  *************")
            return

        # ------------------------------
        # 3) Helper: fit simple LDA in NumPy
        # ------------------------------
        def fit_lda(X, y, num_class, reg=1e-3):
            """
            Fit a simple multi-class LDA with shared covariance.
            Returns (W, b, means, cov_inv) or (None, None, None, None) on failure.

            g_k(x) = x^T W[k,:] + b[k]
            """
            n, d = X.shape
            if n <= num_class:
                return None, None, None, None

            means = np.zeros((num_class, d), dtype=np.float64)
            priors = np.zeros(num_class, dtype=np.float64)

            for c in range(1, num_class + 1):
                idx = (y == c)
                Xc = X[idx]
                if Xc.shape[0] == 0:
                    return None, None, None, None
                means[c - 1, :] = Xc.mean(axis=0)
                priors[c - 1] = Xc.shape[0] / n

            # Shared covariance
            X_centered = np.zeros_like(X, dtype=np.float64)
            for c in range(1, num_class + 1):
                idx = (y == c)
                X_centered[idx] = X[idx] - means[c - 1]

            # Unbiased estimate with small regularization
            dof = max(n - num_class, 1)
            cov = (X_centered.T @ X_centered) / dof

            trace = np.trace(cov)
            if trace <= 0.0:
                reg_term = reg
            else:
                reg_term = reg * (trace / d)

            cov += reg_term * np.eye(d, dtype=np.float64)

            try:
                cov_inv = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                return None, None, None, None

            # Discriminant parameters
            W = (cov_inv @ means.T).T  # (num_class, d)
            b = np.zeros(num_class, dtype=np.float64)
            for k in range(num_class):
                m = means[k]
                b[k] = -0.5 * float(m @ cov_inv @ m) + np.log(priors[k] + 1e-12)

            return W.astype(np.float32), b.astype(np.float32), means.astype(np.float32), cov_inv.astype(np.float32)

        def lda_predict(X, W, b):
            scores = X @ W.T + b  # (N, num_class)
            preds = np.argmax(scores, axis=1) + 1  # 1..num_class
            return preds, scores

        # ------------------------------
        # 4) Baseline LDA on full valid set
        # ------------------------------
        W_base, b_base, means_base, covinv_base = fit_lda(X_full, y_full, self.num_class)
        if W_base is None:
            wr("Baseline LDA fit failed (singular covariance or missing class).")
            wr("************  END OF TRAINING ANALYSIS 3  *************")
            return

        preds_base, scores_base = lda_predict(X_full, W_base, b_base)
        acc_base = float((preds_base == y_full).mean())
        wr("=== Baseline LDA (NumPy implementation) ===")
        wr(f"Baseline window-level accuracy on full training set: {acc_base*100:.2f}%")
        wr("")

        # ------------------------------
        # 5) Mahalanobis distances (per window, per trial)
        # ------------------------------
        wr("=== Mahalanobis Distance Analysis ===")
        # Mahalanobis distance of each window to its own class mean
        dists = np.zeros(N_valid, dtype=np.float32)
        for i in range(N_valid):
            c = y_full[i]
            m = means_base[c - 1]
            diff = X_full[i] - m
            d2 = float(diff @ covinv_base @ diff)
            if d2 < 0:
                d2 = 0.0
            dists[i] = float(np.sqrt(d2))

        wr(f"Global Mahalanobis distance stats:")
        wr(f"  mean  = {float(dists.mean()):.4f}")
        wr(f"  std   = {float(dists.std()):.4f}")
        wr(f"  min   = {float(dists.min()):.4f}")
        wr(f"  max   = {float(dists.max()):.4f}")
        wr("")

        # ------------------------------
        # Map trial ranges into valid indices
        # ------------------------------
        if not hasattr(self, "trial_feature_ranges") or not self.trial_feature_ranges:
            wr("trial_feature_ranges not found. Skipping per-trial analyses.")
            wr("************  END OF TRAINING ANALYSIS 3  *************")
            return

        trial_keys = []
        mahal_mean_per_trial = []
        mahal_std_per_trial = []

        # We'll also store per-trial indices in the valid set for re-use
        trial_to_valid_idx = {}

        for trial_key, (start_row, end_row) in self.trial_feature_ranges.items():
            s = int(start_row)
            e = int(end_row)
            if s < 0 or e > num_samples or s >= e:
                wr(f"WARNING: invalid range for trial {trial_key}: ({s}, {e})")
                continue

            # Global indices for this trial
            trial_global_idx = np.arange(s, e, dtype=np.int32)
            # Which positions in valid_idx correspond to this trial?
            mask_trial_in_valid = np.isin(valid_idx, trial_global_idx)
            trial_valid_positions = np.where(mask_trial_in_valid)[0]

            if trial_valid_positions.size == 0:
                wr(f"Trial {trial_key}: no valid labeled windows in this range. Skipping.")
                continue

            trial_to_valid_idx[trial_key] = trial_valid_positions

            d_trial = dists[trial_valid_positions]
            mahal_mean = float(d_trial.mean())
            mahal_std = float(d_trial.std())

            trial_keys.append(trial_key)
            mahal_mean_per_trial.append(mahal_mean)
            mahal_std_per_trial.append(mahal_std)

            wr(f"Trial: {trial_key}")
            wr(f"  Windows in trial (valid): {d_trial.size}")
            wr(f"  Mahalanobis mean distance = {mahal_mean:.4f}")
            wr(f"  Mahalanobis std  distance = {mahal_std:.4f}")
            wr("")

        if not trial_keys:
            wr("No valid trials with labeled windows were found. Aborting further analyses.")
            wr("************  END OF TRAINING ANALYSIS 3  *************")
            return

        # Plot mean Mahalanobis distance per trial
        fig_path_mahal = os.path.join(analysis_dir, "Mahalanobis_MeanDistance_PerTrial.png")
        plt.figure(figsize=(max(6, len(trial_keys) * 0.4), 4))
        x = np.arange(len(trial_keys))
        plt.bar(x, mahal_mean_per_trial)
        plt.xticks(x, trial_keys, rotation=90)
        plt.ylabel("Mean Mahalanobis distance")
        plt.title("Per-trial Mahalanobis Mean Distance")
        plt.tight_layout()
        plt.savefig(fig_path_mahal)
        plt.close()
        wr(f"Saved mean Mahalanobis distance per trial plot: {fig_path_mahal}")
        wr("")

        # ------------------------------
        # 6) LDA margin (confidence) per trial
        # ------------------------------
        wr("=== LDA Margin (Confidence) Analysis ===")
        # Try to use C-trained Wg/Cg if available, otherwise reuse NumPy W/b
        use_c_params = hasattr(self, "Wg") and hasattr(self, "Cg") and len(self.Wg) == self.num_class * self.feature_dim
        if use_c_params:
            Wg_vec = np.array(self.Wg, dtype=np.float32)
            Cg_vec = np.array(self.Cg, dtype=np.float32)
            Wg_mat = Wg_vec.reshape((self.num_class, self.feature_dim), order="F")
            scores_for_margin = X_full @ Wg_mat.T + Cg_vec  # (N_valid, num_class)
        else:
            scores_for_margin = scores_base  # from NumPy LDA

        margins = np.zeros(N_valid, dtype=np.float32)
        for i in range(N_valid):
            true_c = y_full[i] - 1
            scores_i = scores_for_margin[i]
            true_score = scores_i[true_c]
            if scores_for_margin.shape[1] > 1:
                # max of all other classes
                other_scores = np.delete(scores_i, true_c)
                max_other = float(other_scores.max())
            else:
                max_other = true_score
            margins[i] = float(true_score - max_other)

        wr(f"Global LDA margin stats:")
        wr(f"  mean  = {float(margins.mean()):.4f}")
        wr(f"  std   = {float(margins.std()):.4f}")
        wr(f"  min   = {float(margins.min()):.4f}")
        wr(f"  max   = {float(margins.max()):.4f}")
        wr("")

        margin_mean_per_trial = []
        margin_std_per_trial = []

        for trial_key in trial_keys:
            idxs = trial_to_valid_idx[trial_key]
            m_trial = margins[idxs]
            margin_mean = float(m_trial.mean())
            margin_std = float(m_trial.std())

            margin_mean_per_trial.append(margin_mean)
            margin_std_per_trial.append(margin_std)

            wr(f"Trial: {trial_key}")
            wr(f"  Windows in trial (valid): {m_trial.size}")
            wr(f"  LDA margin mean = {margin_mean:.4f}")
            wr(f"  LDA margin std  = {margin_std:.4f}")
            wr("")

        fig_path_margin = os.path.join(analysis_dir, "LDA_Margin_Mean_PerTrial.png")
        plt.figure(figsize=(max(6, len(trial_keys) * 0.4), 4))
        x = np.arange(len(trial_keys))
        plt.bar(x, margin_mean_per_trial)
        plt.xticks(x, trial_keys, rotation=90)
        plt.ylabel("Mean LDA margin (true - best other)")
        plt.title("Per-trial LDA Margin (Confidence)")
        plt.tight_layout()
        plt.savefig(fig_path_margin)
        plt.close()
        wr(f"Saved mean LDA margin per trial plot: {fig_path_margin}")
        wr("")

        # ------------------------------
        # 7) Leave-One-Trial-Out (LOTO) training
        # ------------------------------
        wr("=== Leave-One-Trial-Out (LOTO) Analysis ===")
        wr("For each trial: train LDA on all *other* trials and test on this trial.")
        wr("This shows how well the model trained on the rest generalizes to that trial.")
        wr("")

        trial_loto_acc = []
        trial_loto_train_acc = []

        for trial_key in trial_keys:
            idx_trial = trial_to_valid_idx[trial_key]
            if idx_trial.size < self.num_class:
                wr(f"Trial {trial_key}: not enough windows ({idx_trial.size}) for a stable test. Skipping LOTO.")
                trial_loto_acc.append(float("nan"))
                trial_loto_train_acc.append(float("nan"))
                continue

            # Indices for the rest of the data
            mask_rest = np.ones(N_valid, dtype=bool)
            mask_rest[idx_trial] = False

            X_rest = X_full[mask_rest]
            y_rest = y_full[mask_rest]
            X_trial = X_full[idx_trial]
            y_trial = y_full[idx_trial]

            W_loto, b_loto, _, _ = fit_lda(X_rest, y_rest, self.num_class)
            if W_loto is None:
                wr(f"Trial {trial_key}: LDA fit failed when leaving this trial out (likely class imbalance).")
                trial_loto_acc.append(float("nan"))
                trial_loto_train_acc.append(float("nan"))
                continue

            preds_rest, _ = lda_predict(X_rest, W_loto, b_loto)
            preds_trial, _ = lda_predict(X_trial, W_loto, b_loto)

            acc_rest = float((preds_rest == y_rest).mean())
            acc_trial = float((preds_trial == y_trial).mean())

            trial_loto_train_acc.append(acc_rest)
            trial_loto_acc.append(acc_trial)

            wr(f"Trial: {trial_key}")
            wr(f"  LOTO train accuracy on remaining trials = {acc_rest*100:.2f}%")
            wr(f"  LOTO test accuracy on this trial        = {acc_trial*100:.2f}%")
            wr("")

        # Plot LOTO test accuracy per trial
        fig_path_loto = os.path.join(analysis_dir, "LOTO_Trial_Accuracy.png")
        plt.figure(figsize=(max(6, len(trial_keys) * 0.4), 4))
        x = np.arange(len(trial_keys))
        # Replace NaNs with 0 for plotting
        plot_vals = np.array(trial_loto_acc, dtype=np.float32)
        nan_mask = np.isnan(plot_vals)
        plot_vals[nan_mask] = 0.0
        plt.bar(x, plot_vals)
        plt.xticks(x, trial_keys, rotation=90)
        plt.ylabel("LOTO test accuracy")
        plt.ylim(0.0, 1.0)
        plt.title("Leave-One-Trial-Out Accuracy (test on left-out trial)")
        plt.tight_layout()
        plt.savefig(fig_path_loto)
        plt.close()
        wr(f"Saved LOTO per-trial accuracy plot: {fig_path_loto}")
        wr("")

        # ------------------------------
        # 8) Combined trial quality summary
        # ------------------------------
        wr("=== Combined Trial Quality Summary ===")
        wr("Each trial has:")
        wr("  - Mahalanobis mean distance (lower is better, more typical of its class)")
        wr("  - LDA margin mean (higher is better, more confident classification)")
        wr("  - LOTO accuracy (higher means better generalization from other trials)")
        wr("")

        # Build a summary table and sort by LOTO accuracy (ascending)
        summary = []
        for i, trial_key in enumerate(trial_keys):
            maha = mahal_mean_per_trial[i]
            marg = margin_mean_per_trial[i]
            acc_loto = trial_loto_acc[i]
            summary.append((trial_key, maha, marg, acc_loto))

        # Sort by LOTO accuracy (ascending, NaNs at the end)
        def sort_key(row):
            acc = row[3]
            if acc != acc:  # NaN check
                return (1, 0.0)
            return (0, acc)

        summary.sort(key=sort_key)

        wr("Trial summary sorted by LOTO accuracy (worst to best):")
        for trial_key, maha, marg, acc_loto in summary:
            acc_str = "NaN" if (acc_loto != acc_loto) else f"{acc_loto*100:.2f}%"
            # Simple qualitative rating
            if acc_loto != acc_loto:
                rating = "UNKNOWN"
            elif acc_loto >= 0.90 and maha <= np.median(mahal_mean_per_trial) and marg >= np.median(margin_mean_per_trial):
                rating = "GOOD"
            elif acc_loto >= 0.70:
                rating = "OK/USABLE"
            else:
                rating = "SUSPECT / REVIEW"

            wr(
                f"  {trial_key}: "
                f"LOTO_acc={acc_str}, "
                f"Mahalanobis_mean={maha:.4f}, "
                f"Margin_mean={marg:.4f}, "
                f"Rating={rating}"
            )

        wr("")
        wr("************  END OF TRAINING ANALYSIS 3  *************")
        wr("********************************************************")


    def test_analysis(self):
        """
        Analyze the TEST set and compare it with TRAIN performance.

        Outputs in ./Report analysis/ :
          - TestAnalysis_Report.txt
          - ConfusionMatrix_Test_Counts.png
          - ConfusionMatrix_Test_Normalized.png
          - TrainTest_FeatureShift.png
          - Test_Margin_Mean_PerTrial.png

        Includes:
          1) Build test feature matrix from Test/actX/trial_*.txt using tdfeats
          2) Apply training normalization (xmean, xstd)
          3) Classify test windows with trained LDA (Wg, Cg)
          4) Test confusion matrix + per-class precision/recall/F1
          5) Train vs Test per-class recall comparison
          6) Feature drift between TRAIN and TEST (mean/std shift)
          7) LDA margin (confidence) statistics per trial
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from pathlib import Path
        from array import array
        import ctypes

        # ------------------------------
        # 0) Basic checks
        # ------------------------------
        if self.testing_folder is None:
            raise ValueError("testing_folder is not set. Run split_training_testing() first.")

        if not hasattr(self, "Wg") or not hasattr(self, "Cg"):
            raise ValueError("Wg/Cg not found. Run train_model() before test_analysis().")

        if not hasattr(self, "featurematrix") or len(self.featurematrix) == 0:
            raise ValueError("Training featurematrix empty. Run train_model() before test_analysis().")

        if self.lib is None:
            raise RuntimeError("DLL libfunctions.dll not loaded; tdfeats is required for test_analysis().")

        # ------------------------------
        # 1) Setup report folder / file
        # ------------------------------
        cwd = os.getcwd()
        analysis_dir = os.path.join(cwd, "Report analysis")
        os.makedirs(analysis_dir, exist_ok=True)

        report_path = os.path.join(analysis_dir, "TestAnalysis_Report.txt")
        if os.path.exists(report_path):
            os.remove(report_path)

        def wr(line=""):
            with open(report_path, "a", encoding="utf-8") as f:
                f.write(str(line) + "\n")

        wr("********************************************************")
        wr("****************  TEST SET ANALYSIS  *******************")
        wr("********************************************************")
        wr(f"Train folder : {self.train_folder}")
        wr(f"Test  folder : {self.testing_folder}")
        wr(f"Analysis dir : {analysis_dir}")
        wr("")

        # ------------------------------
        # 2) Build test feature matrix + labels
        # ------------------------------
        test_root = Path(self.testing_folder)

        act_dirs = [
            d for d in test_root.iterdir()
            if d.is_dir() and d.name.lower().startswith("act")
        ]
        act_dirs.sort()

        if not act_dirs:
            wr("No act* folders found in Test/. Aborting test_analysis.")
            wr("********************************************************")
            return

        X_list = []        # list of feature rows
        y_list = []        # list of true labels (1..num_class)
        trial_ranges = {}  # trial_key -> (start_row, end_row)

        # temp buffer for one feature row computed by tdfeats
        feat_row = array('f', [0.0] * self.feature_dim)
        feat_ptr = ctypes.cast(
            feat_row.buffer_info()[0],
            ctypes.POINTER(ctypes.c_float)
        )

        wr("=== Building test feature matrix from Test/actX/trial_*.txt ===")

        total_trials = 0
        for class_idx, act_dir in enumerate(act_dirs):
            act_name = act_dir.name
            label_val = class_idx + 1  # true label for this folder

            trial_files = sorted(act_dir.glob("trial_*.txt"))
            if not trial_files:
                wr(f"WARNING: no trial_*.txt in {act_dir}")
                continue

            for trial_path in trial_files:
                total_trials += 1
                trial_key = f"{act_name}/{trial_path.name}"
                wr(f"Processing test trial: {trial_key}")

                # 1) load EMG into self.testdata
                self.load_trial_data(trial_path, label="testdata")

                # 2) slide windows over testdata and compute features
                start_idx = len(X_list)
                m = 0
                wins_this_trial = 0
                total_floats = self.channel * self.data_per_trial

                while m < self.win_per_trial:
                    start_float_index = m * self.winc * self.channel
                    end_float_index = start_float_index + self.wl * self.channel
                    if end_float_index > total_floats:
                        break

                    # slice of EMG window
                    window = self.testdata[start_float_index:end_float_index]
                    window_ptr = ctypes.cast(
                        window.buffer_info()[0],
                        ctypes.POINTER(ctypes.c_float)
                    )

                    # tdfeats writes into feat_row
                    self.lib.tdfeats(
                        window_ptr,
                        ctypes.c_int(self.wl),
                        ctypes.c_int(self.channel),
                        ctypes.c_int(0),  # Nframe index (0 because we reuse buffer)
                        feat_ptr,
                        ctypes.c_float(self.deadzone_zc),
                        ctypes.c_float(self.deadzone_turn),
                        ctypes.c_int(self.scale_mav),
                        ctypes.c_int(self.scale_zc),
                        ctypes.c_int(self.tdfeatureN)
                    )

                    # copy Python list of features
                    X_list.append([feat_row[j] for j in range(self.feature_dim)])
                    y_list.append(label_val)
                    wins_this_trial += 1
                    m += 1

                end_idx = len(X_list)
                trial_ranges[trial_key] = (start_idx, end_idx)
                wr(f"  Windows extracted in this trial: {wins_this_trial}")
                wr(f"  Feature rows: {start_idx} .. {end_idx - 1}")
                wr("")

        N_test = len(X_list)
        wr(f"Total test trials processed: {total_trials}")
        wr(f"Total test windows (rows): {N_test}")
        wr("")

        if N_test == 0:
            wr("No test windows extracted. Aborting test_analysis.")
            wr("********************************************************")
            return

        X_test = np.array(X_list, dtype=np.float32)
        y_test = np.array(y_list, dtype=np.int32)

        # ------------------------------
        # 3) Normalize test features with TRAIN xmean/xstd
        # ------------------------------
        wr("=== Applying training normalization (xmean, xstd) to TEST features ===")

        xmean = np.array(self.xmean, dtype=np.float32)
        xstd = np.array(self.xstd, dtype=np.float32)

        if xmean.size != self.feature_dim or xstd.size != self.feature_dim:
            raise ValueError(
                f"xmean/xstd size mismatch with feature_dim ({self.feature_dim})"
            )

        X_test_norm = X_test.copy()
        eps = 1e-8
        for j in range(self.feature_dim):
            std = float(xstd[j])
            if std > eps:
                X_test_norm[:, j] = (X_test_norm[:, j] - xmean[j]) / std
            else:
                X_test_norm[:, j] = X_test_norm[:, j] - xmean[j]

        wr("Normalization of test features complete.")
        wr("")

        # ------------------------------
        # 4) Build LDA matrix & classify TEST windows
        # ------------------------------
        wr("=== LDA classification on TEST windows ===")

        Wg_vec = np.array(self.Wg, dtype=np.float32)
        Cg_vec = np.array(self.Cg, dtype=np.float32)
        if Wg_vec.size != self.num_class * self.feature_dim:
            raise ValueError(
                f"Wg size {Wg_vec.size} != num_class*feature_dim {self.num_class * self.feature_dim}"
            )

        # C layout: Wg[i + j*CLASS]
        Wg_mat = Wg_vec.reshape((self.num_class, self.feature_dim), order="F")

        scores_test = X_test_norm @ Wg_mat.T + Cg_vec  # (N_test, num_class)
        preds_test = np.argmax(scores_test, axis=1) + 1

        # ------------------------------
        # 5) Test confusion matrix + per-class metrics
        # ------------------------------
        wr("=== TEST Confusion Matrix & Metrics ===")

        C_test = np.zeros((self.num_class, self.num_class), dtype=np.int32)
        for t, p in zip(y_test, preds_test):
            if 1 <= t <= self.num_class and 1 <= p <= self.num_class:
                C_test[t - 1, p - 1] += 1

        total_test_windows = C_test.sum()
        wr(f"Total labeled test windows: {total_test_windows}")
        wr("Confusion matrix (rows = true, cols = predicted):")
        for r in range(self.num_class):
            wr("  True class {}: ".format(r + 1) +
               "  ".join(f"{C_test[r, c]:5d}" for c in range(self.num_class)))
        wr("")

        # row-normalized confusion
        C_test_norm = C_test.astype(np.float32)
        row_sums = C_test_norm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        C_test_norm /= row_sums

        wr("Row-normalized confusion (TEST):")
        for r in range(self.num_class):
            wr("  True class {}: ".format(r + 1) +
               "  ".join(f"{C_test_norm[r, c]:0.3f}" for c in range(self.num_class)))
        wr("")

        # save confusion matrix images
        fig_path_counts = os.path.join(analysis_dir, "ConfusionMatrix_Test_Counts.png")
        plt.figure(figsize=(5, 4))
        im = plt.imshow(C_test, interpolation="nearest")
        plt.title("Test Confusion Matrix (Counts)")
        plt.xlabel("Predicted class")
        plt.ylabel("True class")
        plt.colorbar(im, label="Count")
        for i in range(self.num_class):
            for j in range(self.num_class):
                plt.text(j, i, str(C_test[i, j]), ha="center", va="center")
        plt.tight_layout()
        plt.savefig(fig_path_counts)
        plt.close()
        wr(f"Saved: {fig_path_counts}")

        fig_path_norm = os.path.join(analysis_dir, "ConfusionMatrix_Test_Normalized.png")
        plt.figure(figsize=(5, 4))
        im = plt.imshow(C_test_norm, vmin=0.0, vmax=1.0, interpolation="nearest")
        plt.title("Test Confusion Matrix (Row-normalized)")
        plt.xlabel("Predicted class")
        plt.ylabel("True class")
        plt.colorbar(im, label="Fraction")
        for i in range(self.num_class):
            for j in range(self.num_class):
                plt.text(j, i, f"{C_test_norm[i, j]:.2f}", ha="center", va="center")
        plt.tight_layout()
        plt.savefig(fig_path_norm)
        plt.close()
        wr(f"Saved: {fig_path_norm}")
        wr("")

        # per-class precision/recall/F1
        wr("Per-class TEST metrics:")
        test_recall = np.zeros(self.num_class, dtype=np.float32)
        for c in range(self.num_class):
            TP = C_test[c, c]
            FN = C_test[c, :].sum() - TP
            FP = C_test[:, c].sum() - TP
            TN = total_test_windows - TP - FN - FP

            prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            test_recall[c] = rec
            wr(f"  Class {c+1}: TP={TP}, FP={FP}, FN={FN}, TN={TN}")
            wr(f"    Precision = {prec*100:6.2f}%")
            wr(f"    Recall    = {rec*100:6.2f}%")
            wr(f"    F1 score  = {f1*100:6.2f}%")
        wr("")

        overall_test_acc = np.trace(C_test) / total_test_windows if total_test_windows > 0 else 0.0
        wr(f"Overall TEST window-level accuracy = {overall_test_acc*100:.2f}%")
        wr("")

        # ------------------------------
        # 6) TRAIN vs TEST comparison (using training featurematrix)
        # ------------------------------
        wr("=== TRAIN vs TEST Comparison (per-class recall) ===")

        feat_train = np.array(self.featurematrix, dtype=np.float32)
        if feat_train.size % self.feature_dim != 0:
            raise ValueError(
                f"Training featurematrix length {feat_train.size} not divisible by feature_dim {self.feature_dim}"
            )
        num_train_samples = feat_train.size // self.feature_dim
        feat_train = feat_train.reshape((num_train_samples, self.feature_dim))
        labels_train = np.array(self.trainclass, dtype=np.int32)

        valid_mask_train = (labels_train >= 1) & (labels_train <= self.num_class)
        X_train_norm = feat_train[valid_mask_train]
        y_train = labels_train[valid_mask_train]

        scores_train = X_train_norm @ Wg_mat.T + Cg_vec
        preds_train = np.argmax(scores_train, axis=1) + 1

        C_train = np.zeros((self.num_class, self.num_class), dtype=np.int32)
        for t, p in zip(y_train, preds_train):
            if 1 <= t <= self.num_class and 1 <= p <= self.num_class:
                C_train[t - 1, p - 1] += 1

        total_train_windows = C_train.sum()
        train_recall = np.zeros(self.num_class, dtype=np.float32)
        for c in range(self.num_class):
            TP = C_train[c, c]
            FN = C_train[c, :].sum() - TP
            rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            train_recall[c] = rec

        overall_train_acc = np.trace(C_train) / total_train_windows if total_train_windows > 0 else 0.0

        wr(f"Overall TRAIN window-level accuracy (recomputed) = {overall_train_acc*100:.2f}%")
        wr("")
        wr("Per-class recall (TRAIN vs TEST):")
        for c in range(self.num_class):
            tr = train_recall[c]
            te = test_recall[c]
            delta = te - tr
            wr(
                f"  Class {c+1}: "
                f"TRAIN recall={tr*100:6.2f}%, "
                f"TEST recall={te*100:6.2f}%, "
                f"Δ(TEST-TRAIN)={delta*100:6.2f}%"
            )
        wr("")

        # ------------------------------
        # 7) Feature drift between TRAIN and TEST
        # ------------------------------
        wr("=== Feature Drift (TRAIN vs TEST normalized features) ===")

        # means/stds in normalized space
        train_mean = X_train_norm.mean(axis=0)
        train_std = X_train_norm.std(axis=0, ddof=1)
        test_mean = X_test_norm.mean(axis=0)
        test_std = X_test_norm.std(axis=0, ddof=1)

        mean_shift = test_mean - train_mean
        # avoid div-by-zero
        std_ratio = np.ones_like(train_std)
        nz = train_std > 1e-6
        std_ratio[nz] = test_std[nz] / train_std[nz]

        drift_score = np.abs(mean_shift) + np.abs(std_ratio - 1.0)

        # plot drift_score per feature
        fig_path_shift = os.path.join(analysis_dir, "TrainTest_FeatureShift.png")
        plt.figure(figsize=(max(6, self.feature_dim * 0.25), 4))
        x = np.arange(self.feature_dim)
        plt.bar(x, drift_score)
        plt.xlabel("Feature index")
        plt.ylabel("Drift score (|Δmean| + |std_ratio-1|)")
        plt.title("Feature Drift between TRAIN and TEST (normalized space)")
        plt.tight_layout()
        plt.savefig(fig_path_shift)
        plt.close()
        wr(f"Saved feature-drift bar plot: {fig_path_shift}")
        wr("")

        # list top drifting features
        idx_sorted = np.argsort(-drift_score)
        top_k = min(20, self.feature_dim)
        wr(f"Top {top_k} drifting features (by drift_score):")
        for rank in range(top_k):
            j = int(idx_sorted[rank])
            wr(
                f"  Rank {rank+1:02d} - Feature {j}: "
                f"train_mean={train_mean[j]:.3f}, test_mean={test_mean[j]:.3f}, "
                f"train_std={train_std[j]:.3f}, test_std={test_std[j]:.3f}, "
                f"drift_score={drift_score[j]:.3f}"
            )
        wr("")

        # ------------------------------
        # 8) LDA margin (confidence) on TEST per trial
        # ------------------------------
        wr("=== LDA Margin (Confidence) on TEST ===")

        margins = np.zeros(N_test, dtype=np.float32)
        for i in range(N_test):
            true_c = y_test[i] - 1
            if not (0 <= true_c < self.num_class):
                margins[i] = 0.0
                continue
            scores_i = scores_test[i]
            true_score = scores_i[true_c]
            if self.num_class > 1:
                other_scores = np.delete(scores_i, true_c)
                max_other = float(other_scores.max())
            else:
                max_other = true_score
            margins[i] = float(true_score - max_other)

        wr("Global TEST margin stats:")
        wr(f"  mean = {float(margins.mean()):.4f}")
        wr(f"  std  = {float(margins.std()):.4f}")
        wr(f"  min  = {float(margins.min()):.4f}")
        wr(f"  max  = {float(margins.max()):.4f}")
        wr("")

        # per-trial margin stats & accuracy
        margin_mean_per_trial = []
        trial_acc_per_trial = []
        trial_keys = []
        rating_summary = []

        for trial_key, (s, e) in trial_ranges.items():
            if s < 0 or e > N_test or s >= e:
                wr(f"WARNING: invalid test range for trial {trial_key}: ({s}, {e})")
                continue

            idxs = np.arange(s, e, dtype=np.int32)
            m_trial = margins[idxs]
            y_trial = y_test[idxs]
            p_trial = preds_test[idxs]

            acc = float((y_trial == p_trial).mean()) if idxs.size > 0 else 0.0
            m_mean = float(m_trial.mean()) if idxs.size > 0 else 0.0

            trial_keys.append(trial_key)
            margin_mean_per_trial.append(m_mean)
            trial_acc_per_trial.append(acc)

            # rating
            if acc >= 0.9 and m_mean >= margins.mean():
                rating = "GOOD"
            elif acc >= 0.7:
                rating = "OK"
            else:
                rating = "SUSPECT"

            rating_summary.append((trial_key, acc, m_mean, rating))

            wr(f"Trial: {trial_key}")
            wr(f"  Test windows: {idxs.size}")
            wr(f"  Accuracy = {acc*100:.2f}%")
            wr(f"  Mean margin = {m_mean:.4f}")
            wr(f"  Rating: {rating}")
            wr("")

        # Plot mean TEST margin per trial
        if trial_keys:
            fig_path_margin = os.path.join(analysis_dir, "Test_Margin_Mean_PerTrial.png")
            plt.figure(figsize=(max(6, len(trial_keys) * 0.4), 4))
            x = np.arange(len(trial_keys))
            plt.bar(x, margin_mean_per_trial)
            plt.xticks(x, trial_keys, rotation=90)
            plt.ylabel("Mean LDA margin (true - best other)")
            plt.title("Per-trial LDA Margin (TEST)")
            plt.tight_layout()
            plt.savefig(fig_path_margin)
            plt.close()
            wr(f"Saved mean TEST margin per trial plot: {fig_path_margin}")
            wr("")

        # Sort rating summary by accuracy (ascending)
        if rating_summary:
            rating_summary.sort(key=lambda tup: tup[1])  # by acc
            wr("Trials sorted by TEST accuracy (worst to best):")
            for trial_key, acc, m_mean, rating in rating_summary:
                wr(f"  {trial_key}: acc={acc*100:.2f}%, margin={m_mean:.4f}, rating={rating}")
            wr("")

        wr("************  END OF TEST SET ANALYSIS  ***************")
        wr("********************************************************")


    def test_analysis2(self):
        """
        TEST ANALYSIS 2: Distribution & distance analysis.

        Uses:
          - TRAIN normalized features (self.featurematrix, self.trainclass)
          - TEST raw trials in self.testing_folder
          - LDA parameters (self.Wg, self.Cg, self.xmean, self.xstd)

        Outputs (in ./Report analysis/):
          - TestAnalysis2_Report.txt
          - Test_Mahalanobis_TrueDist_PerTrial.png
          - Test_Mahalanobis_Ratio_PerTrial.png
          - Test_Mahalanobis_Hist_Correct_vs_Incorrect.png

        For each test window:
          - Mahalanobis distance to the TRUE class mean (d_true)
          - Mahalanobis distance to NEAREST OTHER class (d_best_other)
          - ratio = d_best_other / d_true

        For each test trial:
          - accuracy (using trained LDA Wg/Cg)
          - mean(d_true), mean(ratio)
          - rating = GOOD / OK / SUSPECT
        """
        import os
        from pathlib import Path
        from array import array
        import ctypes
        import numpy as np
        import matplotlib.pyplot as plt

        # ------------------------------
        # 0) Basic checks
        # ------------------------------
        if self.testing_folder is None:
            raise ValueError("testing_folder is not set. Run split_training_testing() first.")
        if self.train_folder is None:
            raise ValueError("train_folder is not set. Run split_training_testing() first.")
        if not hasattr(self, "featurematrix") or len(self.featurematrix) == 0:
            raise ValueError("featurematrix is empty. Run train_model() before test_analysis2().")
        if self.lib is None:
            raise RuntimeError("DLL libfunctions.dll not loaded; tdfeats is required.")
        if not hasattr(self, "Wg") or not hasattr(self, "Cg"):
            raise ValueError("Wg/Cg not found. Run train_model() before test_analysis2().")

        # ------------------------------
        # 1) Setup report folder / file
        # ------------------------------
        cwd = os.getcwd()
        analysis_dir = os.path.join(cwd, "Report analysis")
        os.makedirs(analysis_dir, exist_ok=True)

        report_path = os.path.join(analysis_dir, "TestAnalysis2_Report.txt")
        if os.path.exists(report_path):
            os.remove(report_path)

        def wr(line=""):
            with open(report_path, "a", encoding="utf-8") as f:
                f.write(str(line) + "\n")

        wr("********************************************************")
        wr("****************  TEST SET ANALYSIS 2  *****************")
        wr("********************************************************")
        wr(f"Train folder : {self.train_folder}")
        wr(f"Test  folder : {self.testing_folder}")
        wr(f"Analysis dir : {analysis_dir}")
        wr("")

        # ------------------------------
        # 2) TRAIN side: build normalized feature matrix & LDA params
        # ------------------------------
        wr("=== Building TRAIN normalized features for distance model ===")

        feat_train = np.array(self.featurematrix, dtype=np.float32)
        if feat_train.size % self.feature_dim != 0:
            raise ValueError(
                f"featurematrix length {feat_train.size} is not divisible by feature_dim {self.feature_dim}"
            )
        num_train_samples = feat_train.size // self.feature_dim
        feat_train = feat_train.reshape((num_train_samples, self.feature_dim))

        labels_train = np.array(self.trainclass, dtype=np.int32)
        if labels_train.size != num_train_samples:
            raise ValueError(
                f"trainclass length {labels_train.size} does not match num_samples {num_train_samples}"
            )

        # Only valid labels 1..num_class
        valid_mask_train = (labels_train >= 1) & (labels_train <= self.num_class)
        X_train = feat_train[valid_mask_train]
        y_train = labels_train[valid_mask_train]
        N_train = X_train.shape[0]

        wr(f"TRAIN windows used: {N_train}")
        wr(f"Feature dimension: {self.feature_dim}")
        wr("")

        if N_train == 0:
            wr("No valid training windows – aborting test_analysis2.")
            wr("********************************************************")
            return

        # class means in normalized space
        num_class = self.num_class
        d = self.feature_dim
        means = np.zeros((num_class, d), dtype=np.float64)
        priors = np.zeros(num_class, dtype=np.float64)

        for k in range(num_class):
            idx = (y_train == (k + 1))
            Xk = X_train[idx]
            priors[k] = Xk.shape[0] / float(N_train) if N_train > 0 else 0.0
            if Xk.shape[0] > 0:
                means[k, :] = Xk.mean(axis=0)
            else:
                means[k, :] = 0.0

        wr("Per-class TRAIN counts & priors:")
        for k in range(num_class):
            cnt = int((y_train == (k + 1)).sum())
            wr(f"  Class {k+1}: count={cnt}, prior={priors[k]:.3f}")
        wr("")

        # shared covariance in normalized space
        reg = 1e-3  # regularization
        # subtract class mean for each sample
        X_centered = np.zeros_like(X_train, dtype=np.float64)
        for i in range(N_train):
            c = y_train[i] - 1
            X_centered[i, :] = X_train[i, :] - means[c, :]

        cov = (X_centered.T @ X_centered) / max(N_train - num_class, 1)
        cov += reg * np.eye(d, dtype=np.float64)

        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            wr("WARNING: Covariance inversion failed; increasing regularization.")
            cov += 1e-2 * np.eye(d, dtype=np.float64)
            cov_inv = np.linalg.inv(cov)

        wr("Covariance matrix (shared) constructed and inverted.")
        wr("")

        # For LDA predictions (same as in test_analysis)
        Wg_vec = np.array(self.Wg, dtype=np.float32)
        Cg_vec = np.array(self.Cg, dtype=np.float32)
        if Wg_vec.size != num_class * d:
            raise ValueError(
                f"Wg size {Wg_vec.size} != num_class*feature_dim {num_class * d}"
            )
        Wg_mat = Wg_vec.reshape((num_class, d), order="F")

        # ------------------------------
        # 3) Build TEST feature matrix (normalized)
        # ------------------------------
        wr("=== Building TEST features using tdfeats + training normalization ===")

        test_root = Path(self.testing_folder)
        act_dirs = [
            d for d in test_root.iterdir()
            if d.is_dir() and d.name.lower().startswith("act")
        ]
        act_dirs.sort()

        if not act_dirs:
            wr("No act* folders found in Test/. Aborting.")
            wr("********************************************************")
            return

        X_list = []
        y_list = []
        trial_ranges = {}   # trial_key -> (start_row, end_row)

        feat_row = array('f', [0.0] * d)
        feat_ptr = ctypes.cast(
            feat_row.buffer_info()[0],
            ctypes.POINTER(ctypes.c_float)
        )

        xmean = np.array(self.xmean, dtype=np.float32)
        xstd = np.array(self.xstd, dtype=np.float32)
        eps = 1e-8

        total_trials = 0
        for class_idx, act_dir in enumerate(act_dirs):
            act_name = act_dir.name
            label_val = class_idx + 1

            trial_files = sorted(act_dir.glob("trial_*.txt"))
            if not trial_files:
                wr(f"WARNING: no trial_*.txt in {act_dir}")
                continue

            for trial_path in trial_files:
                total_trials += 1
                trial_key = f"{act_name}/{trial_path.name}"
                wr(f"Processing TEST trial: {trial_key}")

                # Load EMG into self.testdata
                self.load_trial_data(trial_path, label="testdata")

                start_idx = len(X_list)
                m = 0
                wins_this_trial = 0
                total_floats = self.channel * self.data_per_trial

                while m < self.win_per_trial:
                    start_float_index = m * self.winc * self.channel
                    end_float_index = start_float_index + self.wl * self.channel
                    if end_float_index > total_floats:
                        break

                    window = self.testdata[start_float_index:end_float_index]
                    window_ptr = ctypes.cast(
                        window.buffer_info()[0],
                        ctypes.POINTER(ctypes.c_float)
                    )

                    # C tdfeats -> feat_row (UN-normalized)
                    self.lib.tdfeats(
                        window_ptr,
                        ctypes.c_int(self.wl),
                        ctypes.c_int(self.channel),
                        ctypes.c_int(0),
                        feat_ptr,
                        ctypes.c_float(self.deadzone_zc),
                        ctypes.c_float(self.deadzone_turn),
                        ctypes.c_int(self.scale_mav),
                        ctypes.c_int(self.scale_zc),
                        ctypes.c_int(self.tdfeatureN)
                    )

                    # copy to np array and apply training normalization
                    f_vec = np.array([feat_row[j] for j in range(d)], dtype=np.float32)
                    for j in range(d):
                        std = float(xstd[j])
                        if std > eps:
                            f_vec[j] = (f_vec[j] - xmean[j]) / std
                        else:
                            f_vec[j] = f_vec[j] - xmean[j]

                    X_list.append(f_vec)
                    y_list.append(label_val)
                    wins_this_trial += 1
                    m += 1

                end_idx = len(X_list)
                trial_ranges[trial_key] = (start_idx, end_idx)
                wr(f"  Windows in this trial: {wins_this_trial}")
                wr(f"  Feature rows: {start_idx}..{end_idx - 1}")
                wr("")

        N_test = len(X_list)
        wr(f"Total TEST trials processed: {total_trials}")
        wr(f"Total TEST windows: {N_test}")
        wr("")

        if N_test == 0:
            wr("No TEST windows extracted – aborting test_analysis2.")
            wr("********************************************************")
            return

        X_test = np.stack(X_list, axis=0)  # already normalized
        y_test = np.array(y_list, dtype=np.int32)

        # ------------------------------
        # 4) LDA predictions on TEST (for correct/incorrect)
        # ------------------------------
        wr("=== LDA predictions on TEST windows (for correctness labels) ===")

        scores_test = X_test @ Wg_mat.T + Cg_vec   # (N_test, num_class)
        preds_test = np.argmax(scores_test, axis=1) + 1

        correct_mask = (preds_test == y_test)
        total_correct = int(correct_mask.sum())
        test_acc = total_correct / float(N_test)
        wr(f"Overall TEST window-level accuracy (for this analysis) = {test_acc*100:.2f}%")
        wr("")

        # ------------------------------
        # 5) Mahalanobis distances and ratios
        # ------------------------------
        wr("=== Computing Mahalanobis distances (TEST vs TRAIN model) ===")

        def maha_sq(x, mean_vec):
            diff = x - mean_vec
            return float(diff @ cov_inv @ diff)

        d_true = np.zeros(N_test, dtype=np.float64)
        d_best_other = np.zeros(N_test, dtype=np.float64)
        ratio = np.zeros(N_test, dtype=np.float64)

        for i in range(N_test):
            c_true = y_test[i] - 1
            if not (0 <= c_true < num_class):
                d_true[i] = 0.0
                d_best_other[i] = 0.0
                ratio[i] = 0.0
                continue

            x = X_test[i, :]

            # distance to true class
            dt_sq = maha_sq(x, means[c_true, :])
            dt = np.sqrt(max(dt_sq, 0.0))
            d_true[i] = dt

            # distance to all other classes
            best = None
            for k in range(num_class):
                if k == c_true:
                    continue
                dk_sq = maha_sq(x, means[k, :])
                dk = np.sqrt(max(dk_sq, 0.0))
                if (best is None) or (dk < best):
                    best = dk

            if best is None:
                best = dt  # single-class degenerate case

            d_best_other[i] = best
            if dt > 1e-8:
                ratio[i] = best / dt
            else:
                ratio[i] = 1.0

        wr("Global Mahalanobis stats (TRUE class distance):")
        wr(f"  mean(d_true) = {float(d_true.mean()):.4f}")
        wr(f"  std(d_true)  = {float(d_true.std()):.4f}")
        wr(f"  min(d_true)  = {float(d_true.min()):.4f}")
        wr(f"  max(d_true)  = {float(d_true.max()):.4f}")
        wr("")
        wr("Global distance ratio stats (d_best_other / d_true):")
        wr(f"  mean(ratio) = {float(ratio.mean()):.4f}")
        wr(f"  std(ratio)  = {float(ratio.std()):.4f}")
        wr(f"  min(ratio)  = {float(ratio.min()):.4f}")
        wr(f"  max(ratio)  = {float(ratio.max()):.4f}")
        wr("")

        # ------------------------------
        # 6) Per-trial metrics + rating
        # ------------------------------
        wr("=== Per-trial distance metrics & ratings ===")

        trial_keys = []
        mean_d_true_per_trial = []
        mean_ratio_per_trial = []
        acc_per_trial = []
        ratings = []

        global_mean_ratio = float(ratio.mean())
        global_mean_d_true = float(d_true.mean())

        for trial_key, (s, e) in trial_ranges.items():
            if s < 0 or e > N_test or s >= e:
                wr(f"WARNING: invalid index range for trial {trial_key}: ({s}, {e})")
                continue

            idxs = np.arange(s, e, dtype=np.int32)
            y_t = y_test[idxs]
            p_t = preds_test[idxs]
            dt_t = d_true[idxs]
            r_t = ratio[idxs]

            acc = float((y_t == p_t).mean()) if idxs.size > 0 else 0.0
            md = float(dt_t.mean()) if idxs.size > 0 else 0.0
            mr = float(r_t.mean()) if idxs.size > 0 else 0.0

            # rating rule: combines accuracy + ratio + distance
            if acc >= 0.90 and mr >= global_mean_ratio and md <= global_mean_d_true:
                rating = "GOOD"
            elif acc >= 0.70:
                rating = "OK"
            else:
                rating = "SUSPECT"

            trial_keys.append(trial_key)
            mean_d_true_per_trial.append(md)
            mean_ratio_per_trial.append(mr)
            acc_per_trial.append(acc)
            ratings.append((trial_key, acc, md, mr, rating))

            wr(f"Trial: {trial_key}")
            wr(f"  windows: {idxs.size}")
            wr(f"  accuracy           = {acc*100:.2f}%")
            wr(f"  mean d_true        = {md:.4f}")
            wr(f"  mean ratio         = {mr:.4f}")
            wr(f"  rating             = {rating}")
            wr("")

        # sort trials from worst to best by accuracy
        if ratings:
            ratings_sorted = sorted(ratings, key=lambda t: t[1])  # by accuracy
            wr("Trials sorted by TEST accuracy (worst to best):")
            for trial_key, acc, md, mr, rating in ratings_sorted:
                wr(
                    f"  {trial_key}: acc={acc*100:.2f}%, "
                    f"mean_d_true={md:.4f}, mean_ratio={mr:.4f}, rating={rating}"
                )
            wr("")

        # ------------------------------
        # 7) Plots: per-trial distances & ratios
        # ------------------------------
        if trial_keys:
            x = np.arange(len(trial_keys))

            # (a) mean d_true per trial
            fig_path_dt = os.path.join(analysis_dir, "Test_Mahalanobis_TrueDist_PerTrial.png")
            plt.figure(figsize=(max(6, len(trial_keys) * 0.4), 4))
            plt.bar(x, mean_d_true_per_trial)
            plt.xticks(x, trial_keys, rotation=90)
            plt.ylabel("Mean Mahalanobis distance to TRUE class")
            plt.title("Per-trial TRUE Mahalanobis distance (TEST)")
            plt.tight_layout()
            plt.savefig(fig_path_dt)
            plt.close()
            wr(f"Saved: {fig_path_dt}")

            # (b) mean ratio per trial
            fig_path_ratio = os.path.join(analysis_dir, "Test_Mahalanobis_Ratio_PerTrial.png")
            plt.figure(figsize=(max(6, len(trial_keys) * 0.4), 4))
            plt.bar(x, mean_ratio_per_trial)
            plt.axhline(global_mean_ratio, linestyle="--")
            plt.xticks(x, trial_keys, rotation=90)
            plt.ylabel("Mean distance ratio (d_best_other / d_true)")
            plt.title("Per-trial distance ratio (TEST)")
            plt.tight_layout()
            plt.savefig(fig_path_ratio)
            plt.close()
            wr(f"Saved: {fig_path_ratio}")
            wr("")

        # ------------------------------
        # 8) Histogram: d_true for correct vs incorrect
        # ------------------------------
        wr("=== Histograms: d_true for correct vs incorrect predictions ===")

        d_true_correct = d_true[correct_mask]
        d_true_incorrect = d_true[~correct_mask]

        fig_path_hist = os.path.join(analysis_dir, "Test_Mahalanobis_Hist_Correct_vs_Incorrect.png")
        plt.figure(figsize=(6, 4))
        bins = 30
        if d_true_correct.size > 0:
            plt.hist(d_true_correct, bins=bins, alpha=0.6, density=True, label="Correct")
        if d_true_incorrect.size > 0:
            plt.hist(d_true_incorrect, bins=bins, alpha=0.6, density=True, label="Incorrect")
        plt.xlabel("Mahalanobis distance to TRUE class")
        plt.ylabel("Density")
        plt.title("TEST: d_true distribution (correct vs incorrect)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path_hist)
        plt.close()
        wr(f"Saved: {fig_path_hist}")
        wr("")

        wr("************  END OF TEST SET ANALYSIS 2  *************")
        wr("********************************************************")


    def train_analysis4(self):
        """
        TRAINING ANALYSIS 4: Temporal feature trajectories over time.

        Goal:
          - Look at how each feature behaves over time within a trial,
            and then average across trials for each class.
          - Answer questions like:
              * Do MAV/WL/etc drift up or down across the trial?
              * Are some actions more stable over time than others?
              * Which feature-channel combos have strong trends?

        Inputs (must be populated by train_model):
          - self.featurematrix           : normalized training features, flattened
          - self.trainclass              : labels (1..num_class) aligned with feature rows
          - self.trial_feature_ranges    : dict {trial_key: (start_row, end_row)}
          - self.num_class, self.channel, self.tdfeatureN, self.feature_dim

        Outputs (in ./Report analysis/):
          - TrainAnalysis4_Report.txt
          - Train_FeatureTime_Class{c}_Feat{fname}_Trajectories.png
              (for each class c and each feature type fname, e.g., MAV, WL, etc.)
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt

        # ------------------------------
        # 0) Basic checks
        # ------------------------------
        if self.train_folder is None:
            raise ValueError("train_folder is not set. Run split_training_testing() and train_model() first.")

        if not hasattr(self, "featurematrix") or len(self.featurematrix) == 0:
            raise ValueError("featurematrix is empty. Run train_model() before train_analysis4().")

        if not hasattr(self, "trial_feature_ranges") or not self.trial_feature_ranges:
            raise ValueError(
                "trial_feature_ranges is missing or empty. "
                "Make sure train_model() populates self.trial_feature_ranges."
            )

        # ------------------------------
        # 1) Setup report folder / file
        # ------------------------------
        cwd = os.getcwd()
        analysis_dir = os.path.join(cwd, "Report analysis")
        os.makedirs(analysis_dir, exist_ok=True)

        report_path = os.path.join(analysis_dir, "TrainAnalysis4_Report.txt")
        if os.path.exists(report_path):
            os.remove(report_path)

        def wr(line=""):
            with open(report_path, "a", encoding="utf-8") as f:
                f.write(str(line) + "\n")

        wr("********************************************************")
        wr("************  TRAINING DATA ANALYSIS 4  ****************")
        wr("********  TEMPORAL FEATURE TRAJECTORIES  ***************")
        wr("********************************************************")
        wr(f"Train folder: {self.train_folder}")
        wr(f"Analysis folder: {analysis_dir}")
        wr("")

        # ------------------------------
        # 2) Convert buffers to NumPy
        # ------------------------------
        feat = np.array(self.featurematrix, dtype=np.float32)
        if feat.size % self.feature_dim != 0:
            raise ValueError(
                f"featurematrix length {feat.size} is not divisible by feature_dim {self.feature_dim}"
            )
        num_samples = feat.size // self.feature_dim
        feat = feat.reshape((num_samples, self.feature_dim))

        labels = np.array(self.trainclass, dtype=np.int32)
        if labels.size != num_samples:
            raise ValueError(
                f"trainclass length {labels.size} does not match num_samples {num_samples}"
            )

        # Only keep valid rows (labels 1..num_class)
        valid_mask = (labels >= 1) & (labels <= self.num_class)
        feat = feat[valid_mask]
        labels = labels[valid_mask]
        wr(f"Total windows (rows) in featurematrix: {num_samples}")
        wr(f"Valid labeled windows (1..{self.num_class}): {feat.shape[0]}")
        wr(f"Feature dimension: {self.feature_dim}")
        wr("")

        # ------------------------------
        # 3) Get feature names & helper mapping
        # ------------------------------
        tdfeature_names = getattr(self, "tdfeature_names", None)
        if tdfeature_names is None or len(tdfeature_names) < self.tdfeatureN:
            tdfeature_names = [f"F{i+1}" for i in range(self.tdfeatureN)]

        def feat_index(channel_idx, feat_type_idx):
            """
            Map (channel, feature_type) -> flat feature index.

            channel_idx: 0..(self.channel-1)
            feat_type_idx: 0..(self.tdfeatureN-1)
            """
            return channel_idx * self.tdfeatureN + feat_type_idx

        # ------------------------------
        # 4) Build per-trial info from trial_feature_ranges
        # ------------------------------
        # trial_info: {trial_key: {"class": c, "start": s, "end": e, "len": L}}
        trial_info = {}
        class_to_trials = {c: [] for c in range(1, self.num_class + 1)}

        # NOTE: self.trial_feature_ranges is in terms of row indices
        # AFTER windows were added in train_model, and aligned with self.trainclass.
        # Here we re-check that they fall inside our valid range.
        for trial_key, (s, e) in self.trial_feature_ranges.items():
            if s < 0 or e > labels.size or s >= e:
                # skip bad range (should not happen if train_model is consistent)
                wr(f"WARNING: invalid trial_feature_range for {trial_key}: ({s}, {e}) – skipping.")
                continue

            trial_labels = labels[s:e]
            if trial_labels.size == 0:
                wr(f"WARNING: no labels in range for {trial_key} – skipping.")
                continue

            # Use majority label as the "class" for this trial
            vals, counts = np.unique(trial_labels, return_counts=True)
            c_major = int(vals[np.argmax(counts)])

            L = e - s
            trial_info[trial_key] = {
                "class": c_major,
                "start": s,
                "end": e,
                "len": L,
            }
            class_to_trials[c_major].append(trial_key)

        wr("=== Trials discovered in trial_feature_ranges ===")
        for c in range(1, self.num_class + 1):
            wr(f"  Class {c}: {len(class_to_trials[c])} trials")
        wr("")

        # ------------------------------
        # 5) Per-class temporal trajectories
        # ------------------------------
        wr("=== Temporal feature trends per class ===")
        wr("For each class, we average feature trajectories over trials (in window order).")
        wr("We also fit a simple linear slope over time to detect upward/downward drifts.")
        wr("")

        # We'll record slopes for summary:
        # slopes_summary[(class, ch, feat_type)] = slope
        slopes_summary = {}

        for c in range(1, self.num_class + 1):
            trials_c = class_to_trials[c]
            if not trials_c:
                wr(f"Class {c}: no trials – skipping temporal analysis.")
                wr("")
                continue

            # Determine a common length: minimum number of windows among trials
            lengths = [trial_info[tk]["len"] for tk in trials_c]
            T_min = int(min(lengths))
            if T_min < 2:
                wr(f"Class {c}: T_min={T_min} < 2 windows – skipping temporal analysis.")
                wr("")
                continue

            wr(f"Class {c}: {len(trials_c)} trials, min windows per trial = {T_min}")
            wr("")

            # Build 3D tensor: (num_trials_c, T_min, feature_dim)
            num_trials_c = len(trials_c)
            traj = np.zeros((num_trials_c, T_min, self.feature_dim), dtype=np.float32)

            for idx_t, trial_key in enumerate(trials_c):
                info = trial_info[trial_key]
                s = info["start"]
                e = info["end"]
                # truncate to T_min for alignment
                X_trial = feat[s:s + T_min, :]
                if X_trial.shape[0] != T_min:
                    # In case of any weird mismatch, pad/skip – here we skip after warning
                    wr(f"  WARNING: trial {trial_key} has {X_trial.shape[0]} rows (expected {T_min}); skipping.")
                    continue
                traj[idx_t, :, :] = X_trial

            # Recompute num_trials_c in case we skipped some trials
            valid_trial_mask = (traj.sum(axis=(1, 2)) != 0)  # some trials might be all zeros if skipped
            traj_valid = traj[valid_trial_mask]
            num_trials_c_valid = traj_valid.shape[0]
            if num_trials_c_valid == 0:
                wr(f"Class {c}: after filtering invalid trials, none left for temporal analysis.")
                wr("")
                continue

            wr(f"Class {c}: using {num_trials_c_valid} valid trials for temporal averaging.")
            wr("")

            # Mean & std trajectory over trials: (T_min, feature_dim)
            mean_traj = traj_valid.mean(axis=0)
            std_traj = traj_valid.std(axis=0, ddof=1)

            # time axis for windows
            t = np.arange(T_min, dtype=np.float32)

            # For each feature type, make one figure with all channels
            for feat_type_idx in range(self.tdfeatureN):
                fname = tdfeature_names[feat_type_idx] if feat_type_idx < len(tdfeature_names) else f"F{feat_type_idx+1}"

                plt.figure(figsize=(8, 4))
                for ch in range(self.channel):
                    j = feat_index(ch, feat_type_idx)
                    y = mean_traj[:, j]

                    # store slope for summary
                    if T_min >= 2:
                        slope = float(np.polyfit(t, y, 1)[0])
                    else:
                        slope = 0.0
                    slopes_summary[(c, ch, feat_type_idx)] = slope

                    plt.plot(
                        t,
                        y,
                        label=f"ch{ch+1} (slope={slope:.3f})"
                    )

                plt.xlabel("Window index (time order)")
                plt.ylabel("Normalized feature value")
                plt.title(f"Class {c} – Temporal trajectory of {fname}")
                plt.legend()
                plt.tight_layout()

                # Clean feature name for filename
                safe_fname = "".join(ch_ for ch_ in fname if ch_.isalnum() or ch_ in ("_", "-"))
                fig_path = os.path.join(
                    analysis_dir,
                    f"Train_FeatureTime_Class{c}_Feat{safe_fname}_Trajectories.png"
                )
                plt.savefig(fig_path)
                plt.close()

                wr(f"Saved temporal trajectory plot for Class {c}, feature {fname}: {fig_path}")

            wr("")

        # ------------------------------
        # 6) Global summary of strongest trends
        # ------------------------------
        wr("=== Global summary of strongest temporal trends ===")
        if not slopes_summary:
            wr("No slopes were computed (likely not enough windows).")
            wr("************  END OF TRAINING ANALYSIS 4  *************")
            wr("********************************************************")
            return

        # Collect into list for sorting
        slope_items = []
        for (c, ch, feat_type_idx), slope in slopes_summary.items():
            fname = tdfeature_names[feat_type_idx] if feat_type_idx < len(tdfeature_names) else f"F{feat_type_idx+1}"
            slope_items.append((c, ch, feat_type_idx, fname, slope))

        # Sort by slope (descending: strongest positive trend)
        slope_items_pos = sorted(slope_items, key=lambda x: x[4], reverse=True)
        # Sort by slope (ascending: strongest negative trend)
        slope_items_neg = sorted(slope_items, key=lambda x: x[4])

        top_k = min(10, len(slope_items))

        wr(f"Top {top_k} strongest POSITIVE temporal trends (increasing over time):")
        for i in range(top_k):
            c, ch, fidx, fname, slope = slope_items_pos[i]
            wr(f"  #{i+1}: Class {c}, ch{ch+1}, feature={fname}, slope={slope:.4f}")
        wr("")

        wr(f"Top {top_k} strongest NEGATIVE temporal trends (decreasing over time):")
        for i in range(top_k):
            c, ch, fidx, fname, slope = slope_items_neg[i]
            wr(f"  #{i+1}: Class {c}, ch{ch+1}, feature={fname}, slope={slope:.4f}")
        wr("")

        wr("Interpretation:")
        wr("  * Large positive slope  -> feature tends to increase as the trial progresses.")
        wr("  * Large negative slope  -> feature tends to decrease over the trial.")
        wr("  * Near-zero slope       -> feature is relatively stable over time.")
        wr("")
        wr("************  END OF TRAINING ANALYSIS 4  *************")
        wr("********************************************************")

    def test_analysis3(self):
        """
        TEST ANALYSIS 3: Temporal trajectory comparison (TRAIN vs TEST).

        Goal:
          - Compare how features evolve over time (window index) in TRAIN vs TEST
            for each class, channel, and feature type.
          - Detect differences like:
              * TRAIN feature stable but TEST feature drifting
              * TRAIN and TEST slopes with different signs
              * TEST features noisier or less consistent

        Requirements:
          - Run AFTER train_model(), so that:
              * self.featurematrix
              * self.trainclass
              * self.trial_feature_ranges
            are populated for TRAIN.
          - self.testing_folder points to the Test/ root.
          - DLL (self.lib) is loaded and tdfeats is available.
          - xmean/xstd, tdfeatureN, channel, wl, winc, data_per_trial, win_per_trial are set.

        Outputs (in ./Report analysis/):
          - TestAnalysis3_TemporalTrajectory_Report.txt
          - TrainTest_FeatureTime_Class{c}_Feat{fname}_Trajectories.png
                (overlaid TRAIN vs TEST curves, per class and feature type)
        """
        import os
        from pathlib import Path
        from array import array
        import ctypes
        import numpy as np
        import matplotlib.pyplot as plt

        # ------------------------------
        # 0) Basic checks
        # ------------------------------
        if self.train_folder is None:
            raise ValueError("train_folder is not set. Run split_training_testing() and train_model() first.")
        if self.testing_folder is None:
            raise ValueError("testing_folder is not set. Run split_training_testing() first.")
        if not hasattr(self, "featurematrix") or len(self.featurematrix) == 0:
            raise ValueError("featurematrix is empty. Run train_model() before test_analysis3().")
        if not hasattr(self, "trial_feature_ranges") or not self.trial_feature_ranges:
            raise ValueError("trial_feature_ranges missing. Make sure train_model() populates it.")
        if self.lib is None:
            raise RuntimeError("DLL libfunctions.dll not loaded; tdfeats is required.")
        if not hasattr(self, "xmean") or not hasattr(self, "xstd"):
            raise ValueError("xmean/xstd not found. Run train_model() before test_analysis3().")

        # ------------------------------
        # 1) Setup report folder / file
        # ------------------------------
        cwd = os.getcwd()
        analysis_dir = os.path.join(cwd, "Report analysis")
        os.makedirs(analysis_dir, exist_ok=True)

        report_path = os.path.join(analysis_dir, "TestAnalysis3_TemporalTrajectory_Report.txt")
        if os.path.exists(report_path):
            os.remove(report_path)

        def wr(line=""):
            with open(report_path, "a", encoding="utf-8") as f:
                f.write(str(line) + "\n")

        wr("********************************************************")
        wr("************  TEST ANALYSIS 3 (TEMPORAL)  **************")
        wr("********************************************************")
        wr(f"Train folder : {self.train_folder}")
        wr(f"Test  folder : {self.testing_folder}")
        wr(f"Analysis dir : {analysis_dir}")
        wr("")
        wr("This analysis compares TRAIN vs TEST temporal feature trajectories")
        wr("for each class, channel, and feature type.")
        wr("")

        # ------------------------------
        # 2) Convert TRAIN featurematrix to 2D array
        # ------------------------------
        feat_train = np.array(self.featurematrix, dtype=np.float32)
        if feat_train.size % self.feature_dim != 0:
            raise ValueError(
                f"featurematrix length {feat_train.size} is not divisible by feature_dim {self.feature_dim}"
            )
        num_train_samples = feat_train.size // self.feature_dim
        feat_train = feat_train.reshape((num_train_samples, self.feature_dim))

        labels_train = np.array(self.trainclass, dtype=np.int32)
        if labels_train.size != num_train_samples:
            raise ValueError(
                f"trainclass length {labels_train.size} does not match num_train_samples {num_train_samples}"
            )

        wr(f"TRAIN: total windows = {num_train_samples}")
        wr(f"Feature dimension   = {self.feature_dim}")
        wr("")

        # ------------------------------
        # 3) Helper: feature names and index mapping
        # ------------------------------
        tdfeature_names = getattr(self, "tdfeature_names", None)
        if tdfeature_names is None or len(tdfeature_names) < self.tdfeatureN:
            tdfeature_names = [f"F{i+1}" for i in range(self.tdfeatureN)]

        def feat_index(channel_idx, feat_type_idx):
            """
            Map (channel, feature_type) -> flat feature index.

            channel_idx: 0..(self.channel-1)
            feat_type_idx: 0..(self.tdfeatureN-1)
            """
            return channel_idx * self.tdfeatureN + feat_type_idx

        # ------------------------------
        # 4) Build TRAIN trial info
        # ------------------------------
        trial_info_train = {}
        class_to_trials_train = {c: [] for c in range(1, self.num_class + 1)}

        for trial_key, (s, e) in self.trial_feature_ranges.items():
            if s < 0 or e > labels_train.size or s >= e:
                wr(f"WARNING: invalid train range for {trial_key}: ({s}, {e}) – skipping.")
                continue

            trial_labels = labels_train[s:e]
            if trial_labels.size == 0:
                wr(f"WARNING: no labels in train range for {trial_key} – skipping.")
                continue

            vals, counts = np.unique(trial_labels, return_counts=True)
            c_major = int(vals[np.argmax(counts)])
            L = e - s

            trial_info_train[trial_key] = {
                "class": c_major,
                "start": s,
                "end": e,
                "len": L,
            }
            if 1 <= c_major <= self.num_class:
                class_to_trials_train[c_major].append(trial_key)

        wr("TRAIN: trials per class (from trial_feature_ranges):")
        for c in range(1, self.num_class + 1):
            wr(f"  Class {c}: {len(class_to_trials_train[c])} trials")
        wr("")

        # ------------------------------
        # 5) Build TEST feature matrix + trial ranges (similar to test_analysis)
        # ------------------------------
        wr("=== Building TEST feature matrix (for temporal analysis) ===")

        test_root = Path(self.testing_folder)
        act_dirs = [
            d for d in test_root.iterdir()
            if d.is_dir() and d.name.lower().startswith("act")
        ]
        act_dirs.sort()

        if not act_dirs:
            wr("No act* folders found in Test/. Aborting test_analysis3.")
        X_list_test = []
        y_list_test = []
        trial_ranges_test = {}

        feat_row = array('f', [0.0] * self.feature_dim)
        feat_ptr = ctypes.cast(
            feat_row.buffer_info()[0],
            ctypes.POINTER(ctypes.c_float)
        )

        xmean = np.array(self.xmean, dtype=np.float32)
        xstd = np.array(self.xstd, dtype=np.float32)
        eps = 1e-8

        total_trials_test = 0
        for class_idx, act_dir in enumerate(act_dirs):
            act_name = act_dir.name
            label_val = class_idx + 1  # assume same ordering as Train

            trial_files = sorted(act_dir.glob("trial_*.txt"))
            if not trial_files:
                wr(f"WARNING: no trial_*.txt in {act_dir}")
                continue

            for trial_path in trial_files:
                total_trials_test += 1
                trial_key = f"{act_name}/{trial_path.name}"
                wr(f"Processing TEST trial: {trial_key}")

                # Load EMG into self.testdata
                self.load_trial_data(trial_path, label="testdata")

                start_idx = len(X_list_test)
                m = 0
                wins_this_trial = 0
                total_floats = self.channel * self.data_per_trial

                while m < self.win_per_trial:
                    start_float_index = m * self.winc * self.channel
                    end_float_index = start_float_index + self.wl * self.channel
                    if end_float_index > total_floats:
                        break

                    window = self.testdata[start_float_index:end_float_index]
                    window_ptr = ctypes.cast(
                        window.buffer_info()[0],
                        ctypes.POINTER(ctypes.c_float)
                    )

                    # C tdfeats -> feat_row (UN-normalized)
                    self.lib.tdfeats(
                        window_ptr,
                        ctypes.c_int(self.wl),
                        ctypes.c_int(self.channel),
                        ctypes.c_int(0),
                        feat_ptr,
                        ctypes.c_float(self.deadzone_zc),
                        ctypes.c_float(self.deadzone_turn),
                        ctypes.c_int(self.scale_mav),
                        ctypes.c_int(self.scale_zc),
                        ctypes.c_int(self.tdfeatureN)
                    )

                    # copy + normalize with TRAIN stats
                    f_vec = np.array([feat_row[j] for j in range(self.feature_dim)], dtype=np.float32)
                    for j in range(self.feature_dim):
                        std = float(xstd[j])
                        if std > eps:
                            f_vec[j] = (f_vec[j] - xmean[j]) / std
                        else:
                            f_vec[j] = f_vec[j] - xmean[j]

                    X_list_test.append(f_vec)
                    y_list_test.append(label_val)
                    wins_this_trial += 1
                    m += 1

                end_idx = len(X_list_test)
                trial_ranges_test[trial_key] = (start_idx, end_idx)
                wr(f"  Windows in this TEST trial: {wins_this_trial}")
                wr(f"  Feature rows: {start_idx}..{end_idx - 1}")
                wr("")

        N_test = len(X_list_test)
        wr(f"Total TEST trials processed: {total_trials_test}")
        wr(f"Total TEST windows: {N_test}")
        wr("")

        if N_test == 0:
            wr("No TEST windows extracted – aborting test_analysis3.")
            wr("************  END OF TEST ANALYSIS 3  ****************")
            wr("********************************************************")
            return

        feat_test = np.stack(X_list_test, axis=0)
        labels_test = np.array(y_list_test, dtype=np.int32)

        # Build TEST trial info (similar to TRAIN)
        trial_info_test = {}
        class_to_trials_test = {c: [] for c in range(1, self.num_class + 1)}
        for trial_key, (s, e) in trial_ranges_test.items():
            if s < 0 or e > labels_test.size or s >= e:
                wr(f"WARNING: invalid TEST range for {trial_key}: ({s}, {e}) – skipping.")
                continue

            t_labels = labels_test[s:e]
            if t_labels.size == 0:
                wr(f"WARNING: no labels in TEST range for {trial_key} – skipping.")
                continue

            vals, counts = np.unique(t_labels, return_counts=True)
            c_major = int(vals[np.argmax(counts)])
            L = e - s

            trial_info_test[trial_key] = {
                "class": c_major,
                "start": s,
                "end": e,
                "len": L,
            }
            if 1 <= c_major <= self.num_class:
                class_to_trials_test[c_major].append(trial_key)

        wr("TEST: trials per class:")
        for c in range(1, self.num_class + 1):
            wr(f"  Class {c}: {len(class_to_trials_test[c])} trials")
        wr("")

        # ------------------------------
        # 6) Per-class TRAIN vs TEST temporal trajectories
        # ------------------------------
        wr("=== Per-class TRAIN vs TEST temporal trajectories ===")
        wr("We align trials by window index and compute mean trajectories.")
        wr("Then we fit linear slopes for TRAIN and TEST to quantify trends.")
        wr("")

        slopes_diff_summary = []

        for c in range(1, self.num_class + 1):
            trials_tr = class_to_trials_train[c]
            trials_te = class_to_trials_test[c]

            if not trials_tr or not trials_te:
                wr(f"Class {c}: missing trials in TRAIN or TEST (Train={len(trials_tr)}, Test={len(trials_te)}). Skipping.")
                wr("")
                continue

            # TRAIN lengths and min length
            lengths_tr = [trial_info_train[tk]["len"] for tk in trials_tr]
            T_min_tr = int(min(lengths_tr))

            # TEST lengths and min length
            lengths_te = [trial_info_test[tk]["len"] for tk in trials_te]
            T_min_te = int(min(lengths_te))

            T_min = min(T_min_tr, T_min_te)
            if T_min < 2:
                wr(f"Class {c}: T_min={T_min} < 2 windows – skipping.")
                wr("")
                continue

            wr(f"Class {c}: TRAIN trials={len(trials_tr)}, TEST trials={len(trials_te)}, common windows={T_min}")
            wr("")

            # Build TRAIN tensor: (num_trains, T_min, feat_dim)
            traj_tr = np.zeros((len(trials_tr), T_min, self.feature_dim), dtype=np.float32)
            valid_tr_mask = np.ones(len(trials_tr), dtype=bool)

            for idx_t, trial_key in enumerate(trials_tr):
                info = trial_info_train[trial_key]
                s = info["start"]
                X_trial = feat_train[s:s + T_min, :]
                if X_trial.shape[0] != T_min:
                    wr(f"  WARNING: TRAIN trial {trial_key} has {X_trial.shape[0]} rows (expected {T_min}); skipping.")
                    valid_tr_mask[idx_t] = False
                    continue
                traj_tr[idx_t, :, :] = X_trial

            traj_tr = traj_tr[valid_tr_mask]
            if traj_tr.shape[0] == 0:
                wr(f"Class {c}: no valid TRAIN trials after filtering – skipping.")
                wr("")
                continue

            # Build TEST tensor: (num_tests, T_min, feat_dim)
            traj_te = np.zeros((len(trials_te), T_min, self.feature_dim), dtype=np.float32)
            valid_te_mask = np.ones(len(trials_te), dtype=bool)

            for idx_t, trial_key in enumerate(trials_te):
                info = trial_info_test[trial_key]
                s = info["start"]
                X_trial = feat_test[s:s + T_min, :]
                if X_trial.shape[0] != T_min:
                    wr(f"  WARNING: TEST trial {trial_key} has {X_trial.shape[0]} rows (expected {T_min}); skipping.")
                    valid_te_mask[idx_t] = False
                    continue
                traj_te[idx_t, :, :] = X_trial

            traj_te = traj_te[valid_te_mask]
            if traj_te.shape[0] == 0:
                wr(f"Class {c}: no valid TEST trials after filtering – skipping.")
                wr("")
                continue

            wr(f"  Using {traj_tr.shape[0]} TRAIN and {traj_te.shape[0]} TEST trials for class {c}.")
            wr("")

            # Mean trajectories (T_min, feat_dim)
            mean_tr_tr = traj_tr.mean(axis=0)
            mean_tr_te = traj_te.mean(axis=0)

            t = np.arange(T_min, dtype=np.float32)

            # For each feature type, overlay TRAIN vs TEST
            for feat_type_idx in range(self.tdfeatureN):
                fname = tdfeature_names[feat_type_idx] if feat_type_idx < len(tdfeature_names) else f"F{feat_type_idx+1}"

                plt.figure(figsize=(8, 4))
                for ch in range(self.channel):
                    j = feat_index(ch, feat_type_idx)
                    y_tr = mean_tr_tr[:, j]
                    y_te = mean_tr_te[:, j]

                    # fit slopes
                    slope_tr = float(np.polyfit(t, y_tr, 1)[0]) if T_min >= 2 else 0.0
                    slope_te = float(np.polyfit(t, y_te, 1)[0]) if T_min >= 2 else 0.0
                    slope_diff = slope_te - slope_tr

                    slopes_diff_summary.append((c, ch, feat_type_idx, fname, slope_tr, slope_te, slope_diff))

                    # plot TRAIN (solid) and TEST (dashed) for this channel
                    plt.plot(t, y_tr, label=f"TRAIN ch{ch+1} (slope={slope_tr:.3f})")
                    plt.plot(t, y_te, linestyle="--", label=f"TEST  ch{ch+1} (slope={slope_te:.3f})")

                plt.xlabel("Window index (time order)")
                plt.ylabel("Normalized feature value")
                plt.title(f"Class {c} – TRAIN vs TEST temporal trajectory of {fname}")
                plt.legend()
                plt.tight_layout()

                safe_fname = "".join(ch_ for ch_ in fname if ch_.isalnum() or ch_ in ("_", "-"))
                fig_path = os.path.join(
                    analysis_dir,
                    f"TrainTest_FeatureTime_Class{c}_Feat{safe_fname}_Trajectories.png"
                )
                plt.savefig(fig_path)
                plt.close()

                wr(f"Saved TRAIN vs TEST temporal trajectory plot for Class {c}, feature {fname}: {fig_path}")
            wr("")

        # ------------------------------
        # 7) Global slope difference summary
        # ------------------------------
        wr("=== Global summary: TRAIN vs TEST slope differences ===")
        if not slopes_diff_summary:
            wr("No slopes were computed (likely not enough valid trials in any class).")
            wr("************  END OF TEST ANALYSIS 3  ****************")
            wr("********************************************************")
            return

        # sort by absolute difference in slope
        slopes_diff_sorted = sorted(slopes_diff_summary, key=lambda x: abs(x[6]), reverse=True)
        top_k = min(20, len(slopes_diff_sorted))

        wr(f"Top {top_k} feature-channel-class combos with largest |TEST slope - TRAIN slope|:")
        for i in range(top_k):
            c, ch, fidx, fname, slope_tr, slope_te, slope_diff = slopes_diff_sorted[i]
            wr(
                f"  #{i+1:02d}: Class {c}, ch{ch+1}, feature={fname}, "
                f"TRAIN slope={slope_tr:.4f}, TEST slope={slope_te:.4f}, "
                f"Δslope(TEST-TRAIN)={slope_diff:.4f}"
            )
        wr("")
        wr("Interpretation:")
        wr("  * If TRAIN and TEST slopes are similar and near zero: temporal behavior is consistent.")
        wr("  * If TEST slope is much more positive than TRAIN: feature tends to grow more over time in TEST.")
        wr("  * If TEST slope is much more negative than TRAIN: feature decays faster in TEST (possible fatigue/drift).")
        wr("")
        wr("************  END OF TEST ANALYSIS 3  ****************")
        wr("********************************************************")

    def train_test_trial_distance_analysis(self):
        """
        TRAIN + TEST ANALYSIS: Pairwise distances between trial centroids.

        For TRAIN:
          - Uses self.featurematrix (normalized training features)
          - Uses self.trainclass (labels)
          - Uses self.trial_feature_ranges (trial -> [start_row, end_row])
          - Computes a centroid (mean feature vector) per training trial
          - Computes pairwise Euclidean distance between trial centroids
          - Saves a heatmap: Train_TrialCentroid_DistanceHeatmap.png

        For TEST:
          - Rebuilds features from self.testing_folder using tdfeats
          - Normalizes using TRAIN xmean/xstd
          - Computes centroids per test trial
          - Computes pairwise Euclidean distance between test trial centroids
          - Saves a heatmap: Test_TrialCentroid_DistanceHeatmap.png

        Also writes a text report:
          - TrainTest_TrialCentroidDistance_Report.txt
        """
        import os
        from pathlib import Path
        from array import array
        import ctypes
        import numpy as np
        import matplotlib.pyplot as plt

        # ------------------------------
        # 0) Basic checks
        # ------------------------------
        if self.train_folder is None:
            raise ValueError("train_folder is not set. Run split_training_testing() and train_model() first.")
        if self.testing_folder is None:
            raise ValueError("testing_folder is not set. Run split_training_testing() first.")
        if not hasattr(self, "featurematrix") or len(self.featurematrix) == 0:
            raise ValueError("featurematrix is empty. Run train_model() before train_test_trial_distance_analysis().")
        if not hasattr(self, "trial_feature_ranges") or not self.trial_feature_ranges:
            raise ValueError("trial_feature_ranges missing. Make sure train_model() populates it.")
        if self.lib is None:
            raise RuntimeError("DLL libfunctions.dll not loaded; tdfeats is required.")
        if not hasattr(self, "xmean") or not hasattr(self, "xstd"):
            raise ValueError("xmean/xstd not found. Run train_model() before this analysis.")

        # ------------------------------
        # 1) Setup report folder / file
        # ------------------------------
        cwd = os.getcwd()
        analysis_dir = os.path.join(cwd, "Report analysis")
        os.makedirs(analysis_dir, exist_ok=True)

        report_path = os.path.join(analysis_dir, "TrainTest_TrialCentroidDistance_Report.txt")
        if os.path.exists(report_path):
            os.remove(report_path)

        def wr(line=""):
            with open(report_path, "a", encoding="utf-8") as f:
                f.write(str(line) + "\n")

        wr("********************************************************")
        wr("*** TRAIN + TEST TRIAL CENTROID DISTANCE ANALYSIS   ***")
        wr("********************************************************")
        wr(f"Train folder : {self.train_folder}")
        wr(f"Test  folder : {self.testing_folder}")
        wr(f"Analysis dir : {analysis_dir}")
        wr("")

        # ------------------------------
        # Helper: build centroid matrix and labels from (X, labels, trial_ranges)
        # ------------------------------
        def build_centroids(X, labels, trial_ranges, dataset_name):
            """
            X           : (N, D) normalized features
            labels      : (N,) class labels (1..num_class)
            trial_ranges: dict { trial_key: (start, end) }
            dataset_name: string used for logging

            Returns:
              centroids : (T, D)
              trial_names : list[str] length T
            """
            keys = sorted(trial_ranges.keys())
            centroids = []
            names = []

            wr(f"=== {dataset_name}: building trial centroids ===")
            for trial_key in keys:
                s, e = trial_ranges[trial_key]
                if s < 0 or e > labels.size or s >= e:
                    wr(f"  WARNING: invalid range for {dataset_name} trial {trial_key}: ({s}, {e}) – skipping.")
                    continue

                X_trial = X[s:e, :]
                if X_trial.shape[0] == 0:
                    wr(f"  WARNING: {dataset_name} trial {trial_key} has zero rows – skipping.")
                    continue

                # majority class for naming
                trial_labels = labels[s:e]
                vals, counts = np.unique(trial_labels, return_counts=True)
                c_major = int(vals[np.argmax(counts)])

                centroid = X_trial.mean(axis=0)
                centroids.append(centroid)

                # name like Class1_trial_1.txt
                base_name = Path(trial_key).name
                names.append(f"Class{c_major}_{base_name}")

                wr(f"  {dataset_name} trial {trial_key}: class={c_major}, windows={X_trial.shape[0]}")

            if not centroids:
                wr(f"No valid trials found for {dataset_name}.")
                return None, None

            centroids = np.stack(centroids, axis=0)
            wr(f"{dataset_name}: total valid trials with centroids = {centroids.shape[0]}")
            wr("")
            return centroids, names

        # Helper: make distance heatmap
        def plot_distance_heatmap(centroids, names, title, fig_path):
            """
            centroids: (T, D)
            names    : list[str] length T
            """
            if centroids is None or len(names) == 0:
                return

            T, D = centroids.shape
            # pairwise Euclidean distance
            Dmat = np.zeros((T, T), dtype=np.float32)
            for i in range(T):
                diff = centroids[i] - centroids
                Dmat[i, :] = np.sqrt(np.sum(diff * diff, axis=1))

            plt.figure(figsize=(max(8, T * 0.5), max(6, T * 0.5)))
            im = plt.imshow(Dmat, interpolation="nearest")
            plt.title(title)
            plt.xticks(range(T), names, rotation=90)
            plt.yticks(range(T), names)
            plt.colorbar(im, label="Euclidean distance")
            plt.tight_layout()
            plt.savefig(fig_path)
            plt.close()

            wr(f"Saved distance heatmap: {fig_path}")
            wr("")

        # ------------------------------
        # 2) TRAIN side: centroids from self.featurematrix
        # ------------------------------
        feat_train = np.array(self.featurematrix, dtype=np.float32)
        if feat_train.size % self.feature_dim != 0:
            raise ValueError(
                f"featurematrix length {feat_train.size} is not divisible by feature_dim {self.feature_dim}"
            )
        num_train_samples = feat_train.size // self.feature_dim
        feat_train = feat_train.reshape((num_train_samples, self.feature_dim))

        labels_train = np.array(self.trainclass, dtype=np.int32)
        if labels_train.size != num_train_samples:
            raise ValueError(
                f"trainclass length {labels_train.size} does not match num_train_samples {num_train_samples}"
            )

        wr("=== TRAIN side: computing centroids ===")
        wr(f"TRAIN: windows = {num_train_samples}, feature_dim = {self.feature_dim}")
        wr(f"TRAIN: #trial_feature_ranges = {len(self.trial_feature_ranges)}")
        wr("")

        centroids_train, names_train = build_centroids(
            feat_train,
            labels_train,
            self.trial_feature_ranges,
            dataset_name="TRAIN"
        )

        if centroids_train is not None:
            fig_train = os.path.join(analysis_dir, "Train_TrialCentroid_DistanceHeatmap.png")
            plot_distance_heatmap(
                centroids_train,
                names_train,
                title="Distance Between TRAIN Trial Centroids",
                fig_path=fig_train,
            )

        # ------------------------------
        # 3) TEST side: build normalized features & trial ranges, then centroids
        # ------------------------------
        wr("=== TEST side: extracting features for trial centroids ===")

        test_root = Path(self.testing_folder)
        act_dirs = [
            d for d in test_root.iterdir()
            if d.is_dir() and d.name.lower().startswith("act")
        ]
        act_dirs.sort()

        if not act_dirs:
            wr("No act* folders found in Test/. Cannot build TEST centroids.")
            wr("************  END OF TRIAL CENTROID ANALYSIS  ********")
            wr("********************************************************")
            return

        X_list = []
        y_list = []
        trial_ranges_test = {}

        feat_row = array('f', [0.0] * self.feature_dim)
        feat_ptr = ctypes.cast(
            feat_row.buffer_info()[0],
            ctypes.POINTER(ctypes.c_float)
        )

        xmean = np.array(self.xmean, dtype=np.float32)
        xstd = np.array(self.xstd, dtype=np.float32)
        eps = 1e-8

        total_trials_test = 0
        for class_idx, act_dir in enumerate(act_dirs):
            act_name = act_dir.name
            label_val = class_idx + 1

            trial_files = sorted(act_dir.glob("trial_*.txt"))
            if not trial_files:
                wr(f"WARNING: no trial_*.txt in {act_dir}")
                continue

            for trial_path in trial_files:
                total_trials_test += 1
                trial_key = f"{act_name}/{trial_path.name}"
                wr(f"Processing TEST trial for centroid: {trial_key}")

                self.load_trial_data(trial_path, label="testdata")

                start_idx = len(X_list)
                m = 0
                wins_this_trial = 0
                total_floats = self.channel * self.data_per_trial

                while m < self.win_per_trial:
                    start_float_index = m * self.winc * self.channel
                    end_float_index = start_float_index + self.wl * self.channel
                    if end_float_index > total_floats:
                        break

                    window = self.testdata[start_float_index:end_float_index]
                    window_ptr = ctypes.cast(
                        window.buffer_info()[0],
                        ctypes.POINTER(ctypes.c_float)
                    )

                    self.lib.tdfeats(
                        window_ptr,
                        ctypes.c_int(self.wl),
                        ctypes.c_int(self.channel),
                        ctypes.c_int(0),
                        feat_ptr,
                        ctypes.c_float(self.deadzone_zc),
                        ctypes.c_float(self.deadzone_turn),
                        ctypes.c_int(self.scale_mav),
                        ctypes.c_int(self.scale_zc),
                        ctypes.c_int(self.tdfeatureN)
                    )

                    # copy + normalize with TRAIN stats
                    f_vec = np.array([feat_row[j] for j in range(self.feature_dim)], dtype=np.float32)
                    for j in range(self.feature_dim):
                        std = float(xstd[j])
                        if std > eps:
                            f_vec[j] = (f_vec[j] - xmean[j]) / std
                        else:
                            f_vec[j] = f_vec[j] - xmean[j]

                    X_list.append(f_vec)
                    y_list.append(label_val)
                    wins_this_trial += 1
                    m += 1

                end_idx = len(X_list)
                trial_ranges_test[trial_key] = (start_idx, end_idx)
                wr(f"  TEST windows in trial: {wins_this_trial}")
                wr(f"  Feature rows: {start_idx}..{end_idx - 1}")
                wr("")

        N_test = len(X_list)
        wr(f"Total TEST trials processed (for centroids): {total_trials_test}")
        wr(f"Total TEST windows: {N_test}")
        wr("")

        if N_test == 0:
            wr("No TEST windows extracted – cannot build TEST centroids.")
            wr("************  END OF TRIAL CENTROID ANALYSIS  ********")
            wr("********************************************************")
            return

        feat_test = np.stack(X_list, axis=0)
        labels_test = np.array(y_list, dtype=np.int32)

        centroids_test, names_test = build_centroids(
            feat_test,
            labels_test,
            trial_ranges_test,
            dataset_name="TEST"
        )

        if centroids_test is not None:
            fig_test = os.path.join(analysis_dir, "Test_TrialCentroid_DistanceHeatmap.png")
            plot_distance_heatmap(
                centroids_test,
                names_test,
                title="Distance Between TEST Trial Centroids",
                fig_path=fig_test,
            )

        wr("************  END OF TRIAL CENTROID ANALYSIS  ********")
        wr("********************************************************")


    def external_test_analysis(self, external_folder):
        """
        Run the full TEST analysis suite on a completely separate dataset,
        without changing the TRAIN folder or retraining the model.

        Parameters
        ----------
        external_folder : str or Path
            Path to the *Test root* for the external dataset, e.g.:

                r"...\\set3\\Test"

            This folder must contain:
                act1\\trial_*.txt, act2\\trial_*.txt, ...

        Behavior
        --------
        - Resolves and sanity-checks the external folder.
        - Temporarily sets self.testing_folder = external_folder.
        - Runs the full analysis stack on that external Test set:

              test_analysis()
              test_analysis2()
              test_analysis3()
              train_test_trial_distance_analysis()

          using the *existing* trained model (Wg, Cg, xmean, xstd, etc.)
          that was produced by train_model().

        - Restores the previous self.testing_folder at the end (even on error).

        Notes
        -----
        - Results are written into the same "Report analysis" folder, using the
          same filenames as usual (e.g., TestAnalysis_Report.txt, etc.), so
          running external_test_analysis() multiple times will overwrite the
          previous test-analysis figures/reports.
        """
        from pathlib import Path

        # Resolve and sanity-check the folder
        ext_path = Path(external_folder).resolve()
        if not ext_path.exists():
            raise FileNotFoundError(
                f"external_test_analysis: folder does not exist: {ext_path}"
            )
        if not ext_path.is_dir():
            raise NotADirectoryError(
                f"external_test_analysis: not a directory: {ext_path}"
            )

        # Quick check that there is at least one act* folder
        act_dirs = [
            d for d in ext_path.iterdir()
            if d.is_dir() and d.name.lower().startswith("act")
        ]
        if not act_dirs:
            raise ValueError(
                f"external_test_analysis: no 'act*' folders found inside {ext_path}. "
                "You probably want to pass the Test root, e.g. .../set3/Test"
            )

        # Save current testing folder so we can restore it later
        old_testing_folder = self.testing_folder

        try:
            # Point the object to the external test folder
            self.testing_folder = str(ext_path)

            # Log into the main report so you can see what was run
            self.add_report("")
            self.add_report("=== External TEST analysis run ===")
            self.add_report(f"  External test folder: {self.testing_folder}")
            self.add_report("  Running:")
            self.add_report("    - test_analysis()")
            self.add_report("    - test_analysis2()")
            self.add_report("    - test_analysis3()")
            self.add_report("    - train_test_trial_distance_analysis()")
            self.add_report("============================================")

            # --- Run your full test-analysis stack on this external set ---
            self.test_analysis()
            self.test_analysis2()
            self.test_analysis3()
            self.train_test_trial_distance_analysis()
            self.compare_train_test_features()
            self.train_test_confusion()
            self.confusion_matrix_train_vs_test_trials()

        finally:
            # Restore whatever testing folder you had before
            self.testing_folder = old_testing_folder


    def compare_train_test_features(self):
        """
        Compare TRAIN vs TEST feature distributions.

        Requires
        --------
        - train_model() already executed
        - self.testing_folder set and containing actX/trial_*.txt

        Outputs (in ./Report analysis/):
        - CompareFeatures_Report.txt
        - FeatureDrift.png
        - CentroidDriftHeatmap.png
        - MahalanobisDistance_Test.png
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from pathlib import Path
        from array import array
        import ctypes

        # ----------------------------
        # 0) Basic checks
        # ----------------------------
        if self.train_folder is None:
            raise ValueError("Run train_model() before compare_train_test_features().")
        if self.testing_folder is None:
            raise ValueError("testing_folder is not set.")
        if not hasattr(self, "featurematrix") or len(self.featurematrix) == 0:
            raise ValueError("Training featurematrix is empty.")
        if not hasattr(self, "xmean") or not hasattr(self, "xstd"):
            raise ValueError("xmean/xstd not found. Run train_model() first.")
        if self.lib is None:
            raise RuntimeError("DLL libfunctions.dll not loaded; tdfeats is required.")

        # ----------------------------
        # 1) Create output folder / report
        # ----------------------------
        cwd = os.getcwd()
        analysis_dir = os.path.join(cwd, "Report analysis")
        os.makedirs(analysis_dir, exist_ok=True)

        report_path = os.path.join(analysis_dir, "CompareFeatures_Report.txt")
        if os.path.exists(report_path):
            os.remove(report_path)

        def wr(line=""):
            with open(report_path, "a", encoding="utf-8") as f:
                f.write(str(line) + "\n")

        wr("********************************************************")
        wr("********  TRAIN vs TEST FEATURE COMPARISON  ************")
        wr("********************************************************")
        wr(f"Train folder: {self.train_folder}")
        wr(f"Test folder:  {self.testing_folder}")
        wr("")

        # ----------------------------
        # 2) TRAIN features (already normalized by train_model)
        # ----------------------------
        feat_train = np.array(self.featurematrix, dtype=np.float32)
        if feat_train.size % self.feature_dim != 0:
            raise ValueError("featurematrix size mismatch with feature_dim.")
        feat_train = feat_train.reshape((-1, self.feature_dim))
        labels_train = np.array(self.trainclass, dtype=np.int32)

        wr(f"Train samples: {feat_train.shape[0]}")
        wr(f"Feature dim:   {self.feature_dim}")
        wr("")

        # ----------------------------
        # 3) Build TEST feature matrix
        #    (raw EMG -> tdfeats -> normalize with TRAIN xmean/xstd)
        # ----------------------------
        test_feats = []
        test_labels = []

        test_root = Path(self.testing_folder)
        act_dirs = [
            d for d in test_root.iterdir()
            if d.is_dir() and d.name.lower().startswith("act")
        ]
        act_dirs.sort()

        # buffer for tdfeats output
        feat_row = array('f', [0.0] * self.feature_dim)
        feat_ptr = ctypes.cast(
            feat_row.buffer_info()[0],
            ctypes.POINTER(ctypes.c_float)
        )

        xmean = np.array(self.xmean, dtype=np.float32)
        xstd = np.array(self.xstd, dtype=np.float32)
        eps = 1e-8

        for class_idx, act_dir in enumerate(act_dirs):
            act_name = act_dir.name
            trial_files = sorted(act_dir.glob("trial_*.txt"))
            if not trial_files:
                wr(f"WARNING: no trial_*.txt in {act_dir}")
                continue

            for trial_path in trial_files:
                wr(f"Processing TEST trial: {act_name}/{trial_path.name}")
                # load EMG into self.testdata (interleaved floats)
                self.load_trial_data(trial_path, label="testdata")

                m = 0
                total_floats = len(self.testdata)
                while m < self.win_per_trial:
                    start_float_index = m * self.winc * self.channel
                    end_float_index = start_float_index + self.wl * self.channel
                    if end_float_index > total_floats:
                        break

                    # slice EMG window and get pointer
                    window = self.testdata[start_float_index:end_float_index]
                    window_ptr = ctypes.cast(
                        window.buffer_info()[0],
                        ctypes.POINTER(ctypes.c_float)
                    )

                    # call C tdfeats -> writes into feat_row
                    self.lib.tdfeats(
                        window_ptr,
                        ctypes.c_int(self.wl),
                        ctypes.c_int(self.channel),
                        ctypes.c_int(0),  # frame index (unused)
                        feat_ptr,
                        ctypes.c_float(self.deadzone_zc),
                        ctypes.c_float(self.deadzone_turn),
                        ctypes.c_int(self.scale_mav),
                        ctypes.c_int(self.scale_zc),
                        ctypes.c_int(self.tdfeatureN),
                    )

                    # convert to numpy & normalize with TRAIN stats
                    feat_np = np.array(feat_row, dtype=np.float32)
                    for j in range(self.feature_dim):
                        std = float(xstd[j])
                        if std > eps:
                            feat_np[j] = (feat_np[j] - xmean[j]) / std
                        else:
                            feat_np[j] = feat_np[j] - xmean[j]

                    test_feats.append(feat_np)
                    test_labels.append(class_idx + 1)
                    m += 1

                wr(f"  Windows extracted: {m}")
                wr("")

        if not test_feats:
            wr("No TEST windows extracted. Aborting.")
            wr("********************************************************")
            return

        feat_test = np.vstack(test_feats)
        labels_test = np.array(test_labels, dtype=np.int32)

        wr(f"Test samples: {feat_test.shape[0]}")
        wr("")

        # ---------------------------------------------------
        # 4) Per-feature drift statistics
        # ---------------------------------------------------
        wr("=== Feature Drift (Train vs Test) ===")
        mean_train = feat_train.mean(axis=0)
        mean_test = feat_test.mean(axis=0)
        std_train = feat_train.std(axis=0, ddof=1)
        std_test = feat_test.std(axis=0, ddof=1)

        drift = mean_test - mean_train

        for j in range(self.feature_dim):
            wr(
                f"Feature {j:02d}: "
                f"mean_train={mean_train[j]:.4f}, mean_test={mean_test[j]:.4f}, "
                f"drift={drift[j]:.4f}"
            )

        # drift plot
        plt.figure(figsize=(10, 4))
        plt.plot(drift, "r.-")
        plt.title("Feature Mean Drift (TEST - TRAIN)")
        plt.xlabel("Feature Index")
        plt.ylabel("Drift")
        fig_path = os.path.join(analysis_dir, "FeatureDrift.png")
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
        wr(f"Saved drift plot: {fig_path}\n")

        # ---------------------------------------------------
        # 5) Class-wise centroid comparison heatmap
        # ---------------------------------------------------
        wr("=== Class-wise Centroid Difference ===")
        num_class = self.num_class

        centroid_train = np.zeros((num_class, self.feature_dim), dtype=np.float32)
        centroid_test = np.zeros((num_class, self.feature_dim), dtype=np.float32)

        for c in range(1, num_class + 1):
            centroid_train[c - 1] = feat_train[labels_train == c].mean(axis=0)
            centroid_test[c - 1] = feat_test[labels_test == c].mean(axis=0)

        diff = centroid_test - centroid_train

        plt.figure(figsize=(8, 4))
        im = plt.imshow(diff, aspect="auto", cmap="coolwarm")
        plt.colorbar(im, label="Centroid Test - Train")
        plt.xlabel("Feature Index")
        plt.ylabel("Class")
        plt.yticks(np.arange(num_class), [f"Class {i+1}" for i in range(num_class)])
        plt.title("Centroid Drift Heatmap")
        fig_path = os.path.join(analysis_dir, "CentroidDriftHeatmap.png")
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
        wr(f"Saved centroid drift heatmap: {fig_path}\n")

        # ---------------------------------------------------
        # 6) Mahalanobis Distance (Test → Train)
        # ---------------------------------------------------
        wr("=== Mahalanobis Distance (Test windows to Train distribution) ===")

        # covariance & its pseudo-inverse in TRAIN space
        cov = np.cov(feat_train, rowvar=False)
        cov_inv = np.linalg.pinv(cov)

        def mahalanobis_np(x, mean_vec, inv_cov):
            diff = x - mean_vec
            return float(np.sqrt(diff @ inv_cov @ diff))

        mah_dist = np.array(
            [mahalanobis_np(x, mean_train, cov_inv) for x in feat_test],
            dtype=np.float32,
        )

        wr(f"Mean Mahalanobis distance: {mah_dist.mean():.4f}")
        wr(f"Std Mahalanobis distance:  {mah_dist.std():.4f}\n")

        plt.figure(figsize=(6, 4))
        plt.hist(mah_dist, bins=50, density=True)
        plt.title("Mahalanobis Distance: Test → Train")
        plt.xlabel("Distance")
        plt.ylabel("Density")
        fig_path = os.path.join(analysis_dir, "MahalanobisDistance_Test.png")
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
        wr(f"Saved Mahalanobis histogram: {fig_path}\n")

        wr("********* END OF TRAIN vs TEST COMPARISON *********")
        wr("********************************************************")


    def train_test_confusion(self):
        """
        Build confusion matrices for BOTH:
          - Training windows (using self.featurematrix + LDA weights)
          - Testing windows (using LDA_test over Test/ trials)

        Outputs (in ./Report analysis/):
          - TrainTest_Confusion_Report.txt
          - ConfusionMatrix_Train_Counts.png
          - ConfusionMatrix_Train_Normalized.png
          - ConfusionMatrix_Test_Counts.png
          - ConfusionMatrix_Test_Normalized.png
          - ConfusionMatrix_TrainVsTest_Normalized.png  (side-by-side)
        """
        import os
        from pathlib import Path
        import numpy as np
        import matplotlib.pyplot as plt

        # ------------------------------
        # 0) Basic checks
        # ------------------------------
        if self.train_folder is None:
            raise ValueError(
                "train_folder is not set. Run split_training_testing() and train_model() first."
            )
        if self.testing_folder is None:
            raise ValueError(
                "testing_folder is not set. Run split_training_testing() first."
            )
        if not hasattr(self, "featurematrix") or len(self.featurematrix) == 0:
            raise ValueError(
                "featurematrix is empty. Run train_model() before train_test_confusion()."
            )
        if not hasattr(self, "Wg") or not hasattr(self, "Cg"):
            raise ValueError(
                "Wg/Cg not found. Run train_model() before train_test_confusion()."
            )

        # ------------------------------
        # 1) Setup report folder / file
        # ------------------------------
        cwd = os.getcwd()
        analysis_dir = os.path.join(cwd, "Report analysis")
        os.makedirs(analysis_dir, exist_ok=True)

        report_path = os.path.join(analysis_dir, "TrainTest_Confusion_Report.txt")
        if os.path.exists(report_path):
            os.remove(report_path)

        def wr(line=""):
            with open(report_path, "a", encoding="utf-8") as f:
                f.write(str(line) + "\n")

        wr("********************************************************")
        wr("********  TRAIN vs TEST CONFUSION MATRICES  ************")
        wr("********************************************************")
        wr(f"Train folder: {self.train_folder}")
        wr(f"Test folder:  {self.testing_folder}")
        wr(f"num_class={self.num_class}, feature_dim={self.feature_dim}")
        wr("")

        # =====================================================
        # PART A: TRAINING CONFUSION MATRIX (window-level)
        # =====================================================
        wr("=== TRAINING CONFUSION MATRIX (window-level) ===")

        feat = np.array(self.featurematrix, dtype=np.float32)
        if feat.size % self.feature_dim != 0:
            raise ValueError(
                f"featurematrix length {feat.size} is not divisible by feature_dim {self.feature_dim}"
            )
        num_samples = feat.size // self.feature_dim
        feat = feat.reshape((num_samples, self.feature_dim))

        labels = np.array(self.trainclass, dtype=np.int32)

        # Only use labeled rows 1..num_class
        valid_mask = (labels >= 1) & (labels <= self.num_class)
        feat_valid = feat[valid_mask]
        labels_valid = labels[valid_mask]
        N_valid = feat_valid.shape[0]

        wr(f"Total rows in featurematrix: {num_samples}")
        wr(f"Valid labeled rows (1..{self.num_class}): {N_valid}")
        wr("")

        if N_valid == 0:
            wr("No valid labeled rows in training set; cannot build train confusion matrix.")
        else:
            Wg_vec = np.array(self.Wg, dtype=np.float32)
            Cg_vec = np.array(self.Cg, dtype=np.float32)
            if Wg_vec.size != self.num_class * self.feature_dim:
                raise ValueError(
                    f"Wg size {Wg_vec.size} != num_class * feature_dim "
                    f"({self.num_class * self.feature_dim})"
                )

            # C layout: Wg[i + j*CLASS] -> reshape Fortran-order
            Wg_mat = Wg_vec.reshape((self.num_class, self.feature_dim), order="F")

            # feat_valid is already normalized (after feature_normalization in train_model)
            scores_train = feat_valid @ Wg_mat.T + Cg_vec  # (N_valid, num_class)
            preds_train = np.argmax(scores_train, axis=1) + 1  # 1..num_class

            # Confusion matrix C_train[true, pred]
            C_train = np.zeros((self.num_class, self.num_class), dtype=np.int32)
            for t, p in zip(labels_valid, preds_train):
                C_train[t - 1, p - 1] += 1

            total_train_windows = C_train.sum()
            wr(f"Total valid training windows: {total_train_windows}")
            wr("Confusion matrix (TRAIN, counts; rows=true, cols=pred):")
            for r in range(self.num_class):
                wr(
                    f"True {r+1}: "
                    + "  ".join(f"{C_train[r, c]:5d}" for c in range(self.num_class))
                )
            wr("")

            # Row-normalize
            C_train_norm = C_train.astype(np.float32)
            row_sums = C_train_norm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            C_train_norm /= row_sums

            wr("Row-normalized confusion matrix (TRAIN):")
            for r in range(self.num_class):
                wr(
                    f"True {r+1}: "
                    + "  ".join(f"{C_train_norm[r, c]:0.3f}" for c in range(self.num_class))
                )
            wr("")

            # Save images for TRAIN
            fig_path_train_counts = os.path.join(
                analysis_dir, "ConfusionMatrix_Train_Counts.png"
            )
            plt.figure(figsize=(5, 4))
            im = plt.imshow(C_train, interpolation="nearest")
            plt.title("Training Confusion Matrix (Counts)")
            plt.xlabel("Predicted class")
            plt.ylabel("True class")
            plt.colorbar(im, label="Count")
            for i in range(self.num_class):
                for j in range(self.num_class):
                    plt.text(j, i, str(C_train[i, j]), ha="center", va="center")
            plt.tight_layout()
            plt.savefig(fig_path_train_counts)
            plt.close()
            wr(f"Saved TRAIN confusion (counts): {fig_path_train_counts}")

            fig_path_train_norm = os.path.join(
                analysis_dir, "ConfusionMatrix_Train_Normalized.png"
            )
            plt.figure(figsize=(5, 4))
            im = plt.imshow(C_train_norm, vmin=0.0, vmax=1.0, interpolation="nearest")
            plt.title("Training Confusion Matrix (Row-normalized)")
            plt.xlabel("Predicted class")
            plt.ylabel("True class")
            plt.colorbar(im, label="Fraction")
            for i in range(self.num_class):
                for j in range(self.num_class):
                    plt.text(j, i, f"{C_train_norm[i, j]:.2f}", ha="center", va="center")
            plt.tight_layout()
            plt.savefig(fig_path_train_norm)
            plt.close()
            wr(f"Saved TRAIN confusion (row-normalized): {fig_path_train_norm}")
            wr("")

        # =====================================================
        # PART B: TESTING CONFUSION MATRIX (window-level)
        # =====================================================
        wr("=== TESTING CONFUSION MATRIX (window-level) ===")

        from ctypes import c_float, c_int, POINTER, cast, sizeof

        test_root = Path(self.testing_folder)

        Wg_ptr = cast(self.Wg.buffer_info()[0], POINTER(c_float))
        Cg_ptr = cast(self.Cg.buffer_info()[0], POINTER(c_float))
        xmean_ptr = cast(self.xmean.buffer_info()[0], POINTER(c_float))
        xstd_ptr = cast(self.xstd.buffer_info()[0], POINTER(c_float))

        base_addr = self.testdata.buffer_info()[0]
        elem_size = sizeof(c_float)

        C_test = np.zeros((self.num_class, self.num_class), dtype=np.int32)
        total_test_windows = 0

        # Discover action folders in Test/
        act_dirs = [
            d for d in test_root.iterdir()
            if d.is_dir() and d.name.lower().startswith("act")
        ]
        act_dirs.sort()

        for class_idx, act_dir in enumerate(act_dirs):
            true_label = class_idx + 1
            trial_files = sorted(act_dir.glob("trial_*.txt"))
            if not trial_files:
                continue

            for trial_path in trial_files:
                # Load trial into self.testdata
                self.load_trial_data(trial_path, label="testdata")

                m = 0
                while m < self.win_per_trial:
                    start_float_index = m * self.winc * self.channel
                    end_float_index = start_float_index + self.wl * self.channel
                    if end_float_index > len(self.testdata):
                        break

                    offset_addr = base_addr + start_float_index * elem_size
                    window_ptr = cast(offset_addr, POINTER(c_float))

                    decision = self.lib.LDA_test(
                        window_ptr,
                        Wg_ptr,
                        Cg_ptr,
                        xmean_ptr,
                        xstd_ptr,
                        c_int(self.wl),
                        c_int(self.channel),
                        c_int(self.feature_dim),
                        c_int(self.num_class),
                        c_float(self.deadzone_zc),
                        c_float(self.deadzone_turn),
                        c_int(self.scale_mav),
                        c_int(self.scale_zc),
                        c_int(self.tdfeatureN),
                    )

                    pred_label = int(decision)
                    if 1 <= pred_label <= self.num_class:
                        C_test[true_label - 1, pred_label - 1] += 1
                    total_test_windows += 1
                    m += 1

        wr(f"Total test windows: {total_test_windows}")
        wr("Confusion matrix (TEST, counts; rows=true, cols=pred):")
        for r in range(self.num_class):
            wr(
                f"True {r+1}: "
                + "  ".join(f"{C_test[r, c]:5d}" for c in range(self.num_class))
            )
        wr("")

        # Row-normalize TEST
        C_test_norm = C_test.astype(np.float32)
        row_sums_test = C_test_norm.sum(axis=1, keepdims=True)
        row_sums_test[row_sums_test == 0] = 1.0
        C_test_norm /= row_sums_test

        wr("Row-normalized confusion matrix (TEST):")
        for r in range(self.num_class):
            wr(
                f"True {r+1}: "
                + "  ".join(f"{C_test_norm[r, c]:0.3f}" for c in range(self.num_class))
            )
        wr("")

        # Save images for TEST
        fig_path_test_counts = os.path.join(
            analysis_dir, "ConfusionMatrix_Test_Counts.png"
        )
        plt.figure(figsize=(5, 4))
        im = plt.imshow(C_test, interpolation="nearest")
        plt.title("Testing Confusion Matrix (Counts)")
        plt.xlabel("Predicted class")
        plt.ylabel("True class")
        plt.colorbar(im, label="Count")
        for i in range(self.num_class):
            for j in range(self.num_class):
                plt.text(j, i, str(C_test[i, j]), ha="center", va="center")
        plt.tight_layout()
        plt.savefig(fig_path_test_counts)
        plt.close()
        wr(f"Saved TEST confusion (counts): {fig_path_test_counts}")

        fig_path_test_norm = os.path.join(
            analysis_dir, "ConfusionMatrix_Test_Normalized.png"
        )
        plt.figure(figsize=(5, 4))
        im = plt.imshow(C_test_norm, vmin=0.0, vmax=1.0, interpolation="nearest")
        plt.title("Testing Confusion Matrix (Row-normalized)")
        plt.xlabel("Predicted class")
        plt.ylabel("True class")
        plt.colorbar(im, label="Fraction")
        for i in range(self.num_class):
            for j in range(self.num_class):
                plt.text(j, i, f"{C_test_norm[i, j]:.2f}", ha="center", va="center")
        plt.tight_layout()
        plt.savefig(fig_path_test_norm)
        plt.close()
        wr(f"Saved TEST confusion (row-normalized): {fig_path_test_norm}")
        wr("")

        # =====================================================
        # PART C: SIDE-BY-SIDE COMPARISON (TRAIN vs TEST)
        # =====================================================
        wr("=== SIDE-BY-SIDE: TRAIN vs TEST (Row-normalized) ===")

        fig_path_side = os.path.join(
            analysis_dir, "ConfusionMatrix_TrainVsTest_Normalized.png"
        )
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        im1 = plt.imshow(C_train_norm, vmin=0.0, vmax=1.0, interpolation="nearest")
        plt.title("TRAIN (Row-normalized)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar(im1, fraction=0.046, pad=0.04)

        plt.subplot(1, 2, 2)
        im2 = plt.imshow(C_test_norm, vmin=0.0, vmax=1.0, interpolation="nearest")
        plt.title("TEST (Row-normalized)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar(im2, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(fig_path_side)
        plt.close()

        wr(f"Saved side-by-side TRAIN vs TEST confusion: {fig_path_side}")
        wr("")
        wr("********  END TRAIN vs TEST CONFUSION REPORT  ********")
        wr("********************************************************")

    def confusion_matrix_train_vs_test_trials(self):
        """
        Build a Mahalanobis-distance confusion matrix comparing:
            TEST trial centroids   vs   TRAIN trial centroids

        Output:
            - Heatmap saved to Report analysis folder:
                  TrainTestTrialDistanceMatrix.png
            - Returns (test_names, train_names, dist_matrix)

        Notes:
            • Rows = test trials
            • Columns = training trials
            • Values = Mahalanobis distances
        """
        import ctypes
        import os
        from pathlib import Path

        import numpy as np
        import matplotlib.pyplot as plt

        # ============================================================
        # 1) --- Compute TRAIN trial LDA centroids -------------------
        # ============================================================
        train_centroids = []
        train_names = []

        # Build Wg and Cg as matrix form
        Wg = np.array(self.Wg, dtype=np.float32).reshape(
            (self.num_class, self.feature_dim), order="F"
        )
        Cg = np.array(self.Cg, dtype=np.float32)

        # Normalize featurematrix → Xnorm
        X = np.array(self.featurematrix, dtype=np.float32).reshape(
            (-1, self.feature_dim)
        )
        xmean = np.array(self.xmean, dtype=np.float32)
        xstd = np.array(self.xstd, dtype=np.float32)
        xstd = np.where(xstd == 0, 1, xstd)
        Xnorm = (X - xmean) / xstd

        # LDA projection for training (all windows)
        lda_train = Xnorm @ Wg.T + Cg  # shape (Nwindows, num_class)

        # Get trial centroid ranges from training structure
        for trial_name, (start_idx, end_idx) in self.trial_feature_ranges.items():
            trial_feat = lda_train[start_idx:end_idx]
            centroid = np.mean(trial_feat, axis=0)
            train_centroids.append(centroid)
            train_names.append(trial_name)

        train_centroids = np.array(train_centroids, dtype=np.float32)

        # ============================================================
        # 2) --- Compute TEST trial LDA centroids --------------------
        # ============================================================
        test_centroids = []
        test_names = []

        test_root = Path(self.testing_folder)

        for act_dir in sorted(test_root.iterdir()):
            if not act_dir.is_dir() or not act_dir.name.lower().startswith("act"):
                continue

            class_idx = int(act_dir.name.replace("act", ""))
            trial_files = sorted(act_dir.glob("trial_*.txt"))

            for trial_path in trial_files:
                # Load raw EMG into testdata
                self.load_trial_data(trial_path, label="testdata")

                # Extract windows (same method as training)
                wins = []
                total_floats = self.channel * self.data_per_trial
                for m in range(self.win_per_trial):
                    start = m * self.winc * self.channel
                    end = start + self.wl * self.channel
                    if end > total_floats:
                        break
                    wins.append(self.testdata[start:end])

                # --- extract TD features for all windows (C-based) ---
                fv_list = []
                for win in wins:
                    window_ptr = (ctypes.c_float * len(win))(*win)
                    fv = (ctypes.c_float * self.feature_dim)()
                    self.lib.tdfeats(
                        window_ptr,
                        ctypes.c_int(self.wl),
                        ctypes.c_int(self.channel),
                        ctypes.c_int(0),
                        fv,
                        ctypes.c_float(self.deadzone_zc),
                        ctypes.c_float(self.deadzone_turn),
                        ctypes.c_int(self.scale_mav),
                        ctypes.c_int(self.scale_zc),
                        ctypes.c_int(self.tdfeatureN)
                    )
                    fv_np = np.frombuffer(fv, dtype=np.float32, count=self.feature_dim)
                    fv_list.append(fv_np.copy())

                if len(fv_list) == 0:
                    continue

                Xtest = np.vstack(fv_list).astype(np.float32)

                # normalize using training stats
                Xtest_norm = (Xtest - xmean) / xstd

                # LDA transform
                lda_test = Xtest_norm @ Wg.T + Cg

                # centroid
                centroid = np.mean(lda_test, axis=0)
                test_centroids.append(centroid)
                test_names.append(f"{act_dir.name}/{trial_path.name}")

        test_centroids = np.array(test_centroids, dtype=np.float32)

        # ============================================================
        # 3) Mahalanobis distance test-vs-train
        # ============================================================
        cov = np.atleast_2d(np.cov(train_centroids.T))
        cov_inv = np.linalg.pinv(cov)

        dist_matrix = np.zeros((len(test_centroids), len(train_centroids)), dtype=np.float32)

        for i, tcent in enumerate(test_centroids):
            for j, trcent in enumerate(train_centroids):
                delta = tcent - trcent
                squared_distance = float(delta @ cov_inv @ delta)
                dist_matrix[i, j] = np.sqrt(max(squared_distance, 0.0))

        # ============================================================
        # 4) Heatmap Output
        # ============================================================
        script_dir = os.path.dirname(os.path.abspath(__file__))   # location of the .py file

        out_dir = os.path.join(script_dir, "Report analysis")
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, "TrainTestTrialDistanceMatrix.png")

        plt.figure(figsize=(max(6, len(train_names) * 0.5), max(4, len(test_names) * 0.5 + 2)))
        image = plt.imshow(dist_matrix, aspect="auto", cmap="viridis")
        plt.colorbar(image, label="Mahalanobis distance")
        plt.xticks(range(len(train_names)), train_names, rotation=45, ha="right")
        plt.yticks(range(len(test_names)), test_names)
        for row in range(dist_matrix.shape[0]):
            for col in range(dist_matrix.shape[1]):
                plt.text(col, row, f"{dist_matrix[row, col]:.2f}", ha="center", va="center")
        plt.title("Mahalanobis Distance (TEST trial centroid → TRAIN trial centroid)")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

        print("Saved:", out_path)
        return test_names, train_names, dist_matrix
    
    
