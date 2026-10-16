#!/usr/bin/env python3
"""
train_model.py - train the small 1D CNN on the capacitor waveforms (Mac only).

Input   data/processed/dataset.npz    from prep/prepare_dataset.py
        data/split/split.json         train/test capacitors (must already exist)
Output  models/cap_cnn.keras          trained FP32 model -> converted to .tflite on Saturday
        models/train_config.json      validation capacitors, seed, settings, versions
        results/training_history.csv  loss and accuracy per epoch
        results/train_metrics.json    train / validation / test results
        results/learning_curve.png    training vs validation loss

Run from the repo root, with the Mac venv active:
    python train/train_model.py
    python train/train_model.py --help      # list the settings you can change

The 6 training capacitors are split again: 4 to learn from, 2 (one healthy,
one degraded) for validation. Validation decides when to stop training.
The 2 test capacitors are only looked at once, at the very end.
"""

# =============================================================================
# IMPORTS
# Standard library first, then third-party libraries.
# =============================================================================
import argparse          # reads command-line options such as --epochs 50
import json              # reads split.json and writes the result files
import os                # environment variables (used to quieten TensorFlow)
import platform          # Python version, recorded for reproducibility
import sys               # sys.exit() stops the script with a clear message
from pathlib import Path # file paths that work on macOS, Linux and Windows

import numpy as np       # arrays and maths on whole arrays

# TensorFlow prints many INFO lines when it starts. Level 2 hides most INFO
# and WARNING messages but keeps real errors; a few start-up lines may still
# appear and are harmless. This must be set BEFORE TensorFlow is imported,
# which is why the TensorFlow imports come after this line.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402  (the backend that does the maths)
import keras             # noqa: E402  (the high-level API to build and train models)
from keras import layers # noqa: E402  (Conv1D, Dense, ... building blocks)


# =============================================================================
# HELPER FUNCTIONS
# Small, single-purpose functions used by main() further down.
# =============================================================================

def pick_val_caps(caps: np.ndarray, y: np.ndarray, seed: int) -> list:
    """Choose the validation capacitors: one healthy and one degraded.

    Why: training needs a set of data it does NOT learn from, to decide when
    to stop. The test set cannot be used for that (that would leak the test
    into training), so 2 of the 6 training capacitors are set aside instead.
    They are whole capacitors, not random waveforms, for the same reason the
    test split is by capacitor: waveforms of one capacitor look almost alike.

    caps : capacitor name of every training waveform, e.g. "ES14C1"
    y    : label of every training waveform (0 = healthy, 1 = degraded)
    seed : random seed, so the same capacitors are picked every run
    Returns a sorted list of two capacitor names.
    """
    names_ = sorted(set(caps))

    # Fraction of degraded waveforms per capacitor. y[caps == c] keeps only
    # that capacitor's labels; the mean of 0/1 labels is the fraction of 1s.
    frac = {c: float(y[caps == c].mean()) for c in names_}

    # A capacitor counts as "healthy" if most of its waveforms are healthy.
    healthy = [c for c in names_ if frac[c] < 0.5]
    degraded = [c for c in names_ if frac[c] >= 0.5]

    # At least one of each class must remain for training after the hold-out.
    if len(healthy) < 2 or len(degraded) < 2:
        sys.exit("Need at least 2 healthy and 2 degraded training capacitors "
                 "to hold one of each out for validation.")

    # Seeded random choice: random, but repeatable.
    rng = np.random.default_rng(seed)
    return sorted([str(rng.choice(healthy)), str(rng.choice(degraded))])


def names(caps: np.ndarray) -> list:
    """Return the unique capacitor names as a sorted list of plain strings.

    NumPy 2 prints its own string type as np.str_('ES14C1'); converting with
    str() gives clean output on screen and in the JSON files.
    """
    return sorted({str(c) for c in caps})


def build_model(window: int) -> keras.Model:
    """Build the 1D CNN. It has 721 trainable parameters.

    Data flows top to bottom. Shapes are for one waveform of 190 samples:

      Input              (190, 1)   190 time steps, 1 channel (one voltage)
      Conv1D 8 x 5       (186, 8)   8 learned 5-tap filters slide along the
                                    waveform, like a bank of FIR filters.
                                    Params: 5 taps x 1 channel x 8 + 8 bias = 48
      MaxPooling1D 2     (93, 8)    keep the larger of every 2 values: halves
                                    the length, like decimation + peak detect
      Conv1D 16 x 5      (89, 16)   16 filters that combine the first 8 into
                                    longer shape detectors.
                                    Params: 5 x 8 x 16 + 16 bias = 656
      GlobalAvgPool1D    (16,)      average each filter output over time:
                                    16 numbers that describe the curve shape
      Dense 1, sigmoid   (1,)       weighted sum of the 16 numbers, squeezed
                                    to 0..1 = probability "degraded".
                                    Params: 16 weights + 1 bias = 17

    ReLU (activation="relu") sets negative values to 0. Without such a
    non-linear step, stacked layers would collapse into one linear filter.
    """
    return keras.Sequential([
        layers.Input(shape=(window, 1)),
        layers.Conv1D(8, kernel_size=5, activation="relu"),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(16, kernel_size=5, activation="relu"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(1, activation="sigmoid"),
    ], name="cap_cnn")


def evaluate(name: str, model: keras.Model, X: np.ndarray, y: np.ndarray,
             caps: np.ndarray) -> dict:
    """Score the model on one data set and print a short report.

    Reports:
      accuracy        fraction of waveforms classified correctly
      baseline        accuracy of always guessing the more common class;
                      the model is only useful if it beats this
      confusion       the four counts of right and wrong answers
      per capacitor   accuracy for each capacitor, to see WHERE errors are

    Returns the same numbers as a dict, saved later to train_metrics.json.
    """
    # The model outputs a probability between 0 and 1 for each waveform.
    # .ravel() turns shape (N, 1) into a flat array of N numbers.
    prob = model.predict(X, verbose=0).ravel()

    # Decision threshold: 0.5 or more means "degraded" (1), else "healthy" (0).
    pred = (prob >= 0.5).astype(int)

    # Confusion matrix, counted with boolean masks:
    #   tp = degraded, predicted degraded    (correct)
    #   tn = healthy,  predicted healthy     (correct)
    #   fp = healthy,  predicted degraded    (false alarm)
    #   fn = degraded, predicted healthy     (missed fault)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    acc = (tp + tn) / len(y)

    # y.mean() = fraction degraded. Always guessing the bigger class scores
    # max(fraction degraded, fraction healthy).
    baseline = max(y.mean(), 1 - y.mean())

    # Accuracy per capacitor: shows whether errors sit on one capacitor
    # (e.g. near the 80 % capacitance line) or are spread everywhere.
    per_cap = {}
    for c in names(caps):
        m = caps == c                                  # rows of this capacitor
        per_cap[c] = {"n": int(m.sum()),
                      "frac_degraded": round(float(y[m].mean()), 3),
                      "accuracy": round(float((pred[m] == y[m]).mean()), 4)}

    # ---- printed report ----
    print(f"\n{name}: {len(y)} waveforms, accuracy {acc:.2%} "
          f"(guess-the-bigger-class baseline {baseline:.2%})")
    print("                    predicted healthy   predicted degraded")
    print(f"  actual healthy    {tn:>17}   {fp:>18}")
    print(f"  actual degraded   {fn:>17}   {tp:>18}")
    for c, v in per_cap.items():
        print(f"  {c}: {v['accuracy']:.2%} correct of {v['n']} "
              f"(label fraction degraded {v['frac_degraded']})")

    return {"n": len(y), "accuracy": round(acc, 4), "baseline": round(float(baseline), 4),
            "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "per_capacitor": per_cap}


# =============================================================================
# MAIN: the training procedure, step by step
# =============================================================================

def main():
    # -------------------------------------------------------------------------
    # 0. Settings from the command line
    #    Defaults are used unless you pass an option, e.g.
    #    python train/train_model.py --epochs 50 --lr 0.0005
    # -------------------------------------------------------------------------
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/processed/dataset.npz")
    ap.add_argument("--split", default="data/split/split.json")
    ap.add_argument("--seed", type=int, default=42,
                    help="random seed: same seed -> same model")
    ap.add_argument("--epochs", type=int, default=100,
                    help="upper limit; early stopping usually ends sooner")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="waveforms per weight update")
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="learning rate: size of each weight correction")
    ap.add_argument("--patience", type=int, default=10,
                    help="epochs without validation improvement before stopping")
    args = ap.parse_args()

    # -------------------------------------------------------------------------
    # 1. Reproducibility
    #    Training starts from random weights and shuffles the data randomly.
    #    Fixing the seed makes both repeatable, so the same command gives the
    #    same model. Op determinism also fixes the order of some calculations;
    #    if a TensorFlow version does not support it, training still runs.
    # -------------------------------------------------------------------------
    keras.utils.set_random_seed(args.seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    # -------------------------------------------------------------------------
    # 2. Load the prepared data
    #    dataset.npz holds the arrays made by prep/prepare_dataset.py:
    #      X_train (1800, 190)  scaled waveforms of the 6 training capacitors
    #      y_train (1800,)      labels, 0 = healthy, 1 = degraded
    #      X_test, y_test       the 2 held-out test capacitors
    #      caps_train/test      capacitor name of every row
    #    allow_pickle=True is needed for the name arrays (stored as Python
    #    strings). Safe here because dataset.npz is your own file.
    # -------------------------------------------------------------------------
    if not Path(args.data).exists():
        sys.exit(f"{args.data} not found - run prep/prepare_dataset.py first.")
    d = np.load(args.data, allow_pickle=True)
    X_all, y_all = d["X_train"], d["y_train"].astype(np.int64)
    caps_all = d["caps_train"].astype(str)
    X_test, y_test = d["X_test"], d["y_test"].astype(np.int64)
    caps_test = d["caps_test"].astype(str)

    # Safety check: the test capacitors in the data must be the ones fixed in
    # split.json. If not, dataset.npz is out of date.
    split = json.loads(Path(args.split).read_text())
    if names(caps_test) != sorted(split["test_capacitors"]):
        sys.exit("dataset.npz does not match split.json - re-run prep/prepare_dataset.py.")
    window = X_all.shape[1]          # samples per waveform (190)

    # -------------------------------------------------------------------------
    # 3. Validation split, by capacitor
    #    is_val is a boolean mask: True for rows of the 2 validation
    #    capacitors. "~" flips it, so ~is_val selects the 4 training ones.
    # -------------------------------------------------------------------------
    val_caps = pick_val_caps(caps_all, y_all, args.seed)
    is_val = np.isin(caps_all, val_caps)
    X_tr, y_tr, caps_tr = X_all[~is_val], y_all[~is_val], caps_all[~is_val]
    X_val, y_val, caps_val = X_all[is_val], y_all[is_val], caps_all[is_val]
    print(f"train capacitors:      {names(caps_tr)}  "
          f"({len(y_tr)} waveforms, {int(y_tr.sum())} degraded)")
    print(f"validation capacitors: {val_caps}  ({len(y_val)} waveforms, {int(y_val.sum())} degraded)")
    print(f"test capacitors:       {names(caps_test)}  (not used until the end)")

    # Conv1D expects 3 dimensions: (samples, time steps, channels).
    # [..., None] adds a channel axis of size 1: (1200, 190) -> (1200, 190, 1).
    # Like a mono audio track: one channel.
    X_tr, X_val, X_test = X_tr[..., None], X_val[..., None], X_test[..., None]

    # -------------------------------------------------------------------------
    # 4. Model, loss and optimizer
    #    loss      = binary cross-entropy: the penalty for wrong probabilities.
    #                Confident mistakes cost far more than unsure ones.
    #    optimizer = Adam: decides how to change each weight to reduce the
    #                loss. learning_rate sets the step size (like loop gain:
    #                too high overshoots, too low creeps).
    #    metrics   = accuracy: printed each epoch for you; not used to train.
    # -------------------------------------------------------------------------
    model = build_model(window)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=args.lr),
                  loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()                  # prints layers, shapes and 721 parameters

    # -------------------------------------------------------------------------
    # 5. Train
    #    One epoch = every training waveform seen once, in batches of 32.
    #    For each batch: predict -> compute loss -> backpropagation finds how
    #    each weight contributed -> Adam nudges every weight a little.
    #    After each epoch, Keras measures the loss on the validation set.
    #
    #    EarlyStopping watches that validation loss. If it has not improved
    #    for `patience` epochs, training stops and the weights from the best
    #    epoch are restored. This prevents overfitting: the point where the
    #    model starts memorising the 4 training capacitors instead of
    #    learning what degradation looks like.
    # -------------------------------------------------------------------------
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience,
                                         restore_best_weights=True, verbose=1)
    hist = model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                     epochs=args.epochs, batch_size=args.batch_size,
                     shuffle=True,             # new order each epoch
                     callbacks=[stop], verbose=2)

    # hist.history is a dict of lists, one value per epoch:
    #   loss, accuracy, val_loss, val_accuracy
    h = hist.history
    best_epoch = int(np.argmin(h["val_loss"])) + 1   # +1: epochs count from 1
    print(f"\nran {len(h['loss'])} epochs; best validation loss at epoch {best_epoch} "
          f"(weights from that epoch are kept)")

    # -------------------------------------------------------------------------
    # 6. Evaluate
    #    Train and validation first. The test set is evaluated ONCE, here at
    #    the end. Do not change the model because of the test result, or the
    #    test set quietly becomes a second validation set.
    # -------------------------------------------------------------------------
    results = {
        "train": evaluate("TRAIN", model, X_tr, y_tr, caps_tr),
        "validation": evaluate("VALIDATION", model, X_val, y_val, caps_val),
        "test": evaluate("TEST (held-out capacitors)", model, X_test, y_test, caps_test),
    }

    # -------------------------------------------------------------------------
    # 7. Save everything
    #    - the trained model (input for the .tflite conversion)
    #    - the settings and library versions, so the run can be repeated
    #    - the metrics and the per-epoch history, for the README
    #    - the learning curve plot
    # -------------------------------------------------------------------------
    Path("models").mkdir(exist_ok=True)       # create the folder if missing
    Path("results").mkdir(exist_ok=True)
    model.save("models/cap_cnn.keras")        # architecture + weights in one file

    config = {
        "seed": args.seed, "epochs_max": args.epochs, "epochs_run": len(h["loss"]),
        "best_epoch": best_epoch, "batch_size": args.batch_size, "learning_rate": args.lr,
        "patience": args.patience, "window_samples": window,
        "train_capacitors": names(caps_tr), "validation_capacitors": val_caps,
        "test_capacitors": names(caps_test), "parameters": int(model.count_params()),
        "versions": {"python": platform.python_version(), "tensorflow": tf.__version__,
                     "keras": keras.__version__, "numpy": np.__version__},
    }
    Path("models/train_config.json").write_text(json.dumps(config, indent=2) + "\n")
    Path("results/train_metrics.json").write_text(json.dumps(results, indent=2) + "\n")

    # Per-epoch history as CSV: one row per epoch, easy to open in a spreadsheet.
    with open("results/training_history.csv", "w") as f:
        f.write("epoch,loss,accuracy,val_loss,val_accuracy\n")
        for i in range(len(h["loss"])):
            f.write(f"{i + 1},{h['loss'][i]:.6f},{h['accuracy'][i]:.6f},"
                    f"{h['val_loss'][i]:.6f},{h['val_accuracy'][i]:.6f}\n")

    # Learning curve: training vs validation loss per epoch, best epoch marked.
    # Healthy picture: both fall, then validation flattens. If validation
    # rises while training keeps falling, the model is overfitting.
    # The "Agg" backend writes the image to a file without opening a window.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ep = range(1, len(h["loss"]) + 1)
        fig, ax = plt.subplots(figsize=(7, 4))   # one figure, one plot area
        ax.plot(ep, h["loss"], label="training loss")
        ax.plot(ep, h["val_loss"], label="validation loss")
        ax.axvline(best_epoch, linestyle="--", linewidth=1, color="grey",
                   label=f"best epoch ({best_epoch})")
        ax.set_xlabel("epoch")
        ax.set_ylabel("binary cross-entropy loss")
        ax.set_title("Learning curve")
        ax.legend()
        fig.tight_layout()
        fig.savefig("results/learning_curve.png", dpi=150)
        print("wrote results/learning_curve.png")
    except ImportError:
        print("matplotlib not installed - skipped the learning-curve plot")

    print("wrote models/cap_cnn.keras, models/train_config.json,")
    print("      results/train_metrics.json, results/training_history.csv")


# Runs main() only when the file is started as a script
# (python train/train_model.py), not when another file imports it.
if __name__ == "__main__":
    main()