# Trains the behavior classifier on the clips in dataset/ and compares it
# to the simple rule-based detector (a touch that lasts long enough = habit).

import glob
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

import config
import features


def longest_run(flags):
    best = 0
    run = 0
    for f in flags:
        if f > 0.5:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


dropped = {"count": 0}   # how many windows load_dataset skipped


def load_dataset():
    dropped["count"] = 0

    # windows with no hand near the face are label noise in these classes. when idle is merged
    # into harmless, harmless keeps them (they are a legitimate "not a habit")
    clean_labels = list(config.HABIT_LABELS)
    if not config.MERGE_IDLE_INTO_HARMLESS:
        clean_labels.append("benign")

    X = []
    y = []
    groups = []
    rule_cal = []     # rule-based detector using the calibrated contact flag
    rule_raw = []     # rule-based detector with no calibration
    elsewhere = []    # window: hand touched the face, but never at the mouth or nose
    clips_per_label = {}

    for path in sorted(glob.glob(os.path.join(config.DATASET_DIR, "*.csv"))):
        label = os.path.basename(path).split("__")[0]
        if label not in config.LABELS:
            continue
        if config.MERGE_IDLE_INTO_HARMLESS and label == "none":
            label = "benign"

        data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        # new clips have contact + contact_raw, older clips only have contact
        extra = data.shape[1] - features.N_FRAME
        contact = data[:, 0]
        contact_raw = data[:, 1] if extra == 2 else contact
        frames = data[:, extra:]

        # cut the start and end of the clip (hand moving into position)
        trim = config.TRIM_FRAMES
        if len(frames) > 2 * trim + config.WINDOW_FRAMES:
            contact = contact[trim:-trim]
            contact_raw = contact_raw[trim:-trim]
            frames = frames[trim:-trim]

        clips_per_label[label] = clips_per_label.get(label, 0) + 1

        for start in range(0, len(frames) - config.WINDOW_FRAMES + 1, config.WINDOW_STRIDE):
            end = start + config.WINDOW_FRAMES
            vector = features.window_features(frames[start:end])

            # a window where the hand never came near the face (the hand left, or you paused)
            # is really idle, so inside a habit (or harmless) clip it only adds label noise
            no_contact = vector[-1] == 0     # last feature = fraction of frames near the face
            if config.CLEAN_WINDOWS and label in clean_labels and no_contact:
                dropped["count"] += 1
                continue

            X.append(vector)
            y.append(label)
            groups.append(path)
            rule_cal.append(longest_run(contact[start:end]) >= config.PERSISTENCE_FRAMES)
            rule_raw.append(longest_run(contact_raw[start:end]) >= config.PERSISTENCE_FRAMES)

            # approximate the tracker's region check from the fingertip distances in the window:
            # the hand touched the face, but never at the mouth or just under the nose
            win = frames[start:end]
            touching = (win[:, features.ANY_MIN] < config.NEAR_FRACTION).any()
            at_mouth = ((win[:, features.MOUTH_MIN] < config.MOUTH_RADIUS) |
                        (win[:, features.NOSE_MIN] < config.NOSE_RADIUS)).any()
            elsewhere.append(bool(touching and not at_mouth))

    return (np.array(X), np.array(y), np.array(groups), np.array(rule_cal),
            np.array(rule_raw), np.array(elsewhere), clips_per_label)


def metrics_row(name, true_habit, pred_habit, y, nonhabit):
    row = [name,
           str(round(precision_score(true_habit, pred_habit, zero_division=0) * 100, 1)),
           str(round(recall_score(true_habit, pred_habit, zero_division=0) * 100, 1)),
           str(round(f1_score(true_habit, pred_habit, zero_division=0) * 100, 1))]
    for label in nonhabit:
        mask = y == label
        row.append(str(round(pred_habit[mask].mean() * 100, 1)))
    return row


def print_table(header, rows):
    everything = [header] + rows
    widths = [max(len(r[i]) for r in everything) + 3 for i in range(len(header))]
    for r in everything:
        print("".join(r[i].ljust(widths[i]) for i in range(len(r))))


def save_confusion_matrix(y, pred, labels):
    counts = confusion_matrix(y, pred, labels=labels)
    rates = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    names = [config.BEHAVIOR_NAMES[l] for l in labels]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(rates, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title("Cross-validated confusion matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if rates[i, j] > 0.5 else "black"
            ax.text(j, i, str(counts[i, j]), ha="center", va="center", color=color)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()


def pick_labels(clips):
    # behaviors with at least 2 clips, split into habits and non-habits
    used = [l for l in config.LABELS if clips.get(l, 0) >= 2]
    habits = [l for l in used if l in config.HABIT_LABELS]
    nonhabit = [l for l in used if l not in config.HABIT_LABELS]
    return used, habits, nonhabit


def quick_train():
    # fits on every clip with no cross-validation. used by the tracker's retrain key.
    # returns (bundle or None, message for the screen)
    X, y, groups, rule_cal, rule_raw, elsewhere, clips = load_dataset()
    if len(X) == 0:
        return None, "No examples yet"

    used, habits, nonhabit = pick_labels(clips)
    if not habits or not nonhabit:
        return None, "Need 2+ examples of a habit and of not-a-habit"

    keep = np.isin(y, used)
    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                 random_state=0, n_jobs=1)
    clf.fit(X[keep], y[keep])

    bundle = {"model": clf, "labels": used,
              "window": config.WINDOW_FRAMES, "stride": config.WINDOW_STRIDE}
    joblib.dump(bundle, config.MODEL_PATH)
    return bundle, "Model updated (" + str(len(used)) + " behaviors)"


def main():
    X, y, groups, rule_cal, rule_raw, elsewhere, clips = load_dataset()
    if len(X) == 0:
        print("No clips found in " + config.DATASET_DIR + "/. Run collect.py first.")
        return

    # use the behaviors that have at least 2 clips
    used, habits, nonhabit = pick_labels(clips)
    skipped = [l for l in config.LABELS if l not in used]
    if skipped:
        print("Skipping (need at least 2 clips): " + ", ".join(skipped))
    if not habits or not nonhabit:
        print("Need at least one habit behavior and one non-habit behavior with 2+ clips.")
        print("clips so far:", clips)
        return

    keep = np.isin(y, used)
    X, y, groups = X[keep], y[keep], groups[keep]
    rule_cal, rule_raw = rule_cal[keep], rule_raw[keep]
    elsewhere = elsewhere[keep]

    print("clips:", clips)
    print("windows:", len(X), " features per window:", X.shape[1])
    if config.CLEAN_WINDOWS:
        print("dropped " + str(dropped["count"]) + " windows where the hand never came near the face "
              "(from clips labeled as habits" + ("" if config.MERGE_IDLE_INTO_HARMLESS else " or harmless") + ")")

    clf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                 random_state=0, n_jobs=-1)

    # cross-validate by clip so windows from the same clip never land in both train and test
    n_splits = min(5, min(clips[l] for l in used))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
    proba = cross_val_predict(clf, X, y, groups=groups, cv=cv, method="predict_proba")
    class_list = np.unique(y)          # column order of predict_proba
    pred = class_list[proba.argmax(axis=1)]

    # the same predictions with the mouth rule: no nail biting when touching somewhere else
    pred_rule = pred
    if "nail_biting" in class_list:
        adjusted = proba.copy()
        adjusted[elsewhere, list(class_list).index("nail_biting")] = 0.0
        pred_rule = class_list[adjusted.argmax(axis=1)]

    print("\n---- cross-validation (" + str(n_splits) + " folds, split by clip) ----")
    print("accuracy:  " + str(round(accuracy_score(y, pred) * 100, 1)) + "%")
    print("macro F1:  " + str(round(f1_score(y, pred, average="macro") * 100, 1)) + "%\n")
    print(classification_report(y, pred, labels=used, zero_division=0))

    # habit vs not-habit, classifier vs the rule-based detector
    true_habit = np.isin(y, config.HABIT_LABELS)
    clf_habit = np.isin(pred, config.HABIT_LABELS)
    print("---- habit vs not habit (percent) ----")
    print("last columns = how often each non-habit behavior was wrongly flagged as a habit\n")
    header = ["method", "precision", "recall", "F1"] + [config.BEHAVIOR_NAMES[l] for l in nonhabit]
    rows = [metrics_row("rule-based", true_habit, rule_raw, y, nonhabit),
            metrics_row("rule + calibration", true_habit, rule_cal, y, nonhabit),
            metrics_row("classifier", true_habit, clf_habit, y, nonhabit),
            metrics_row("classifier + mouth rule", true_habit, np.isin(pred_rule, config.HABIT_LABELS), y, nonhabit)]
    print_table(header, rows)

    if "nail_biting" in class_list:
        nb = y == "nail_biting"
        hit = (pred[nb] == "nail_biting").mean() * 100
        hit_rule = (pred_rule[nb] == "nail_biting").mean() * 100
        fake = ((pred == "nail_biting") & (y != "nail_biting")).sum()
        fake_rule = ((pred_rule == "nail_biting") & (y != "nail_biting")).sum()
        print("\nnail biting recall: " + str(round(hit, 1)) + "% -> " + str(round(hit_rule, 1)) +
              "% with the mouth rule;  windows wrongly called nail biting: " + str(fake) + " -> " + str(fake_rule))

    save_confusion_matrix(y, pred, used)
    print("\nsaved confusion_matrix.png")

    # train on everything and save
    clf.fit(X, y)
    joblib.dump({"model": clf, "labels": used,
                 "window": config.WINDOW_FRAMES, "stride": config.WINDOW_STRIDE},
                config.MODEL_PATH)
    print("saved " + config.MODEL_PATH)

    order = np.argsort(clf.feature_importances_)[::-1][:10]
    print("\ntop features:")
    for i in order:
        print("  " + features.WINDOW_FEATURE_NAMES[i] + "  " + str(round(clf.feature_importances_[i], 3)))


if __name__ == "__main__":
    main()