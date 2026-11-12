# Compares detected touches (events.csv) to the touches you marked with G (labels.csv).
# usage: python evaluate.py            (uses the data folder)
#        python evaluate.py data_auto

import csv
import os
import sys
from datetime import datetime

import config

PAD_SECONDS = 2   # slack around each marked touch, since you press G by hand
FMT = "%Y-%m-%d %H:%M:%S"


def read_rows(data_dir, name):
    path = os.path.join(data_dir, name)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def to_seconds(text):
    return datetime.strptime(text, FMT).timestamp()


def overlaps(a, b):
    return a[0] <= b[1] and b[0] <= a[1]


def count_matches(events, labels):
    # Best possible one-to-one matching between alerts and marked touches.
    # One alert can only explain ONE marked touch, even if it overlaps several of them.
    owner = [None] * len(labels)    # owner[i] = the event matched to marked touch i

    def try_event(e, seen):
        for i in range(len(labels)):
            if i not in seen and overlaps(events[e], labels[i]):
                seen.add(i)
                if owner[i] is None or try_event(owner[i], seen):
                    owner[i] = e
                    return True
        return False

    for e in range(len(events)):
        try_event(e, set())
    return sum(1 for o in owner if o is not None)


def evaluate(data_dir):
    # returns a dict of results, or None if there are no marked touches
    # only touches that triggered an alert count as detections
    # (with the classifier on, harmless contact is logged but does not alert)
    events = []
    for row in read_rows(data_dir, "events.csv"):
        if row.get("alerted", "yes") != "yes":
            continue
        start = to_seconds(row["start"])
        events.append((start, start + float(row["duration_seconds"])))

    labels = []
    for row in read_rows(data_dir, "labels.csv"):
        labels.append((to_seconds(row["start"]) - PAD_SECONDS, to_seconds(row["end"]) + PAD_SECONDS))

    if not labels:
        return None

    minutes = sum(float(row["duration_min"]) for row in read_rows(data_dir, "sessions.csv"))

    # each alert can match at most one marked touch, and each marked touch at most one alert
    matched = count_matches(events, labels)
    true_events = matched
    false_events = len(events) - matched
    found_labels = matched

    hours = minutes / 60
    return {
        "folder": data_dir,
        "minutes": minutes,
        "real": len(labels),
        "detected": len(events),
        "correct": true_events,
        "false": false_events,
        "missed": len(labels) - found_labels,
        "precision": true_events / len(events) if events else 0,
        "recall": found_labels / len(labels),
        "fa_per_hour": false_events / hours if hours > 0 else None,
    }


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else config.DATA_DIR
    r = evaluate(data_dir)
    if r is None:
        print("No labels found. While running tracker.py, press G, do a real touch, press G again.")
        return

    print("folder:             " + r["folder"])
    print("session time:       " + str(round(r["minutes"], 1)) + " min")
    print("real touches:       " + str(r["real"]))
    print("detected events:    " + str(r["detected"]))
    print("correct detections: " + str(r["correct"]))
    print("false alarms:       " + str(r["false"]))
    print("missed touches:     " + str(r["missed"]))
    print("precision:          " + str(round(r["precision"] * 100, 1)) + "%")
    print("recall:             " + str(round(r["recall"] * 100, 1)) + "%")
    if r["fa_per_hour"] is not None:
        print("false alarms/hour:  " + str(round(r["fa_per_hour"], 1)))


if __name__ == "__main__":
    main()