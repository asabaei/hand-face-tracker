import csv
import os
import time
from collections import Counter

import numpy as np

import config
import features

data_dir = "data"


def setup(path):
    global data_dir
    data_dir = path
    os.makedirs(data_dir, exist_ok=True)


def stamp(t):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))


def add_row(filename, header, row):
    path = os.path.join(data_dir, filename)

    # if an older version wrote different columns, move that file aside
    if os.path.exists(path):
        with open(path) as f:
            first_line = f.readline().strip()
        if first_line != ",".join(header):
            os.rename(path, path + ".old-" + str(int(time.time())))

    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(header)
        writer.writerow(row)


def log_event(start, region, behavior, alerted, duration):
    add_row("events.csv", ["start", "region", "behavior", "alerted", "duration_seconds"],
            [stamp(start), region, behavior, "yes" if alerted else "no", round(duration, 2)])


def log_label(start, end):
    # a stretch of time where you really were touching your face (for evaluation)
    add_row("labels.csv", ["start", "end"], [stamp(start), stamp(end)])


def summarize(session_start, session_end, events):
    # events is a list of (region, duration)
    minutes = (session_end - session_start) / 60
    touches = len(events)
    contact = sum(duration for region, duration in events)
    longest = max([duration for region, duration in events], default=0)
    common = Counter(region for region, duration in events).most_common(1)
    top_region = common[0][0] if common else "none"
    return minutes, touches, contact, longest, top_region


def log_session(session_start, session_end, events):
    minutes, touches, contact, longest, top_region = summarize(session_start, session_end, events)
    add_row("sessions.csv",
            ["start", "duration_min", "touches", "contact_seconds", "longest_seconds", "top_region"],
            [stamp(session_start), round(minutes, 2), touches, round(contact, 1),
             round(longest, 1), top_region])


def summary_text(session_start, session_end, events):
    minutes, touches, contact, longest, top_region = summarize(session_start, session_end, events)
    lines = ["", "---- session summary ----",
             "length:        " + str(round(minutes, 1)) + " min",
             "touches:       " + str(touches)]
    if touches > 0:
        lines.append("total contact: " + str(round(contact, 1)) + " sec")
        lines.append("longest:       " + str(round(longest, 1)) + " sec")
        lines.append("most common:   " + top_region)
    return "\n".join(lines)


def save_clip(label, rows, tag=""):
    # saves landmark numbers (never video) as a labeled training clip
    os.makedirs(config.DATASET_DIR, exist_ok=True)
    name = label + "__" + tag + str(int(time.time() * 1000)) + ".csv"
    path = os.path.join(config.DATASET_DIR, name)
    header = "contact,contact_raw," + ",".join(features.FRAME_FEATURE_NAMES)
    np.savetxt(path, np.array(rows), delimiter=",", header=header, comments="")
    return path