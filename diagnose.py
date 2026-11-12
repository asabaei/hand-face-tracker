# After a run where you marked touches with G, explains what happened to each one.
# tracker.py calls report() when you quit, and saves the table as diagnosis.csv.

import csv
import os

import config

PAD_SECONDS = 0.5   # small slack, so an event from a neighboring touch is not credited to this one


def report(frame_log, label_intervals, event_log, gating, data_dir):
    # frame_log:  list of (time, hand_seen, contact_without_size_check, contact)
    # event_log:  list of (start, end, alerted)
    if len(frame_log) < 2:
        return

    seconds = frame_log[-1][0] - frame_log[0][0]
    if seconds > 0:
        print("average frame rate: " + str(round((len(frame_log) - 1) / seconds, 1)) + " fps")

    if not label_intervals:
        return

    rows = []
    for number, (start, end) in enumerate(label_intervals, 1):
        inside = [f for f in frame_log if start <= f[0] <= end]
        if not inside:
            continue
        n = len(inside)
        hand_seen = sum(f[1] for f in inside) / n
        raw_frames = sum(f[2] for f in inside)
        contact_frames = sum(f[3] for f in inside)

        # longest unbroken stretch of contact, in seconds
        longest = 0.0
        run_start = None
        for t, hand, raw, contact in inside:
            if contact:
                if run_start is None:
                    run_start = t
                longest = max(longest, t - run_start)
            else:
                run_start = None

        # events during this marked touch
        pad = PAD_SECONDS
        near = [e for e in event_log if e[0] <= end + pad and start - pad <= e[1]]
        alerted = any(e[2] for e in near)

        if alerted:
            reason = "alerted"
        elif near and gating:
            reason = "contact event, but the classifier did not call it a habit"
        elif near:
            reason = "contact event, no alert"
        elif hand_seen < 0.5:
            reason = "hand not detected"
        elif contact_frames == 0 and raw_frames > 0:
            reason = "rejected by the hand-size check"
        elif contact_frames == 0:
            reason = "fingertips never got close enough to the face"
        elif longest < config.PERSISTENCE_SECONDS:
            reason = "contact too brief or flickering"
        else:
            reason = "long enough contact but no event"

        rows.append([number, round(end - start, 1), round(hand_seen * 100), round(contact_frames / n * 100),
                     round(longest, 2), reason])

    print("\n---- what happened to each marked touch ----")
    print("#   length  hand seen  contact  longest run  result")
    for r in rows:
        print(str(r[0]).ljust(4) + (str(r[1]) + "s").ljust(8) + (str(r[2]) + "%").ljust(11) +
              (str(r[3]) + "%").ljust(9) + (str(r[4]) + "s").ljust(13) + r[5])

    counts = {}
    for r in rows:
        counts[r[5]] = counts.get(r[5], 0) + 1
    print("\nsummary:")
    for reason in sorted(counts, key=counts.get, reverse=True):
        print("  " + str(counts[reason]) + "  " + reason)

    path = os.path.join(data_dir, "diagnosis.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["touch", "length_s", "hand_seen_pct", "contact_pct", "longest_run_s", "result"])
        writer.writerows(rows)