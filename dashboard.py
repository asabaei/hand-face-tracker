import csv
import os
import sys
from collections import Counter
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import config

data_dir = sys.argv[1] if len(sys.argv) > 1 else config.DATA_DIR
path = os.path.join(data_dir, "events.csv")

if not os.path.exists(path):
    print("No events.csv in " + data_dir + " yet. Run tracker.py first.")
    raise SystemExit

events = []
with open(path) as f:
    for row in csv.DictReader(f):
        # only count touches that actually alerted (harmless contact is logged but not alerted)
        if row.get("alerted", "yes") != "yes":
            continue
        start = datetime.strptime(row["start"], "%Y-%m-%d %H:%M:%S")
        events.append((start, row["region"], float(row["duration_seconds"]),
                       row.get("behavior", "unclassified")))

if not events:
    print("events.csv is empty, nothing to plot yet.")
    raise SystemExit

by_hour = Counter()
count_by_region = Counter()
time_by_region = Counter()
by_behavior = Counter()
for start, region, duration, behavior in events:
    by_behavior[behavior] += 1
    by_hour[start.hour] += 1
    count_by_region[region] += 1
    time_by_region[region] += duration

regions = sorted(count_by_region, key=count_by_region.get, reverse=True)
names = [config.REGION_NAMES.get(r, r) for r in regions]
blue = "#6C9BD2"
coral = "#E07A6F"

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    pass

# add a behavior chart if the classifier was used
use_behavior = any(b != "unclassified" for b in by_behavior)
if use_behavior:
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(18, 4))
else:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))

hours = list(range(24))
ax1.bar(hours, [by_hour[x] for x in hours], color=blue)
ax1.set_title("Touches by hour of day")
ax1.set_xlabel("hour")
ax1.set_ylabel("touches")
ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

ax2.bar(names, [count_by_region[r] for r in regions], color=blue)
ax2.set_title("Touches by region")
ax2.tick_params(axis="x", rotation=30)
ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

ax3.bar(names, [time_by_region[r] for r in regions], color=coral)
ax3.set_title("Total contact time by region (sec)")
ax3.tick_params(axis="x", rotation=30)

if use_behavior:
    kinds = sorted(by_behavior, key=by_behavior.get, reverse=True)
    ax4.bar([config.BEHAVIOR_NAMES.get(k, k) for k in kinds], [by_behavior[k] for k in kinds], color=blue)
    ax4.set_title("Touches by behavior")
    ax4.tick_params(axis="x", rotation=30)
    ax4.yaxis.set_major_locator(MaxNLocator(integer=True))

fig.suptitle("Habit alerts: " + str(len(events)) + " total", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "dashboard.png"), dpi=150)
plt.show()