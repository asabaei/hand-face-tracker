# Puts several runs side by side as a table you can paste into the README.
# usage: python compare.py data_nocal data_auto data_manual

import sys

import evaluate


def main():
    folders = sys.argv[1:]
    if not folders:
        print("usage: python compare.py data_nocal data_auto ...")
        return

    print("| Run | Real touches | Precision | Recall | False alarms / hour |")
    print("| --- | --- | --- | --- | --- |")
    for folder in folders:
        r = evaluate.evaluate(folder)
        if r is None:
            print("| " + folder + " | no marked touches | | | |")
            continue
        if r["fa_per_hour"] is None:
            per_hour = "n/a"
        else:
            per_hour = str(round(r["fa_per_hour"], 1))
        print("| " + folder + " | " + str(r["real"]) +
              " | " + str(round(r["precision"] * 100, 1)) + "%" +
              " | " + str(round(r["recall"] * 100, 1)) + "%" +
              " | " + per_hour + " |")


if __name__ == "__main__":
    main()