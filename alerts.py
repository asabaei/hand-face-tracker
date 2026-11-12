import subprocess
import sys
import threading
from collections import deque

import config


def play_sound(region):
    if not config.SOUNDS_ENABLED:
        return

    if sys.platform == "darwin":
        name = config.MAC_SOUNDS.get(region, "Ping")
        subprocess.Popen(["afplay", "/System/Library/Sounds/" + name + ".aiff"])
    elif sys.platform.startswith("win"):
        import winsound
        freq = config.WINDOWS_BEEPS.get(region, 800)
        threading.Thread(target=winsound.Beep, args=(freq, 200), daemon=True).start()
    else:
        print("\a", end="", flush=True)


def notify(title, message):
    try:
        if sys.platform == "darwin":
            script = 'display notification "' + message + '" with title "' + title + '"'
            subprocess.Popen(["osascript", "-e", script])
        elif sys.platform.startswith("linux"):
            subprocess.Popen(["notify-send", title, message])
        else:
            print(title + ": " + message)
    except FileNotFoundError:
        print(title + ": " + message)


class AlertManager:
    # plays the sound and handles the "too many touches" notification

    def __init__(self):
        self.count = 0
        self.recent = deque()
        self.last_sound = 0
        self.last_notify = 0

    def fire(self, region, now):
        self.count += 1

        if now - self.last_sound > config.COOLDOWN_SECONDS:
            play_sound(region)
            self.last_sound = now

        self.recent.append(now)
        while self.recent and now - self.recent[0] > config.ESCALATION_WINDOW_MIN * 60:
            self.recent.popleft()

        too_many = len(self.recent) >= config.ESCALATION_COUNT
        if too_many and now - self.last_notify > config.NOTIFICATION_COOLDOWN_MIN * 60:
            message = ("You touched your face " + str(len(self.recent)) +
                       " times in the last " + str(config.ESCALATION_WINDOW_MIN) + " minutes")
            notify("Hand-Face Tracker", message)
            self.last_notify = now