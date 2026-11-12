# All the settings for the tracker live here.

# ---- camera ----
CAMERA_ORDER = [1, 0]     # tries camera 1 first (MacBook when an iPhone is nearby), then 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ---- detection ----
PERSISTENCE_SECONDS = 0.5     # a touch must last this long before it counts
GRACE_SECONDS = 0.2           # a gap this short does not end a touch
PERSISTENCE_FRAMES = 15       # only used by train.py's offline baseline (about 0.5 sec at 30 fps)
NEAR_FRACTION = 0.09          # a finger must be this close to a face point (fraction of face width)
EYE_RADIUS = 0.08             # how close to the eyes/eyebrows counts as "eyes"
NOSE_RADIUS = 0.08
MOUTH_RADIUS = 0.08
MAX_HAND_RATIO = 1.1          # before calibration: ignore hands that look too big
CALIBRATION_TOLERANCE = 1.2   # after calibration: hand can be this much bigger than baseline
FACE_LOST_FRAMES = 10         # keep using the last face position this long if a hand hides the face

# ---- alerts ----
SOUNDS_ENABLED = True
COOLDOWN_SECONDS = 5          # min seconds between sounds
MAC_SOUNDS = {                # a different sound for each region
    "eyes": "Tink",
    "nose": "Pop",
    "mouth": "Ping",
    "forehead": "Purr",
    "cheek_chin": "Bottle",
}
WINDOWS_BEEPS = {             # frequencies in Hz
    "eyes": 1000,
    "nose": 800,
    "mouth": 600,
    "forehead": 1200,
    "cheek_chin": 700,
}

# ---- escalation (desktop notification if it keeps happening) ----
ESCALATION_COUNT = 3              # this many touches...
ESCALATION_WINDOW_MIN = 10        # ...within this many minutes
NOTIFICATION_COOLDOWN_MIN = 10    # don't send notifications more often than this

# ---- snooze ----
SNOOZE_MINUTES = 10
ALERT_DISPLAY_SECONDS = 2.0   # keep an alert on screen at least this long (display only)

# ---- files ----
DATA_DIR = "data"

# ---- look ----
REGION_NAMES = {
    "eyes": "Eyes",
    "nose": "Nose",
    "mouth": "Mouth",
    "forehead": "Forehead",
    "cheek_chin": "Cheek / chin",
}
LINE_COLOR = (235, 200, 150)      # soft blue (BGR)
ALERT_COLOR = (110, 110, 235)     # soft coral
SNOOZE_COLOR = (170, 170, 170)    # gray
JOINT_COLOR = (255, 255, 255)

# ---- automatic calibration ----
AUTO_CALIBRATE = True         # learn the hand-size baseline from your own touches
AUTOCAL_TIGHT_FRACTION = 0.04 # a fingertip this close to the face counts as a sure touch
AUTOCAL_HOLD_FRAMES = 20      # a sure touch must last this long before it is used
AUTOCAL_MIN_SAMPLES = 60      # samples needed before the baseline is set
AUTOCAL_MAX_SAMPLES = 600     # keeps the most recent samples (running median)
AUTOCAL_OUTLIER = 0.3         # ignore samples more than 30% away from the current baseline

# ---- behavior classifier ----
LABELS = ["none", "hover", "benign", "nail_biting", "skin_picking", "eye_rubbing"]
HABIT_LABELS = ["nail_biting", "skin_picking", "eye_rubbing"]
BEHAVIOR_NAMES = {
    "none": "Idle",
    "hover": "Hovering (no touch)",
    "benign": "Harmless / idle",
    "nail_biting": "Nail biting",
    "skin_picking": "Skin picking",
    "eye_rubbing": "Eye rubbing",
}
DATASET_DIR = "dataset"
MODEL_PATH = "model.joblib"
COUNTDOWN_SECONDS = 3         # collect.py: time to get your hand in position
RECORD_SECONDS = 15           # collect.py: length of each clip
WINDOW_FRAMES = 30            # frames the classifier looks at (about 1 sec)
WINDOW_STRIDE = 10            # train.py: step between training windows
CLASSIFY_EVERY = 5            # tracker.py: re-check the last window every this many frames
TRIM_FRAMES = 15              # frames cut from the start/end of each clip when training
CLASSIFIER_THRESHOLD = 0.6    # habit probability needed to count as a habit
CLASSIFIER_CONFIRM = 2        # this many windows in a row before alerting
NAIL_BITING_NEEDS_MOUTH = True  # only allow "nail biting" when the contact is at the mouth (or just under the nose)

# ---- learning from corrections (tracker.py) ----
FEEDBACK_FRAMES = 90          # how much recent footage a correction key saves (about 3 sec)
FEEDBACK_KEYS = {             # key -> what the last few seconds really were
    "x": "benign",            # X = that was NOT a habit
    "b": "nail_biting",       # B = that was nail biting
    "p": "skin_picking",      # P = that was skin picking
    "e": "eye_rubbing",       # E = that was eye rubbing
}

# ---- training clean-up ----
CLEAN_WINDOWS = True          # drop windows from harmless/habit clips where the hand never comes near the face
MERGE_IDLE_INTO_HARMLESS = True   # train idle and harmless contact as one "not a habit" class