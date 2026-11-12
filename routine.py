# The scripted test routine for: python tracker.py --routine
# The tracker shows each step on screen with a countdown. Marked steps are the real
# habit touches, and they are saved as labels automatically (no G presses needed).

ACTION_SECONDS = 3
REST_SECONDS = 3
ROUNDS = 5


def build():
    steps = []
    steps.append({"name": "Warm-up: skin picking on cheek", "marked": True, "seconds": 6})
    steps.append({"name": "Rest", "marked": False, "seconds": REST_SECONDS, "rest": True})

    rotate = ["Nail biting", "Eye rubbing", "Skin picking (cheek/chin)", "Nail biting", "Eye rubbing"]
    alternate = ["Rest chin on hand", "Reach past your face"]

    for r in range(ROUNDS):
        block = [
            ("Nail biting", True),
            ("Hover in front of face (no touch)", False),
            ("Eye rubbing", True),
            ("Sip from a cup", False),
            ("Skin picking (cheek/chin)", True),
            ("Adjust glasses or hair", False),
            (rotate[r], True),
            (alternate[r % 2], False),
        ]
        for name, marked in block:
            steps.append({"name": name, "marked": marked, "seconds": ACTION_SECONDS})
            steps.append({"name": "Rest", "marked": False, "seconds": REST_SECONDS, "rest": True})
    return steps