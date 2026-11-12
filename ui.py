import os

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config


mp_hands = mp.solutions.hands


def load_font(size):
    paths = [
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNS.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def draw_hands(frame, hands_points, mode):
    overlay = frame.copy()
    color = {
        "alert": config.ALERT_COLOR,
        "snoozed": config.SNOOZE_COLOR,
    }.get(mode, config.LINE_COLOR)

    for hand_points in hands_points:
        for start, end in mp_hands.HAND_CONNECTIONS:
            a = (int(hand_points[start][0]), int(hand_points[start][1]))
            b = (int(hand_points[end][0]), int(hand_points[end][1]))
            cv2.line(overlay, a, b, color, 2, cv2.LINE_AA)

        for index, point in enumerate(hand_points):
            center = (int(point[0]), int(point[1]))
            if index in (4, 8, 12, 16, 20):
                cv2.circle(overlay, center, 6, color, 2, cv2.LINE_AA)
                cv2.circle(overlay, center, 3, config.JOINT_COLOR, -1, cv2.LINE_AA)
            else:
                cv2.circle(overlay, center, 3, config.JOINT_COLOR, -1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)


def draw_ui(frame, status_text, mode, lines, hint_text, badge_text,
            title_font, small_font):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    status_color = {
        "alert": (235, 110, 110, 255),
        "snoozed": (170, 170, 170, 255),
    }.get(mode, (150, 200, 235, 255))
    content = [line for line in lines if line]
    panel_lines = content + ([badge_text] if badge_text else [])
    text_width = max(
        [draw.textlength(status_text, font=title_font)]
        + [draw.textlength(line, font=small_font) for line in panel_lines]
        + [0]
    )
    panel_width = min(int(text_width) + 78, image.width - 40)
    panel_height = 54 + len(panel_lines) * 24
    draw.rounded_rectangle(
        (20, 20, 20 + panel_width, 20 + panel_height),
        radius=14,
        fill=(18, 18, 22, 185),
    )
    draw.ellipse((36, 39, 48, 51), fill=status_color)
    draw.text((58, 45), status_text, font=title_font,
              fill=(245, 245, 245, 255), anchor="lm")

    y = 70
    for line in content:
        draw.text((36, y), line, font=small_font,
                  fill=(200, 200, 208, 255), anchor="lm")
        y += 24
    if badge_text:
        draw.text((36, y), badge_text, font=small_font,
                  fill=(245, 220, 150, 255), anchor="lm")

    if hint_text:
        hint_width = min(int(draw.textlength(hint_text, font=small_font)) + 32,
                         image.width - 40)
        y = image.height - 56
        draw.rounded_rectangle(
            (20, y, 20 + hint_width, y + 36),
            radius=18,
            fill=(18, 18, 22, 185),
        )
        draw.text((36, y + 18), hint_text, font=small_font,
                  fill=(215, 215, 220, 255), anchor="lm")

    composited = Image.alpha_composite(image, layer)
    return cv2.cvtColor(np.array(composited.convert("RGB")), cv2.COLOR_RGB2BGR)
