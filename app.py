
from flask import Flask, render_template, jsonify, request
import cv2
import mediapipe as mp
import math
import time
import os

app = Flask(__name__)

monitoring = False
blink_count = 0
closed_since = None
current_status = "Monitoring stopped"

face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


def distance(p1, p2):
    return math.dist(p1, p2)


def eye_ratio(points, indexes):
    p = [points[i] for i in indexes]

    vertical1 = distance(p[1], p[5])
    vertical2 = distance(p[2], p[4])
    horizontal = distance(p[0], p[3])

    if horizontal == 0:
        return 0

    return (vertical1 + vertical2) / (2 * horizontal)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/start")
def start():
    global monitoring, blink_count
    global closed_since, current_status

    blink_count = 0
    closed_since = None
    current_status = "Face not detected"
    monitoring = True

    return jsonify({"status": "Monitoring started"})


@app.route("/stop")
def stop():
    global monitoring, closed_since, current_status

    monitoring = False
    closed_since = None
    current_status = "Monitoring stopped"

    return jsonify({"status": "Monitoring stopped"})


@app.route("/process_frame", methods=["POST"])
def process_frame():
    global monitoring, blink_count
    global closed_since, current_status

    if not monitoring:
        return jsonify({"status": "Monitoring stopped"})

    file = request.files.get("frame")

    if file is None:
        return jsonify({"error": "No camera frame received"}), 400

    image_bytes = file.read()
    image