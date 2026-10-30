
from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import math
import time

app = Flask(__name__)

monitoring = False
blink_count = 0
closed_since = None

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


def generate_frames():
    global monitoring, blink_count, closed_since, current_status

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        monitoring = False
        camera.release()
        return

    try:
        while monitoring:
            success, frame = camera.read()

            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            status = "Face not detected"

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                height, width = frame.shape[:2]

                points = [
                    (int(point.x * width), int(point.y * height))
                    for point in landmarks
                ]

                left = eye_ratio(points, LEFT_EYE)
                right = eye_ratio(points, RIGHT_EYE)
                ear = (left + right) / 2
                now = time.time()

                if ear < 0.22:
                    status = "Eyes Closed"

                    if closed_since is None:
                        closed_since = now
                    elif now - closed_since >= 2:
                        status = "DROWSINESS ALERT!"

                else:
                    if closed_since is not None:
                        duration = now - closed_since

                        if 0.1 < duration < 1.0:
                            blink_count += 1

                    closed_since = None
                    status = "Eyes Open"

                color = (0, 0, 255) if "ALERT" in status else (0, 255, 0)

                cv2.putText(
                    frame, status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
                )

                cv2.putText(
                    frame, f"Blinks: {blink_count}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2
                )

            else:
                closed_since = None
                current_status = "Monitoring stopped"

            current_status = status
            success, buffer = cv2.imencode(".jpg", frame)

            if success:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buffer.tobytes() + b"\r\n"
                )

    finally:
        camera.release()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/start")
def start():
    global monitoring, blink_count, closed_since

    blink_count = 0
    closed_since = None
    current_status = "Face not detected"
    monitoring = True

    return jsonify({"status": "Monitoring started"})


@app.route("/stop")
def stop():
    global monitoring, closed_since

    monitoring = False
    closed_since = None

    return jsonify({"status": "Monitoring stopped"})

@app.route("/status")
def get_status():
    return jsonify({
        "detection_status": current_status,
        "blink_count": blink_count,
        "monitoring": monitoring
    })
@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    import webbrowser
    from threading import Timer

    Timer(2, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(debug=True, use_reloader=False)
