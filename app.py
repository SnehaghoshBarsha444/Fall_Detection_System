# app.py

import sys
import cv2
import mediapipe as mp
import numpy as np
import threading
import socket
import time
import webbrowser
from collections import deque
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QPushButton, QTextEdit, QVBoxLayout, QHBoxLayout, QSizePolicy, QListWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from flask import Flask, Response

# --- KEY PARAMETERS FOR ACCURACY TUNING ---
class Cfg:
    VELOCITY_THRESHOLD = 0.04; ON_GROUND_THRESHOLD = 0.7; SUSTAINED_DOWN_STATE_SECONDS = 1.5

# --- System states for the "Three-Check" logic ---
class SystemState:
    STABLE, HIGH_VELOCITY_DETECTED, ON_GROUND_DETECTED, FALL_CONFIRMED = 0, 1, 2, 3

# --- Core Detection Logic ---
class FallDetector:
    def __init__(self):
        self.state = SystemState.STABLE; self.torso_y_history = deque(maxlen=10)
        self.on_ground_start_time = 0; self.fall_count = 0
    def process_frame(self, pose_landmarks):
        if not pose_landmarks:
            if self.state != SystemState.STABLE: self.state = SystemState.STABLE
            return "No Person", False, {}
        lms = pose_landmarks.landmark
        if any(lms[p].visibility < 0.6 for p in [11, 12, 23, 24]): return "Low Confidence", False, {}
        torso_y = (lms[11].y + lms[12].y + lms[23].y + lms[24].y) / 4
        self.torso_y_history.append(torso_y)
        if len(self.torso_y_history) < 10: return "Initializing", False, {}
        velocity_y = self.torso_y_history[-1] - self.torso_y_history[-5]
        alert = False; current_state_str = "STABLE"

        if self.state == SystemState.STABLE:
            if velocity_y > Cfg.VELOCITY_THRESHOLD: self.state = SystemState.HIGH_VELOCITY_DETECTED
        elif self.state == SystemState.HIGH_VELOCITY_DETECTED:
            current_state_str = "HIGH VELOCITY"
            if torso_y > Cfg.ON_GROUND_THRESHOLD:
                self.state = SystemState.ON_GROUND_DETECTED; self.on_ground_start_time = time.time()
            else: self.state = SystemState.STABLE
        elif self.state == SystemState.ON_GROUND_DETECTED:
            current_state_str = "ON GROUND"
            if torso_y > Cfg.ON_GROUND_THRESHOLD:
                if (time.time() - self.on_ground_start_time) > Cfg.SUSTAINED_DOWN_STATE_SECONDS:
                    self.state = SystemState.FALL_CONFIRMED; self.fall_count += 1; alert = True
            else: self.state = SystemState.STABLE
        elif self.state == SystemState.FALL_CONFIRMED:
            current_state_str = "FALL CONFIRMED"
            if torso_y < Cfg.ON_GROUND_THRESHOLD: self.state = SystemState.STABLE
        
        metrics = {"State": current_state_str, "Torso Position": torso_y, "Vertical Velocity": velocity_y, "Total Falls": self.fall_count}
        return current_state_str, alert, metrics

# --- Simple Person Tracker for History Panel ---
class SimplePersonTracker:
    def __init__(self):
        self.persons = {}; self.next_person_id = 0
    def _get_bbox(self, landmarks):
        if not landmarks: return None
        lms = landmarks.landmark; x_coords, y_coords = [lm.x for lm in lms], [lm.y for lm in lms]
        return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
    def _iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1]); xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]); boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)
    def update(self, pose_landmarks, fall_detected):
        bbox = self._get_bbox(pose_landmarks)
        for pid in list(self.persons.keys()):
            self.persons[pid]["frames_unseen"] += 1
            if self.persons[pid]["frames_unseen"] > 30: del self.persons[pid]
        if bbox:
            best_match_id = -1; max_iou = 0.3
            for pid, p_data in self.persons.items():
                iou = self._iou(bbox, p_data["bbox"])
                if iou > max_iou: max_iou = iou; best_match_id = pid
            
            if best_match_id != -1:
                self.persons[best_match_id]["bbox"] = bbox; self.persons[best_match_id]["frames_unseen"] = 0
                if fall_detected: self.persons[best_match_id]["falls"] += 1
            else:
                self.persons[self.next_person_id] = {"bbox": bbox, "appearances": 1, "falls": 1 if fall_detected else 0, "frames_unseen": 0}
                self.next_person_id += 1
        return self.persons

# --- GLOBAL SHARED BUFFER & STREAMING SERVER ---
class StreamingBuffer:
    def __init__(self): self.frame = None; self.lock = threading.Lock()
    def update_frame(self, frame):
        with self.lock: self.frame = frame
    def generate_stream(self):
        while True:
            time.sleep(1/20)
            with self.lock:
                if self.frame is None: continue
                (flag, encodedImage) = cv2.imencode(".jpg", self.frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if not flag: continue
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

GLOBAL_BUFFER = StreamingBuffer()
flask_app = Flask(__name__)
@flask_app.route("/")
def index(): return "<html><body style='background-color:#111;'><img src='/video_feed' width='100%'></body></html>"
@flask_app.route("/video_feed")
def video_feed(): return Response(GLOBAL_BUFFER.generate_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")
def run_server(): flask_app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, ssl_context='adhoc')

# --- VIDEO WORKER ---
class VideoWorker(QThread):
    frame_ready = pyqtSignal(QImage); metrics_ready = pyqtSignal(dict); history_ready = pyqtSignal(dict)
    def __init__(self):
        super().__init__()
        self.running = True
        self.drawing_spec_normal = mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=2)
        self.drawing_spec_alert = mp.solutions.drawing_utils.DrawingSpec(color=(0,0,255), thickness=2)
    def run(self):
        mp_pose = mp.solutions.pose; pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        detector = FallDetector(); tracker = SimplePersonTracker()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened(): print("Error: Cannot open camera."); return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        while self.running and cap.isOpened():
            success, frame = cap.read()
            if not success: time.sleep(0.1); continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            status, alert, metrics = detector.process_frame(results.pose_landmarks)
            history = tracker.update(results.pose_landmarks, alert)
            drawing_spec = self.drawing_spec_alert if alert else self.drawing_spec_normal
            mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS, landmark_drawing_spec=drawing_spec)
            GLOBAL_BUFFER.update_frame(frame.copy())
            h, w, ch = frame.shape; bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888).copy()
            self.frame_ready.emit(qt_image); self.metrics_ready.emit(metrics); self.history_ready.emit(history)
            time.sleep(0.01)
        cap.release(); pose.close()
    def stop(self): self.running = False

# --- MAIN UI ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Fall Detection System v15.2"); self.setGeometry(50, 50, 1480, 900); self.apply_stylesheet()
        central_widget = QWidget(self); self.setCentralWidget(central_widget)
        main_hbox = QHBoxLayout(central_widget); main_hbox.setContentsMargins(10, 10, 10, 10)
        
        # --- LEFT PANEL: Analytics ---
        self.left_analytics = QTextEdit(); self.left_analytics.setReadOnly(True)
        self.left_analytics.setFixedWidth(320); self.left_analytics.setObjectName("analyticsPanel")
        
        # --- CENTER: Video feed ---
        self.video_label = QLabel("Connecting..."); self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setObjectName("videoLabel"); self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # --- RIGHT PANELS: Share/Log/History ---
        right_panel = QVBoxLayout()
        top_right_box = QVBoxLayout()
        self.share_button = QPushButton(" Share Secure Dashboard (HTTPS)")
        self.share_log = QListWidget(); self.share_log.setFixedHeight(120); self.share_log.setObjectName("shareLogList")
        top_right_box.addWidget(self.share_button); top_right_box.addWidget(QLabel("Sharing Log:"))
        top_right_box.addWidget(self.share_log); top_right_box.addStretch()
        
        bottom_right_box = QVBoxLayout()
        self.history_list = QListWidget(); self.history_list.setObjectName("historyList")
        bottom_right_box.addWidget(QLabel("Person Detection History:")); bottom_right_box.addWidget(self.history_list)
        
        right_panel.addLayout(top_right_box); right_panel.addLayout(bottom_right_box)
        
        # --- Layout Placement ---
        main_hbox.addWidget(self.left_analytics, 0); main_hbox.addWidget(self.video_label, 1); main_hbox.addLayout(right_panel, 0)
        
        self.video_worker = VideoWorker(); self.stream_thread = None; self.is_streaming = False
        self.video_worker.frame_ready.connect(self.update_video_frame)
        self.video_worker.metrics_ready.connect(self.update_analytics)
        self.video_worker.history_ready.connect(self.update_history)
        self.share_button.clicked.connect(self.toggle_sharing)
        self.video_worker.start()

    def update_analytics(self, metrics):
        html = "<style>td{padding:4px;font-size:13px;}.key{font-weight:bold;color:#a9b7c6;}</style><table>"
        for key, value in metrics.items():
            val_str = f"{value:.3f}" if isinstance(value, float) else str(value)
            html += f"<tr><td class='key'>{key}</td><td>{val_str}</td></tr>"
        html += "</table>"; self.left_analytics.setHtml(html)

    def update_history(self, history_data):
        self.history_list.clear()
        for pid, data in sorted(history_data.items()):
            self.history_list.addItem(f"Person {pid}: {data['falls']} falls, {data['appearances']} appearances")

    def add_share_log(self, text, timestamp): self.share_log.insertItem(0, f"{timestamp}: {text}")

    def update_video_frame(self, image): self.video_label.setPixmap(QPixmap.fromImage(image).scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
    
    def toggle_sharing(self):
        if not self.is_streaming:
            self.stream_thread = threading.Thread(target=run_server, daemon=True); self.stream_thread.start()
            self.is_streaming = True; self.share_button.setText(" Stop Sharing"); self.share_button.setObjectName("shareButtonStop")
            self.share_button.setStyleSheet(self.styleSheet())
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(("8.8.8.8", 80)); ip_addr = s.getsockname()[0]; s.close()
                url = f"https://{ip_addr}:5000"
                webbrowser.open(url)
                QApplication.clipboard().setText(url)
                self.add_share_log("URL Copied & Opened!", time.strftime('%H:%M:%S'))
                self.share_log.addItem(url)
            except Exception as e:
                self.add_share_log(f"Error: {e}", time.strftime('%H:%M:%S'))
        else:
            self.is_streaming = False; self.share_button.setText(" Share Secure Dashboard (HTTPS)"); self.share_button.setObjectName("shareButton")
            self.share_button.setStyleSheet(self.styleSheet()); self.add_share_log("Sharing stopped.", time.strftime('%H:%M:%S'))
            
    def closeEvent(self, event): self.video_worker.stop(); self.video_worker.wait(); event.accept()
    
    def apply_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; }
            QTextEdit#analyticsPanel { background: #282828; color: #eee; font-size:13px; border-radius: 8px; }
            QLabel#videoLabel { background-color: #111; border-radius: 9px; }
            QListWidget#historyList, QListWidget#shareLogList { background: #222; color: #dfd; border-radius: 8px; font-size: 12px; }
            QPushButton { background-color: #5588a3; color: #fff; border: none; padding: 12px 15px; border-radius: 5px; font-weight: bold; font-size: 14px; }
            QPushButton:hover { background-color: #65aadc; }
            QPushButton#shareButtonStop { background-color: #2a5c32; }
            QPushButton#shareButtonStop:hover { background-color: #3a7c42; }
            QLabel { color: #a9b7c6; font-size: 12px; font-weight: bold; }
        """)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
