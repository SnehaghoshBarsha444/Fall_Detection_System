# 🛡️ Fall Detection System

A **real-time, multi-user fall detection and posture monitoring system** powered by **computer vision and advanced analytics**.  
Designed for **home care, clinical monitoring, and assisted-living environments**.

---

## 🚀 Features

- **Real-time Capturing** → Continuously processes live camera feed for accurate fall detection.  
- **Body Angle Logging** → Records and tracks posture angles over time.  
- **History Log** → Maintains detection events with per-person statistics.  
- **Secure Live Stream Sharing** → Share the ongoing detection feed over the **local network** for remote monitoring.  
- **Multi-person Tracking** → Detects, identifies, and logs falls for all individuals in the frame.  

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SnehaghoshBarsha444/fall-detection-system.git
   cd fall-detection-system
````

2. **Install dependencies**

   ```bash
   pip install PyQt5 opencv-python mediapipe numpy Flask
   ```

---

## ▶️ Usage

To start the application:

```bash
python app.py
```

### 🖥️ User Interface

* **Center** → Live camera feed with on-screen fall/posture visualization.
* **Left Panel** → Real-time analytics (fall count, body angle, detection state).
* **Top-Right** → Stream sharing button + log (timestamps, URLs).
* **Bottom-Right** → Person history panel (unique persons + fall counts).

### 🌐 Live Stream

Click **“Share Secure Dashboard (HTTPS)”** → Get the local network link in the sharing log.
View the live feed from **any device** on the same network.

---

## 🔬 How It Works

1. **Pose Estimation** → Uses [MediaPipe](https://github.com/google/mediapipe) for real-time human pose detection.
2. **Fall Detection Algorithm** → Analyzes velocity, body angle, and movement transitions to distinguish falls from normal activity.
3. **Multi-User Recognition** → Differentiates between individuals for accurate reporting. *(Future-ready: facial recognition integration)*
4. **Event Logging** → Each detection event is stored with timestamp, person ID, fall state, and recovery attempts.
5. **Remote Monitoring** → Streams a secure dashboard accessible via local URL.

---

## 📊 History & Analytics

* Per-user **fall statistics** and **detection history**.
* Session logs for **caregiver review** and **further analysis**.

---

## ❗ Troubleshooting

* **Camera/Streaming Errors** → Check permissions & open required ports.
* **Multi-person Differentiation** → Enhance with a face recognition module.
* **Performance Issues** → Reduce camera resolution or frame rate.

---

## 📜 License

This project is open-source for **research and educational purposes**.
⚠️ For **clinical or commercial deployment**, additional validation & certification is required.

---
