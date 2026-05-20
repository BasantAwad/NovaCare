import time
from flask import Flask, Response, render_template_string, request, jsonify
import cv2
import json
import multiprocessing
import subprocess
import os
import numpy as np
try:
    from flask_cors import CORS
except ImportError:
    # If not installed, we can try to install it or just pass
    CORS = None

app = Flask(__name__)
if CORS:
    CORS(app)
app.config['UPLOAD_FOLDER'] = '/tmp/audio_uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    try:
        os.makedirs(app.config['UPLOAD_FOLDER'])
    except:
        app.config['UPLOAD_FOLDER'] = './'

# Initialize Robot
try:
    from pop.Pilot import SerBot
    bot = SerBot()
    print("Motors initialized!")
except Exception as e:
    bot = None
    print(f"Motor warning: {e}")

# We use a separate process to safely fetch LiDAR data
def lidar_worker(shared_dict):
    try:
        from pop import LiDAR
        lidar = LiDAR.Rplidar()
        lidar.connect()
        lidar.startMotor()
        print("LiDAR initialized in subprocess!")
        while True:
            vectors = lidar.getVectors()
            data = []
            if isinstance(vectors, dict):
                for angle, dist in vectors.items():
                    data.append({"angle": float(angle), "distance": float(dist)})
            else:
                for v in vectors:
                    if isinstance(v, tuple) or isinstance(v, list):
                        data.append({"angle": float(v[0]), "distance": float(v[1])})
                    else:
                        data.append({"angle": float(v.angle), "distance": float(v.distance)})
            shared_dict['data'] = json.dumps(data)
            time.sleep(0.05) # Faster update rate
    except Exception as e:
        shared_dict['error'] = str(e)
        print(f"LiDAR subprocess error: {e}")

# Global dict for the web threads to read
shared_lidar_data = None
follow_mode = False

def set_max_volume():
    """Attempts to set all common audio mixers to 100% volume."""
    controls = ['Master', 'PCM', 'Line out', 'Headphone', 'Speaker', 'Front', 'Rear']
    for control in controls:
        try:
            # Set to 100% volume and unmute
            subprocess.run(['amixer', 'sset', control, '100%'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(['amixer', 'sset', control, 'unmute'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except:
            pass
    # PulseAudio fallback
    try:
        subprocess.run(['pactl', 'set-sink-volume', '0', '100%'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except:
        pass

def get_current_lidar_data():
    if shared_lidar_data is not None and shared_lidar_data.get('data') is not None:
        return json.loads(shared_lidar_data['data'])
    return []

def check_obstacle(angle_center, angle_width, min_dist):
    """Helper to check if an obstacle exists within a certain angular range and distance."""
    data = get_current_lidar_data()
    if not data:
        return False
    
    half_width = angle_width / 2
    for p in data:
        angle = p['angle']
        # Normalize angle to -180 to 180 or handle wrap-around
        # LiDAR 0 is usually front
        diff = (angle - angle_center + 180) % 360 - 180
        if abs(diff) <= half_width and p['distance'] < min_dist:
            return True
    return False

# Premium HTML Interface
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>SERBot NovaCare Command Center</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #6366f1;
            --secondary: #a855f7;
            --danger: #ef4444;
            --success: #22c55e;
            --bg: #0f172a;
            --card-bg: #1e293b;
            --text: #f8fafc;
        }
        body { 
            text-align: center; 
            font-family: 'Inter', sans-serif; 
            background: var(--bg); 
            color: var(--text); 
            margin: 0; 
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .container { 
            display: flex; 
            flex-direction: row; 
            justify-content: center; 
            align-items: flex-start; 
            gap: 24px; 
            flex-wrap: wrap; 
            max-width: 1400px;
            width: 100%;
        }
        .panel { 
            border: 1px solid rgba(255,255,255,0.1); 
            border-radius: 16px; 
            padding: 24px; 
            background: var(--card-bg); 
            box-shadow: 0 10px 25px -5px rgba(0,0,0,0.3);
            backdrop-filter: blur(10px);
            transition: transform 0.2s;
        }
        .panel:hover { transform: translateY(-4px); }
        h1 { font-size: 2.5rem; font-weight: 700; margin-bottom: 40px; background: linear-gradient(to right, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        h2 { margin-top: 0; font-weight: 600; color: #cbd5e1; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 12px; margin-bottom: 20px; }
        
        .video-container { position: relative; border-radius: 12px; overflow: hidden; background: #000; line-height: 0; }
        img { max-width: 100%; height: auto; }
        
        canvas { background: #020617; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); }
        
        .controls-grid { 
            display: grid; 
            grid-template-columns: repeat(3, 1fr); 
            gap: 12px; 
            margin-top: 20px; 
        }
        .btn { 
            font-size: 16px; 
            padding: 14px; 
            cursor: pointer; 
            border-radius: 10px; 
            border: none; 
            background: rgba(255,255,255,0.05); 
            color: white; 
            transition: all 0.2s; 
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
        }
        .btn:hover { background: rgba(255,255,255,0.15); transform: scale(1.05); }
        .btn:active { transform: scale(0.95); }
        
        .btn-primary { background: var(--primary); }
        .btn-primary:hover { background: #4f46e5; }
        .btn-danger { background: var(--danger); }
        .btn-success { background: var(--success); }
        
        .btn-follow { 
            width: 100%; 
            margin-top: 20px; 
            padding: 16px;
            font-size: 18px;
            background: #475569;
        }
        .btn-follow.active { 
            background: linear-gradient(135deg, #a855f7, #6366f1);
            box-shadow: 0 0 20px rgba(168, 85, 247, 0.4);
        }

        .audio-section {
            display: flex;
            flex-direction: column;
            gap: 12px;
            margin-top: 15px;
        }
        .file-input-wrapper {
            position: relative;
            overflow: hidden;
            display: inline-block;
            width: 100%;
        }
        .file-input-wrapper input[type=file] {
            font-size: 100px;
            position: absolute;
            left: 0;
            top: 0;
            opacity: 0;
            cursor: pointer;
        }
        
        .status-msg { font-size: 14px; margin-top: 12px; min-height: 20px; transition: color 0.3s; }
        .indicator { width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 8px; }
        .online { background: var(--success); box-shadow: 0 0 8px var(--success); }
        .offline { background: var(--danger); }
    </style>
</head>
<body>
    <h1>NovaCare SERBot Command</h1>
    
    <div class="container">
        <!-- Live Feed Panel -->
        <div class="panel" style="flex: 2; min-width: 600px;">
            <h2>Live Intelligence Feed</h2>
            <div class="video-container">
                <img src="/video_feed" id="mainFeed" />
            </div>
            
            <div class="controls-grid">
                <div></div>
                <button class="btn" onmousedown="move('forward')" onmouseup="move('stop')">Forward ⬆️</button>
                <div></div>
                
                <button class="btn" onmousedown="move('left')" onmouseup="move('stop')">Left ⬅️</button>
                <button class="btn btn-danger" onclick="move('stop')">STOP ⏹️</button>
                <button class="btn" onmousedown="move('right')" onmouseup="move('stop')">Right ➡️</button>
                
                <div></div>
                <button class="btn" onmousedown="move('backward')" onmouseup="move('stop')">Backward ⬇️</button>
                <div></div>
            </div>
            
            <button class="btn btn-follow" onclick="toggleFollow()" id="followBtn">
                <span id="followIcon">👤</span> Start Person Following
            </button>
        </div>
        
        <!-- Sensor Data Panel -->
        <div class="panel" style="flex: 1; min-width: 350px;">
            <h2>LiDAR Environment</h2>
            <canvas id="lidarCanvas" width="350" height="350"></canvas>
            <div style="margin-top: 15px; text-align: left; font-size: 13px; color: #94a3b8;">
                <p><span class="indicator online"></span> System Operational</p>
                <p>Resolution: 360° Scan</p>
                <p>Obstacle Avoidance: Active</p>
            </div>
            
            <h2 style="margin-top: 30px;">Audio Diagnostics</h2>
            <div class="audio-section">
                <button class="btn btn-primary" onclick="testSpeaker()">🔊 Speak Test Text</button>
                <button class="btn" style="background: #ea580c;" onclick="boostVolume()">🔥 Boost Audio Volume</button>
                
                <div class="file-input-wrapper">
                    <button class="btn" style="width: 100%; background: #334155;">📁 Upload Audio File</button>
                    <input type="file" id="audioFile" accept=".wav,.mp3" onchange="uploadAudio()" />
                </div>
                
                <button class="btn btn-success" onclick="testMic()">🎤 Record & Test Mic</button>
                <button class="btn" style="background: #0ea5e9; margin-top: 10px;" onclick="launchDisplay()">🖥️ Launch UI on Robot Screen</button>
                <p id="audio-result" class="status-msg"></p>
            </div>
        </div>
    </div>

    <script>
        let followActive = false;

        function launchDisplay() {
            const status = document.getElementById('audio-result');
            status.innerText = "Launching frontend on robot screen...";
            status.style.color = "#0ea5e9";
            fetch('/launch_display').then(r => r.text()).then(t => {
                status.innerText = t;
                status.style.color = "var(--success)";
            });
        }

        function move(direction) {
            fetch('/move/' + direction);
        }

        // --- Low-Latency Multi-Key Movement ---
        let pressedKeys = new Set();
        let lastCommand = 'stop';

        function processMovement() {
            let cmd = 'stop';
            const up = pressedKeys.has('ArrowUp');
            const down = pressedKeys.has('ArrowDown');
            const left = pressedKeys.has('ArrowLeft');
            const right = pressedKeys.has('ArrowRight');

            if (up && left) cmd = 'forward_left';
            else if (up && right) cmd = 'forward_right';
            else if (down && left) cmd = 'backward_left';
            else if (down && right) cmd = 'backward_right';
            else if (up) cmd = 'forward';
            else if (down) cmd = 'backward';
            else if (left) cmd = 'left';
            else if (right) cmd = 'right';

            if (cmd !== lastCommand) {
                lastCommand = cmd;
                fetch('/move/' + cmd);
            }
        }

        window.addEventListener('keydown', function(e) {
            if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(e.code)) {
                e.preventDefault();
                if (!pressedKeys.has(e.code)) {
                    pressedKeys.add(e.code);
                    processMovement();
                }
            }
        });

        window.addEventListener('keyup', function(e) {
            if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(e.code)) {
                pressedKeys.delete(e.code);
                processMovement();
            }
        });
        
        function toggleFollow() {
            fetch('/toggle_follow');
            followActive = !followActive;
            const btn = document.getElementById('followBtn');
            if (followActive) {
                btn.classList.add('active');
                btn.innerHTML = '<span>🛑</span> Stop Person Following';
            } else {
                btn.classList.remove('active');
                btn.innerHTML = '<span>👤</span> Start Person Following';
            }
        }
        
        function uploadAudio() {
            const fileInput = document.getElementById('audioFile');
            const file = fileInput.files[0];
            if (!file) return;

            const formData = new FormData();
            formData.append('audio', file);
            
            const status = document.getElementById('audio-result');
            status.innerText = "Uploading & playing...";
            status.style.color = "var(--primary)";

            fetch('/upload_audio', {
                method: 'POST',
                body: formData
            })
            .then(r => r.json())
            .then(data => {
                status.innerText = data.message;
                status.style.color = data.success ? "var(--success)" : "var(--danger)";
            })
            .catch(err => {
                status.innerText = "Error uploading file";
                status.style.color = "var(--danger)";
            });
        }

        function testSpeaker() {
            const status = document.getElementById('audio-result');
            status.innerText = "Playing text...";
            status.style.color = "var(--primary)";
            fetch('/test_speaker?text=Testing+robot+speakers+system').then(r => r.text()).then(t => {
                status.innerText = t;
                status.style.color = "var(--success)";
            });
        }

        function boostVolume() {
            const status = document.getElementById('audio-result');
            status.innerText = "Boosting hardware volume...";
            status.style.color = "#fbbf24";
            fetch('/boost_volume').then(r => r.text()).then(t => {
                status.innerText = t;
                status.style.color = "var(--success)";
            });
        }

        function testMic() {
            const status = document.getElementById('audio-result');
            status.innerText = "Recording (3s)... Speak now!";
            status.style.color = "#fbbf24";
            fetch('/test_mic').then(r => r.text()).then(t => {
                status.innerText = t;
                status.style.color = "var(--success)";
            });
        }

        function fetchLidar() {
            fetch('/lidar_data').then(r => r.json()).then(data => {
                const canvas = document.getElementById('lidarCanvas');
                const ctx = canvas.getContext('2d');
                const cx = canvas.width / 2;
                const cy = canvas.height / 2;
                
                ctx.fillStyle = '#020617';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                
                // Grid lines
                ctx.strokeStyle = 'rgba(255,255,255,0.05)';
                ctx.beginPath();
                for(let i=0; i<4; i++) {
                    ctx.arc(cx, cy, (i+1)*40, 0, 2*Math.PI);
                }
                ctx.stroke();
                
                // Robot
                ctx.fillStyle = '#6366f1';
                ctx.beginPath();
                ctx.arc(cx, cy, 6, 0, 2 * Math.PI);
                ctx.fill();
                
                // Points
                ctx.fillStyle = '#ef4444';
                data.forEach(pt => {
                    const angle_rad = (pt.angle - 90) * Math.PI / 180;
                    const dist_px = pt.distance / 15; 
                    const x = cx + dist_px * Math.cos(angle_rad);
                    const y = cy + dist_px * Math.sin(angle_rad);
                    if (x >= 0 && x <= canvas.width && y >= 0 && y <= canvas.height) {
                        ctx.fillRect(x, y, 2, 2);
                    }
                });
            }).catch(err => console.error("LiDAR Error:", err));
        }
        
        setInterval(fetchLidar, 200);
    </script>
</body>
</html>
"""

def generate_frames():
    try:
        from pop import Util
        Util.enable_imshow()
        pipeline = Util.gstrmer(640, 480)
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    except:
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    # Initialize MediaPipe Pose (Better than face detection for tracking people)
    mp_pose = None
    pose = None
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        print("✅ MediaPipe Pose initialized!")
    except Exception as e:
        print(f"❌ MediaPipe Pose failed: {e}")

    global follow_mode
    while True:
        try:
            success, frame = cap.read()
            if not success or frame is None:
                continue
                
            if follow_mode and pose:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)
                
                if results.pose_landmarks:
                    # Draw landmarks for visual feedback
                    h, w, c = frame.shape
                    # Use center of hips/shoulders as the target
                    landmarks = results.pose_landmarks.landmark
                    # Target center point (average of shoulders)
                    left_sh = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                    right_sh = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    
                    target_x = (left_sh.x + right_sh.x) / 2 * w
                    target_y = (left_sh.y + right_sh.y) / 2 * h
                    
                    # Bounding box estimate based on shoulders/hips
                    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                    
                    # Approximate width of person
                    person_width = abs(left_sh.x - right_sh.x) * w
                    
                    # Visual indicators
                    cv2.circle(frame, (int(target_x), int(target_y)), 10, (0, 255, 0), -1)
                    cv2.putText(frame, "HUMAN TARGET LOCKED", (int(target_x)-50, int(target_y)-20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    if bot:
                        # --- OBSTACLE AVOIDANCE INTEGRATION ---
                        # Check if something is in front while we want to move
                        obstacle_ahead = check_obstacle(0, 50, 400) # 50deg wide, 400mm dist
                        
                        if obstacle_ahead:
                            cv2.putText(frame, "🛑 OBSTACLE AVOIDANCE ACTIVE", (10, 60), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            bot.stop()
                        else:
                            frame_center = w / 2
                            if target_x < frame_center - 80:
                                bot.turnLeft()
                            elif target_x > frame_center + 80:
                                bot.turnRight()
                            else:
                                # person_width is a proxy for distance
                                # Adjust these thresholds based on testing
                                if person_width < 100:  # Far away
                                    bot.forward(35)
                                elif person_width > 200: # Too close
                                    bot.backward(30)
                                else:
                                    bot.stop()
                else:
                    cv2.putText(frame, "Searching for person...", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    if bot:
                        bot.stop()
            
            # Encode and send
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as e:
            print(f"Video feed error: {e}")
            break

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_follow')
def toggle_follow():
    global follow_mode
    follow_mode = not follow_mode
    if not follow_mode and bot:
        bot.stop()
    return "OK"

@app.route('/lidar_data')
def lidar_data_route():
    data = get_current_lidar_data()
    return jsonify(data)

@app.route('/move/<direction>')
def move_robot(direction):
    global follow_mode
    if follow_mode: follow_mode = False
    if not bot: return "No robot", 500
    
    speed = 35
    
    # Obstacle avoidance for any forward-inclined movement
    if 'forward' in direction and check_obstacle(0, 70, 350):
        bot.stop()
        return "Obstacle detected", 403

    # Primary Directions
    if direction == 'forward': bot.forward(speed)
    elif direction == 'backward': bot.backward(speed)
    elif direction == 'left': bot.turnLeft()
    elif direction == 'right': bot.turnRight()
    
    # Diagonal Directions (NW, NE, SW, SE)
    # Using specific methods if available in pop Pilot, else fallback to composite
    elif direction == 'forward_left':
        try: bot.forwardLeft(speed)
        except: bot.forward(speed); bot.turnLeft()
    elif direction == 'forward_right':
        try: bot.forwardRight(speed)
        except: bot.forward(speed); bot.turnRight()
    elif direction == 'backward_left':
        try: bot.backwardLeft(speed)
        except: bot.backward(speed); bot.turnLeft()
    elif direction == 'backward_right':
        try: bot.backwardRight(speed)
        except: bot.backward(speed); bot.turnRight()
        
    elif direction == 'stop': bot.stop()
    return "OK"

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({"success": False, "message": "No file part"}), 400
    file = request.files['audio']
    if file.filename == '':
        return jsonify({"success": False, "message": "No selected file"}), 400
    
    if file:
        filename = "test_audio_" + str(int(time.time())) + os.path.splitext(file.filename)[1]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Play the audio
        try:
            # We use 'aplay' for wav or 'mpg123'/'cvlc' for mp3 if available
            # Standard Jetson usually has aplay
            if filename.endswith('.wav'):
                subprocess.Popen(['aplay', filepath])
            else:
                # Try to use cvlc (vlc) if installed, or just try aplay (might fail for mp3)
                subprocess.Popen(['cvlc', '--play-and-exit', filepath])
            
            return jsonify({"success": True, "message": f"Playing {file.filename} on robot!"})
        except Exception as e:
            return jsonify({"success": False, "message": str(e)}), 500

@app.route('/test_speaker')
def test_speaker():
    set_max_volume()
    text = request.args.get('text', 'Speaker system test')
    try:
        # -a 200 for max amplitude (0-200)
        subprocess.run(['espeak', '-a', '200', text])
        return "Speaker test played successfully at max volume!"
    except Exception as e:
        return f"Error: {e}", 500

@app.route('/boost_volume')
def boost_volume_route():
    set_max_volume()
    return "Hardware volume set to 100% on all channels."

@app.route('/test_mic')
def test_mic():
    filepath = '/tmp/mic_test.wav'
    try:
        if os.path.exists(filepath): os.remove(filepath)
        result = subprocess.run(['arecord', '-d', '3', '-f', 'cd', filepath], timeout=5)
        if result.returncode == 0:
            return "Microphone test passed! Recorded 3 seconds."
        return "Microphone test failed.", 500
    except Exception as e:
        return f"Error: {e}", 500

@app.route('/launch_display')
def launch_display():
    """Launch Chromium on the robot's physical display pointing to the frontend with camera permissions."""
    try:
        # Use the IP of the laptop that is currently accessing this dashboard
        laptop_ip = request.remote_addr
        
        # Allow passing a custom URL via query param, fallback to laptop's dev server
        # We use port 3001 as default since Next.js often jumps there if 3000 is busy
        target_url = request.args.get('url', f"http://{laptop_ip}:3001/rover")
        
        # Flags explained:
        # --kiosk: Fullscreen mode
        # --use-fake-ui-for-media-stream: Auto-accept camera/mic permissions
        # --unsafely-treat-insecure-origin-as-secure: Allows getUserMedia on HTTP (non-localhost)
        chrome_flags = [
            "--kiosk",
            "--disable-infobars",
            "--no-first-run",
            "--use-fake-ui-for-media-stream",
            f"--unsafely-treat-insecure-origin-as-secure={target_url}",
            "--window-position=0,0",
            "--check-for-update-interval=31536000", # Disable update checks
            "--ignore-certificate-errors"
        ]
        
        flags_str = " ".join(chrome_flags)
        
        # Display :0 is the default local display on Jetson
        cmd = f"DISPLAY=:0 chromium-browser {flags_str} {target_url} &"
        
        print(f"[Display] Launching: {cmd}")
        subprocess.Popen(cmd, shell=True)
        return f"Display command sent! Target: {target_url}"
    except Exception as e:
        return f"Failed to launch display: {e}", 500

if __name__ == '__main__':
    manager = multiprocessing.Manager()
    shared_lidar_data = manager.dict()
    shared_lidar_data['data'] = None
    
    lidar_p = multiprocessing.Process(target=lidar_worker, args=(shared_lidar_data,))
    lidar_p.start()
    
    print("=======================================")
    print(" SERBot COMMAND CENTER ACTIVE ")
    print("=======================================")
    app.run(host='0.0.0.0', port=5000, threaded=True)
