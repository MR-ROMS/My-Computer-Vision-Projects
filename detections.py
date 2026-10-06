import cv2
import requests
import threading
import time
import os
import numpy as np
from ultralytics import YOLO
from collections import deque

# ── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_PATH  = r'D:\CV_Projects\runs\detect\runs\train\traffic_v11\weights\best.pt'
ESP32_IP = 'http://10.27.99.141'
ESP32_URL = f'{ESP32_IP}:81/stream'

# Model confidence thresholds
CLASS_CONF = {
    0: 0.005,  # Green_Light - VERY tolerant
    1: 0.008,  # Red_Light - MORE TOLERANT
    2: 0.01,   # Right_Turn
    3: 0.01,   # Left_Turn
}

# Color detection thresholds - ULTRA TIGHT (only pure colors)
GREEN_LOWER = np.array([55, 100, 100])    # Extremely tight green
GREEN_UPPER = np.array([65, 255, 255])

RED_LOWER1 = np.array([0, 100, 100])      # Extremely tight red
RED_UPPER1 = np.array([3, 255, 255])
RED_LOWER2 = np.array([177, 100, 100])
RED_UPPER2 = np.array([180, 255, 255])

# Minimum confidence to even consider a detection
ABSOLUTE_MIN_CONF = 0.001

COMMANDS = {
    'Green_Light' : 'F',
    'Red_Light'   : 'S',
    'Right_Turn'  : 'R',
    'Left_Turn'   : 'L',
}

CLASS_COLORS = {
    'Green_Light' : (  0, 255,   0),
    'Red_Light'   : (  0,   0, 255),
    'Right_Turn'  : (  0, 165, 255),
    'Left_Turn'   : (255,   0, 255),
}

# For smoothing detections
DETECTION_HISTORY = {
    'Green_Light': deque(maxlen=10),
    'Red_Light': deque(maxlen=10),
    'Right_Turn': deque(maxlen=10),
    'Left_Turn': deque(maxlen=10),
}

# Track last commands to prevent spam
last_command_time = 0
command_cooldown = 0.5  # Seconds between commands

def send_command(cmd):
    global last_command_time
    current_time = time.time()
    
    # Don't send commands too frequently
    if current_time - last_command_time < command_cooldown:
        return False
    
    try:
        response = requests.post(f'{ESP32_IP}/motor', data=cmd, timeout=0.5)
        print(f" Command sent: {cmd}, Response: {response.status_code}")
        last_command_time = current_time
        return True
    except Exception as e:
        print(f" Send error: {e}")
        return False

class ESP32Stream:
    def __init__(self, url):
        self.url     = url
        self.frame   = None
        self.lock    = threading.Lock()
        self.running = True
        self.thread  = threading.Thread(target=self._grab, daemon=True)
        self.thread.start()

    def _grab(self):
        while self.running:
            try:
                cap = cv2.VideoCapture(self.url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                for _ in range(5):
                    cap.read()
                
                while self.running:
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        if frame.shape[0] > 0 and frame.shape[1] > 0:
                            frame = cv2.resize(frame, (640, 480))
                            with self.lock:
                                self.frame = frame
                    else:
                        print("Stream lost — reconnecting...")
                        break
                cap.release()
            except Exception as e:
                print(f"Stream error: {e}")
            time.sleep(1)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False

def is_traffic_light_shape(contour, frame_shape):
    """
    ULTRA STRICT filter - only perfect traffic light shapes
    Returns: (is_valid, confidence_boost)
    """
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    
    # CRITICAL: Must be in upper 35% of frame (most traffic lights are high)
    if y + h > frame_shape[0] * 0.35:
        return False, 0
    
    # Very specific size range for traffic lights
    if w < 30 or h < 30 or w > 80 or h > 80:
        return False, 0
    
    # Almost perfect aspect ratio (traffic lights are nearly circular)
    aspect_ratio = w / h
    if aspect_ratio < 0.85 or aspect_ratio > 1.15:
        return False, 0
    
    # Calculate circularity
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False, 0
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    # Must be very circular (like a real traffic light)
    if circularity < 0.8:
        return False, 0
    
    # Check convexity (traffic lights should be convex)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return False, 0
    convexity = area / hull_area
    if convexity < 0.9:  # Must be mostly convex
        return False, 0
    
    # Confidence boost for perfect shape
    confidence_boost = 0.2
    if circularity > 0.9:
        confidence_boost += 0.1
    if 0.9 < aspect_ratio < 1.1:
        confidence_boost += 0.1
    if y < frame_shape[0] * 0.2:  # Very high in frame
        confidence_boost += 0.1
    
    return True, min(confidence_boost, 0.5)

def detect_color_by_hsv(frame, lower, upper, min_area=300):
    """
    ULTRA STRICT color detection with maximum filtering
    Returns: (x, y, w, h, confidence) or None
    """
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create mask for color
    mask = cv2.inRange(hsv, lower, upper)
    
    # Aggressive morphological operations
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=3)  # Heavy erosion
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=1)  # Final cleanup
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Filter contours by traffic light shape
    valid_contours = []
    for contour in contours:
        is_valid, _ = is_traffic_light_shape(contour, frame.shape)
        if is_valid:
            valid_contours.append(contour)
    
    if not valid_contours:
        return None
    
    # Find the largest valid contour
    largest_contour = max(valid_contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    # High minimum area threshold
    if area < min_area:
        return None
    
    # Get bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Get confidence boost from shape
    _, confidence_boost = is_traffic_light_shape(largest_contour, frame.shape)
    
    # Conservative confidence calculation
    base_confidence = min(0.35, (area / 5000) * 0.2)
    confidence = base_confidence + confidence_boost
    
    return (x, y, w, h, min(confidence, 0.8))

def detect_green_by_color(frame):
    """Detect green areas that look like traffic lights"""
    return detect_color_by_hsv(frame, GREEN_LOWER, GREEN_UPPER, min_area=300)

def detect_red_by_color(frame):
    """Detect red areas that look like traffic lights"""
    result1 = detect_color_by_hsv(frame, RED_LOWER1, RED_UPPER1, min_area=300)
    result2 = detect_color_by_hsv(frame, RED_LOWER2, RED_UPPER2, min_area=300)
    
    # Return the one with higher confidence
    if result1 is None and result2 is None:
        return None
    elif result1 is None:
        return result2
    elif result2 is None:
        return result1
    else:
        return result1 if result1[4] > result2[4] else result2

def draw_box(frame, x1, y1, x2, y2, label, color, conf, is_color_detection=False, is_red=False):
    # Different style for color-based detection
    thickness = 3 if is_color_detection else 2
    line_type = cv2.LINE_AA
    
    # Special border for color detection
    if is_color_detection:
        cv2.rectangle(frame, (x1-3, y1-3), (x2+3, y2+3), (255, 255, 255), 1, line_type)
    
    # Draw main box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, line_type)
    
    # Add dashed border for color detection
    if is_color_detection:
        dash_length = 6
        for i in range(0, 12, 2):
            # Top edge
            cv2.line(frame, (x1 + i*dash_length, y1), 
                     (x1 + (i+1)*dash_length, y1), color, 2, line_type)
            # Bottom edge
            cv2.line(frame, (x1 + i*dash_length, y2), 
                     (x1 + (i+1)*dash_length, y2), color, 2, line_type)
            # Left edge
            cv2.line(frame, (x1, y1 + i*dash_length), 
                     (x1, y1 + (i+1)*dash_length), color, 2, line_type)
            # Right edge
            cv2.line(frame, (x2, y1 + i*dash_length), 
                     (x2, y1 + (i+1)*dash_length), color, 2, line_type)
    
    # Draw label background
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1 + 4), color, -1, line_type)
    
    # Draw label text
    cv2.putText(frame, label, (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, line_type)

def draw_hud(frame, detections, last_cmd, fps=0, best_conf=0, best_class="", color_detected=False, red_color_detected=False):
    # Show all detected classes and their counts
    det_text = " | ".join([f"{k}:{v}" for k, v in detections.items() if v > 0])
    
    # Add detection indicators
    indicators = ""
    if color_detected:
        indicators += " [GREEN-COLOR]"
    if red_color_detected:
        indicators += " [RED-COLOR]"
    
    hud = f"Detections: {det_text}{indicators}  Last CMD: {last_cmd}  Best: {best_class} {best_conf:.2f}  FPS: {fps:.1f}"
    
    # Use color background based on detection
    if color_detected:
        bg_color = (0, 80, 0)  # Dark green
    elif red_color_detected:
        bg_color = (0, 0, 80)  # Dark red
    else:
        bg_color = (0, 0, 0)   # Black
    
    cv2.putText(frame, hud, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, bg_color, 6)
    cv2.putText(frame, hud, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)

def main():
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return
    
    print("Loading model...")
    model = YOLO(MODEL_PATH)
    class_names = model.names
    print(f"Model loaded! Classes: {class_names}")
    print(f" Thresholds: Green={CLASS_CONF[0]}, Red={CLASS_CONF[1]}, Right={CLASS_CONF[2]}, Left={CLASS_CONF[3]}")
    
    stream = ESP32Stream(ESP32_URL)
    print(f"Connecting to stream: {ESP32_URL}")
    
    # Wait for first frame
    timeout = 10
    start_time = time.time()
    print("Waiting for first frame...")
    while stream.read() is None:
        if time.time() - start_time > timeout:
            print("ERROR: Timeout waiting for stream.")
            stream.stop()
            return
        time.sleep(0.1)
    print("Stream connected!")
    
    last_command = None
    frame_count = 0
    start_time = time.time()
    detection_stats = {'Green_Light': 0, 'Red_Light': 0, 'Right_Turn': 0, 'Left_Turn': 0}
    frame_with_detection = 0
    
    print("\n Starting REAL-TIME detection with ULTRA STRICT filtering...")
    print("📌 Color detection: Only perfect traffic-light shapes")
    print("📌 Must be in upper 35% of frame")
    print("📌 Circularity > 0.8 (almost perfect circles)")
    print("📌 Minimum area: 300 pixels")
    print("📌 Aspect ratio: 0.85-1.15 (near square)")
    print("Press 'q' to quit, 's' to save frame, 'd' to toggle debug")
    
    debug_mode = True
    green_detection_count = 0
    red_detection_count = 0
    color_detection_active = False
    red_color_detection_active = False
    
    while True:
        frame = stream.read()
        if frame is None:
            time.sleep(0.05)
            continue
        
        frame_count += 1
        
        # Run YOLO inference (primary detection)
        results = model(frame, conf=0.001, verbose=False)[0]
        
        # Track best detection for each class
        best_per_class = {}
        class_detections = {'Green_Light': 0, 'Red_Light': 0, 'Right_Turn': 0, 'Left_Turn': 0}
        color_detection_active = False
        red_color_detection_active = False
        
        # Process YOLO detections
        for box in results.boxes:
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            name = class_names.get(class_id, 'unknown')
            
            if name not in COMMANDS:
                continue
            
            # Update detection history for smoothing
            if name in DETECTION_HISTORY:
                DETECTION_HISTORY[name].append(conf)
            
            # Track best per class
            if name not in best_per_class or conf > best_per_class[name]:
                best_per_class[name] = conf
            
            # Count detections above absolute minimum
            if conf >= ABSOLUTE_MIN_CONF:
                class_detections[name] = class_detections.get(name, 0) + 1
            
            # Get class-specific threshold
            min_conf = CLASS_CONF.get(class_id, 0.02)
            
            # Draw box if confidence meets threshold
            if conf >= min_conf:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                label = f"{name} {conf:.3f}"
                color = CLASS_COLORS.get(name, (255, 255, 255))
                is_red = (name == 'Red_Light')
                draw_box(frame, x1, y1, x2, y2, label, color, conf, False, is_red)
        
        # COLOR-BASED DETECTION - ONLY as emergency fallback
        # Only use if YOLO completely failed (no detection)
        if best_per_class.get('Green_Light', 0) < 0.005:
            green_bbox = detect_green_by_color(frame)
            if green_bbox is not None:
                x, y, w, h, conf = green_bbox
                # High confidence required for color detection
                if conf >= 0.25:  # Very high threshold
                    label = f"Green [COLOR] {conf:.2f}"
                    color = (0, 255, 0)
                    draw_box(frame, x, y, x+w, y+h, label, color, conf, True, False)
                    class_detections['Green_Light'] += 1
                    if conf > best_per_class.get('Green_Light', 0):
                        best_per_class['Green_Light'] = conf
                    color_detection_active = True
                    green_detection_count += 1
                    if debug_mode:
                        print(f" ULTRA-STRICT Green traffic light [COLOR]! (conf: {conf:.2f})")
        
        # RED color detection - only as emergency fallback
        if best_per_class.get('Red_Light', 0) < 0.005:
            red_bbox = detect_red_by_color(frame)
            if red_bbox is not None:
                x, y, w, h, conf = red_bbox
                # High confidence required for color detection
                if conf >= 0.25:  # Very high threshold
                    label = f"Red [COLOR] {conf:.2f}"
                    color = (0, 0, 255)
                    draw_box(frame, x, y, x+w, y+h, label, color, conf, True, True)
                    class_detections['Red_Light'] += 1
                    if conf > best_per_class.get('Red_Light', 0):
                        best_per_class['Red_Light'] = conf
                    red_color_detection_active = True
                    red_detection_count += 1
                    if debug_mode:
                        print(f" ULTRA-STRICT Red traffic light [COLOR]! (conf: {conf:.2f})")
        
        # Determine which command to send - GREEN PRIORITY
        command_to_send = None
        command_reason = ""
        
        # Priority 1: GREEN (only if it's a real detection)
        if class_detections.get('Green_Light', 0) > 0 and best_per_class.get('Green_Light', 0) >= 0.005:
            command_to_send = 'F'
            command_reason = f"Green_Light detected (conf: {best_per_class.get('Green_Light', 0):.3f})"
        
        # Priority 2: RED
        elif class_detections.get('Red_Light', 0) > 0 and best_per_class.get('Red_Light', 0) >= 0.008:
            command_to_send = 'S'
            command_reason = f"Red_Light detected (conf: {best_per_class.get('Red_Light', 0):.3f})"
        
        # Priority 3: Right_Turn
        elif class_detections.get('Right_Turn', 0) > 0 and best_per_class.get('Right_Turn', 0) >= 0.01:
            command_to_send = 'R'
            command_reason = f"Right_Turn detected (conf: {best_per_class.get('Right_Turn', 0):.3f})"
        
        # Priority 4: Left_Turn
        elif class_detections.get('Left_Turn', 0) > 0 and best_per_class.get('Left_Turn', 0) >= 0.01:
            command_to_send = 'L'
            command_reason = f"Left_Turn detected (conf: {best_per_class.get('Left_Turn', 0):.3f})"
        
        # Send command if we have one
        if command_to_send:
            if command_to_send != last_command:
                if send_command(command_to_send):
                    print(f" Sent: {command_to_send} ({command_reason})")
                    last_command = command_to_send
        
        # Update stats
        if len(results.boxes) > 0 or color_detection_active or red_color_detection_active:
            frame_with_detection += 1
            for class_name in class_detections:
                if class_detections[class_name] > 0:
                    detection_stats[class_name] += 1
        
        # Calculate FPS
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
        else:
            fps = 0
        
        # Get best raw confidence for display
        best_display_conf = max(best_per_class.values()) if best_per_class else 0
        best_display_class = max(best_per_class, key=best_per_class.get) if best_per_class else "none"
        
        draw_hud(frame, class_detections, last_command or 'none', fps, best_display_conf, best_display_class, color_detection_active, red_color_detection_active)
        
        # Show detection status
        green_detected = class_detections.get('Green_Light', 0) > 0
        red_detected = class_detections.get('Red_Light', 0) > 0
        
        # Green status
        green_status = f"GREEN: {class_detections.get('Green_Light', 0)}" if green_detected else "⚫ NO GREEN"
        cv2.putText(frame, green_status, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if green_detected else (128, 128, 128), 2)
        
        # Red status
        red_status = f" RED: {class_detections.get('Red_Light', 0)}" if red_detected else "⚫ NO RED"
        cv2.putText(frame, red_status, (10, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if red_detected else (128, 128, 128), 2)
        
        # Show detection methods
        y_offset = 110
        if color_detection_active:
            cv2.putText(frame, "ULTRA-STRICT COLOR (GREEN)", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            y_offset += 25
        if red_color_detection_active:
            cv2.putText(frame, "ULTRA-STRICT COLOR (RED)", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Show detection quality
        if color_detection_active or red_color_detection_active:
            cv2.putText(frame, "COLOR DETECTION (EMERGENCY)", (10, y_offset + 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 2)
        
        cv2.imshow("ESP32-CAM Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f'debug_frame_{frame_count}.jpg', frame)
            print(f"Frame {frame_count} saved")
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
    
    stream.stop()
    cv2.destroyAllWindows()
    
    print(f"\n Summary:")
    print(f"  - Processed {frame_count} frames")
    print(f"  - Frames with detections: {frame_with_detection} ({frame_with_detection/frame_count*100:.1f}%)")
    print(f"  - Detection stats: {detection_stats}")
    print(f"  - Green_Light detections: {detection_stats['Green_Light']}")
    print(f"  - Red_Light detections: {detection_stats['Red_Light']}")
    print(f"  - Green color detections (emergency): {green_detection_count}")
    print(f"  - Red color detections (emergency): {red_detection_count}")
    print(f"  - Last command sent: {last_command}")

if __name__ == '__main__':
    main()
    