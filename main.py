import cv2 as cv
import numpy as np
import tensorflow as tf
import sys

# ── CONFIG ──────────────────
MODEL_PATH  = 'traffic_model.keras'
CLASS_NAMES = ['Background', 'Green_Light', 'Left_Turn', 'Red_Light', 'Right_Turn']
IMG_SIZE    = 64

# ── LOAD MODEL ──────────────
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("[*] Model loaded")
except Exception as e:
    print("Error loading model:", e)
    sys.exit()

# ── MASKS ──────────────────
def get_green_mask(frame):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    return cv.inRange(hsv, np.array([35,40,40]), np.array([95,255,255]))

def get_red_mask(frame):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    m1 = cv.inRange(hsv, np.array([0,100,100]), np.array([10,255,255]))
    m2 = cv.inRange(hsv, np.array([160,100,100]), np.array([180,255,255]))
    return cv.bitwise_or(m1, m2)

def get_sign_mask(frame):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    blue   = cv.inRange(hsv, np.array([100,100,50]), np.array([135,255,255]))
    white  = cv.inRange(hsv, np.array([0,0,180]), np.array([180,50,255]))
    yellow = cv.inRange(hsv, np.array([20,100,100]), np.array([35,255,255]))
    return cv.bitwise_or(cv.bitwise_or(blue, white), yellow)

def preprocess(frame):
    kernel = np.ones((5,5), np.uint8)
    r = cv.morphologyEx(get_red_mask(frame),   cv.MORPH_CLOSE, kernel)
    g = cv.morphologyEx(get_green_mask(frame), cv.MORPH_CLOSE, kernel)
    s = cv.morphologyEx(get_sign_mask(frame),  cv.MORPH_CLOSE, kernel)
    return r, g, s

# ── CNN ─────────────────────
def run_cnn(crop):
    img = cv.resize(crop, (IMG_SIZE, IMG_SIZE))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = np.expand_dims(img.astype(np.float32), axis=0)
    pred = model.predict(img, verbose=0)
    return int(np.argmax(pred)), float(np.max(pred))

# ── LANE DETECTION (RESTORED) ──────────
def detect_lanes(frame):
    gray  = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur  = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blur, 50, 150)

    h, w  = edges.shape
    mask  = np.zeros_like(edges)

    polygon = np.array([[
        (int(w * 0.05), h),
        (int(w * 0.95), h),
        (int(w * 0.60), int(h * 0.60)),
        (int(w * 0.40), int(h * 0.60)),
    ]])

    cv.fillPoly(mask, polygon, 255)
    masked = cv.bitwise_and(edges, mask)

    lines = cv.HoughLinesP(masked, 1, np.pi / 180,
                           threshold=50, minLineLength=80, maxLineGap=40)
    return lines

# ── MAIN ────────────────────
def main():
    cap = cv.VideoCapture('Videos/traffic4.mp4')

    if not cap.isOpened():
        print("Video not found")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.resize(frame, (640,480))
        display = frame.copy()

        # Masks
        rm, gm, sm = preprocess(frame)

        cv.imshow("RED mask", rm)
        cv.imshow("GREEN mask", gm)

        # Contours
        r_contours, _ = cv.findContours(rm, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        g_contours, _ = cv.findContours(gm, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        s_contours, _ = cv.findContours(sm, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # ── GREEN ──
        for c in g_contours:
            if cv.contourArea(c) < 20:
                continue
            x,y,w,h = cv.boundingRect(c)
            cv.rectangle(display,(x,y),(x+w,y+h),(0,255,0),2)
            cv.putText(display,"Green",(x,y-5),
                       cv.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

        # ── RED ──
        for c in r_contours:
            if cv.contourArea(c) < 20:
                continue
            x,y,w,h = cv.boundingRect(c)
            cv.rectangle(display,(x,y),(x+w,y+h),(0,0,255),2)
            cv.putText(display,"Red",(x,y-5),
                       cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)

        # ── SIGNS ──
        for c in s_contours:
            if cv.contourArea(c) < 300:
                continue
            x,y,w,h = cv.boundingRect(c)
            crop = frame[y:y+h, x:x+w]

            if crop.size == 0:
                continue

            cls, conf = run_cnn(crop)
            if cls != 0 and conf > 0.75:
                label = CLASS_NAMES[cls]
                cv.rectangle(display,(x,y),(x+w,y+h),(0,165,255),2)
                cv.putText(display,f"{label} {conf:.2f}",
                           (x,y-5),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,165,255),2)

        # ── LANES (RESTORED) ──
        lines = detect_lanes(frame)
        if lines is not None:
            for line in lines:
                x1,y1,x2,y2 = line[0]
                cv.line(display,(x1,y1),(x2,y2),(0,255,255),2)

        # Show
        cv.imshow("AI Traffic Detection", display)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()