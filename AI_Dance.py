import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils


def normalize_pose(ref_points, live_points):
    # Use shoulders and hips for scaling and centering
    ref_shoulder = np.mean([ref_points[11][:2], ref_points[12][:2]], axis=0)
    ref_hip = np.mean([ref_points[23][:2], ref_points[24][:2]], axis=0)
    live_shoulder = np.mean([live_points[11][:2], live_points[12][:2]], axis=0)
    live_hip = np.mean([live_points[23][:2], live_points[24][:2]], axis=0)

    ref_torso = np.linalg.norm(ref_shoulder - ref_hip)
    live_torso = np.linalg.norm(live_shoulder - live_hip)
    scale = live_torso / ref_torso if ref_torso > 0 else 1

    ref_center = (ref_shoulder + ref_hip) / 2
    live_center = (live_shoulder + live_hip) / 2

    normalized = []
    for pt in ref_points:
        pt_xy = np.array(pt[:2])
        pt_xy = (pt_xy - ref_center) * scale + live_center
        normalized.append([pt_xy[0], pt_xy[1], pt[2]])
    return normalized

def extract_keypoints(results):
    landmarks = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z])
    return landmarks

def pose_similarity(ref_points, live_points):
    if len(ref_points) != len(live_points) or len(ref_points) == 0:
        return 0
    ref = np.array(ref_points).flatten()
    live = np.array(live_points).flatten()
    ref = ref / np.linalg.norm(ref)
    live = live / np.linalg.norm(live)
    return np.dot(ref, live)

def highlight_wrong_joints(frame, ref_points, live_points, threshold=0.15):
    h, w, _ = frame.shape
    for r, l in zip(ref_points, live_points):
        lx, ly = int(l[0]*w), int(l[1]*h)
        dist = np.linalg.norm(np.array(r[:2]) - np.array(l[:2]))
        color = (0, 255, 0) if dist < threshold else (0, 0, 255)
        cv2.circle(frame, (lx, ly), 6, color, -1)

# ---------- Extract reference poses ----------
ref_cap = cv2.VideoCapture("C:\\Users\\Bhagya\\Desktop\\Hackathons\\MSME\\AI_Dance\\Dance2.mp4")
reference_poses, reference_frames = [], []
frame_counter, step_interval = 0, 15

while ref_cap.isOpened():
    ret, frame = ref_cap.read()
    if not ret:
        break
    frame_counter += 1
    if frame_counter % step_interval != 0:
        continue

    frame = cv2.resize(frame, (480, 360))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    if results.pose_landmarks:
        reference_poses.append(extract_keypoints(results))
        ref_frame = frame.copy()
        mp_drawing.draw_landmarks(ref_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        reference_frames.append(ref_frame)

ref_cap.release()
if len(reference_poses) == 0:
    print("❌ No reference poses extracted.")
    exit()

print(f"✅ Extracted {len(reference_poses)} key poses")

# ---------- Real-time coaching ----------
live_cap = cv2.VideoCapture(0)
pose_index, hold_counter, hold_required = 0, 0, 30  # need 30 frames steady (~1 sec)

while live_cap.isOpened() and pose_index < len(reference_poses):
    ret, frame_live = live_cap.read()
    if not ret:
        break

    frame_live = cv2.resize(frame_live, (480, 360))
    rgb_live = cv2.cvtColor(frame_live, cv2.COLOR_BGR2RGB)
    results_live = pose.process(rgb_live)

    ref_points = reference_poses[pose_index]
    if results_live.pose_landmarks:
        live_points = extract_keypoints(results_live)
        score = pose_similarity(ref_points, live_points)
        score_percent = int(score * 100)

        highlight_wrong_joints(frame_live, ref_points, live_points)
        cv2.putText(frame_live, f"Match: {score_percent}%", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Only advance if pose is held long enough
        if score > 0.92:
            hold_counter += 1
            if hold_counter >= hold_required:
                pose_index += 1
                hold_counter = 0
        else:
            hold_counter = 0

        normalized_ref_points = normalize_pose(ref_points, live_points)
        # Draw normalized skeleton on blank image
        blank = np.zeros((360, 480, 3), dtype=np.uint8)
        for lm in normalized_ref_points:
            x, y = int(lm[0]*480), int(lm[1]*360)
            cv2.circle(blank, (x, y), 6, (255, 255, 255), -1)
        ref_frame = blank
    else:
        # If no pose detected, show a blank reference frame
        ref_frame = np.zeros((360, 480, 3), dtype=np.uint8)

    # Side-by-side display
    combined = cv2.hconcat([ref_frame, frame_live])
    cv2.imshow("Dance Tutor (Hold pose to advance)", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):  # Press 's' to skip to next pose
        pose_index += 1
        hold_counter = 0

live_cap.release()
cv2.destroyAllWindows()
