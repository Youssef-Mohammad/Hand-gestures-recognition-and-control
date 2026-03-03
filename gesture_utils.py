import math
import numpy as np

# landmark indices following MediaPipe hand model
WRIST = 0
# for each finger we keep the indices of the mcp, pip and tip joints
FINGERS = {
    "thumb": (2, 3, 4),
    "index": (5, 6, 8),
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20),
}

def dist(a, b):
    """Euclidean distance between two 3D points."""
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)

def _angle(a, b, c):
    """Return the angle (in degrees) formed at point *b* by segments ba and bc."""
    ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
    bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
    norm = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm == 0:
        return 0.0
    cos_val = np.clip(np.dot(ba, bc) / norm, -1.0, 1.0)
    return math.degrees(math.acos(cos_val))


def finger_extended(landmarks, mcp, pip, tip):
    """Determine whether a finger is extended by checking the
    angle between the mcp->pip and pip->tip vectors.

    A straight finger produces an angle close to 180 degrees, so we
    treat anything above ~160° as extended. This is more robust than
    comparing distances to the wrist.
    """
    ang = _angle(landmarks[mcp], landmarks[pip], landmarks[tip])
    return ang > 160

def classify_gesture(landmarks):
    """Classify the hand pose into a small set of gestures.

    The function returns a tuple ``(gesture_name, states)`` where
    ``states`` is a dictionary indicating whether each finger is
    currently extended.  We rely on the more robust angle-based
    ``finger_extended`` above and then apply a few simple shape
    rules.  The set of recognized gestures is intentionally small so
    that the application logic remains predictable.
    """
    states = {}

    # compute extended/curl state for every finger
    for finger, indices in FINGERS.items():
        if finger == "thumb":
            # thumb tuple is (mcp, pip, tip)
            mcp, pip, tip = indices
        else:
            mcp, pip, tip = indices
        states[finger] = finger_extended(landmarks, mcp, pip, tip)

    # OPEN PALM: all five fingers are straight
    if all(states.values()):
        return "OPEN_PALM", states

    # FIST: none of the fingers are extended
    if not any(states.values()):
        return "FIST", states

    # THUMB UP / DOWN: only the thumb is straight and oriented vertically
    if states["thumb"] and not any(states[f] for f in ("index", "middle", "ring", "pinky")):
        # orientation check: thumb tip should lie above/below the mcp
        tip = landmarks[FINGERS["thumb"][2]]
        mcp = landmarks[FINGERS["thumb"][0]]
        if tip.y < mcp.y:  # y grows downward in MediaPipe coordinates
            return "THUMB_UP", states
        else:
            return "THUMB_DOWN", states

    # PEACE sign: index and middle extended, others curled
    if states["index"] and states["middle"] and not states["ring"] and not states["pinky"]:
        return "PEACE", states

    # fallback for anything we don't explicitly handle
    return "OTHER", states


def thumb_index_distance(landmarks):
    """Return the Euclidean distance between the thumb tip and index tip.

    This helper is used by the application for volume control and is not
    part of the gesture classifier itself, but lives here for
    convenience.
    """
    return dist(landmarks[4], landmarks[8])