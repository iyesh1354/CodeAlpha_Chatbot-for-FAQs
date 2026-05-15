"""
SORT: Simple Online and Realtime Tracking
Bewley et al., 2016  —  https://arxiv.org/abs/1602.00763

Included directly so no extra pip install is needed.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


# ─── Kalman Filter ─────────────────────────────────────────────────────────────

class KalmanBoxTracker:
    """
    Tracks a single bounding box using a Kalman Filter.
    State: [cx, cy, area, ratio, dcx, dcy, darea]
    """
    count = 0

    def __init__(self, bbox):
        # State transition matrix
        self.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1],
        ], dtype=float)

        # Measurement matrix
        self.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0],
        ], dtype=float)

        # Covariance matrices
        self.R = np.diag([1., 1., 10., 10.])
        self.P = np.diag([10., 10., 10., 10., 1e4, 1e4, 1e4])
        self.Q = np.diag([1., 1., 1., 1., .01, .01, .0001])

        self.x = np.zeros((7, 1))
        z = _bbox_to_z(bbox)
        self.x[:4] = z

        self.time_since_update = 0
        self.id    = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits        = 0
        self.hit_streak  = 0
        self.age         = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        z = _bbox_to_z(bbox)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    def predict(self):
        if self.x[6] + self.x[2] <= 0:
            self.x[6] = 0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(_z_to_bbox(self.x))
        return self.history[-1]

    def get_state(self):
        return _z_to_bbox(self.x)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _bbox_to_z(bbox):
    """[x1,y1,x2,y2] → [cx,cy,area,ratio]"""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.
    cy = bbox[1] + h / 2.
    return np.array([[cx], [cy], [w * h], [w / float(h)]])


def _z_to_bbox(x, score=None):
    """[cx,cy,area,ratio] → [x1,y1,x2,y2(,score)]"""
    w = np.sqrt(abs(x[2] * x[3]))
    h = x[2] / w if w != 0 else 0
    box = [x[0] - w/2., x[1] - h/2., x[0] + w/2., x[1] + h/2.]
    if score is None:
        return np.array(box).reshape((1, 4))
    return np.array([box[0], box[1], box[2], box[3], score]).reshape((1, 5))


def _iou_matrix(dets, trks):
    """Compute IoU between every detection and every tracker."""
    iou_mat = np.zeros((len(dets), len(trks)), dtype=float)
    for d, det in enumerate(dets):
        for t, trk in enumerate(trks):
            iou_mat[d, t] = _single_iou(det, trk)
    return iou_mat


def _single_iou(bb1, bb2):
    xx1 = max(bb1[0], bb2[0]); yy1 = max(bb1[1], bb2[1])
    xx2 = min(bb1[2], bb2[2]); yy2 = min(bb1[3], bb2[3])
    w = max(0., xx2 - xx1); h = max(0., yy2 - yy1)
    inter = w * h
    area1 = (bb1[2]-bb1[0]) * (bb1[3]-bb1[1])
    area2 = (bb2[2]-bb2[0]) * (bb2[3]-bb2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.


def _associate(detections, trackers, iou_threshold):
    if len(trackers) == 0:
        return (np.empty((0, 2), dtype=int),
                np.arange(len(detections)),
                np.empty((0,), dtype=int))

    iou_mat = _iou_matrix(detections, trackers)
    row_ind, col_ind = linear_sum_assignment(-iou_mat)
    matched_indices  = np.stack([row_ind, col_ind], axis=1)

    unmatched_dets  = [d for d in range(len(detections))
                       if d not in matched_indices[:, 0]]
    unmatched_trks  = [t for t in range(len(trackers))
                       if t not in matched_indices[:, 1]]

    matches = [m for m in matched_indices
               if iou_mat[m[0], m[1]] >= iou_threshold]

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.array(matches)

    return matches, np.array(unmatched_dets), np.array(unmatched_trks)


# ─── SORT ─────────────────────────────────────────────────────────────────────

class Sort:
    """
    Simple Online Realtime Tracking (SORT).

    Parameters
    ----------
    max_age       : frames a track survives without a detection match
    min_hits      : minimum hits before a track is confirmed
    iou_threshold : IoU threshold for matching
    """

    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age       = max_age
        self.min_hits      = min_hits
        self.iou_threshold = iou_threshold
        self.trackers      = []
        self.frame_count   = 0
        KalmanBoxTracker.count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        Parameters
        ----------
        dets : ndarray shape (N, 5) — [x1, y1, x2, y2, confidence]

        Returns
        -------
        ndarray shape (M, 5) — [x1, y1, x2, y2, track_id]
        """
        self.frame_count += 1

        # Predict new positions for all existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = _associate(
            dets[:, :4] if len(dets) else np.empty((0, 4)),
            trks[:, :4] if len(trks) else np.empty((0, 4)),
            self.iou_threshold,
        )

        # Update matched trackers
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[i, :]))

        # Collect active tracks
        ret = []
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1 and
                    (trk.hit_streak >= self.min_hits or
                     self.frame_count <= self.min_hits)):
                ret.append(np.concatenate((d, [trk.id + 1])))
            if trk.time_since_update > self.max_age:
                self.trackers.remove(trk)

        return np.array(ret) if ret else np.empty((0, 5))