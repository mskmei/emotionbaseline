# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np


POSE_LANDMARKS = 33
FACE_LANDMARKS = 468
HAND_LANDMARKS = 21
KEYPOINT_DIM = POSE_LANDMARKS * 4 + FACE_LANDMARKS * 3 + HAND_LANDMARKS * 3 * 2


def _landmarks_to_array(landmarks, count: int, dims: int, include_visibility: bool = False) -> np.ndarray:
    out = np.zeros((count, dims), dtype=np.float32)
    if landmarks is None:
        return out.reshape(-1)

    for i, lm in enumerate(landmarks.landmark[:count]):
        out[i, 0] = float(lm.x)
        out[i, 1] = float(lm.y)
        out[i, 2] = float(lm.z)
        if include_visibility and dims == 4:
            out[i, 3] = float(getattr(lm, "visibility", 0.0))
    return out.reshape(-1)


def holistic_result_to_vector(results) -> np.ndarray:
    pose = _landmarks_to_array(results.pose_landmarks, POSE_LANDMARKS, 4, include_visibility=True)
    face = _landmarks_to_array(results.face_landmarks, FACE_LANDMARKS, 3)
    left_hand = _landmarks_to_array(results.left_hand_landmarks, HAND_LANDMARKS, 3)
    right_hand = _landmarks_to_array(results.right_hand_landmarks, HAND_LANDMARKS, 3)
    vec = np.concatenate([pose, face, left_hand, right_hand], axis=0).astype(np.float32)
    if vec.shape[0] != KEYPOINT_DIM:
        raise RuntimeError(f"Unexpected keypoint dim {vec.shape[0]} != {KEYPOINT_DIM}")
    return vec


def iter_video_frames(
    video_path: Path,
    start: Optional[float] = None,
    end: Optional[float] = None,
    sample_fps: float = 10.0,
    max_frames: int = 0,
) -> Iterator[Tuple[int, float, np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps <= 0:
        source_fps = 30.0

    start_frame = max(int(round(float(start) * source_fps)), 0) if start is not None else 0
    end_frame = int(round(float(end) * source_fps)) if end is not None else None
    if end_frame is not None and end_frame <= start_frame:
        cap.release()
        raise RuntimeError(f"Invalid segment times for {video_path}: start={start}, end={end}")

    step = 1
    if sample_fps and sample_fps > 0:
        step = max(int(round(source_fps / float(sample_fps))), 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    yielded = 0
    try:
        while True:
            if end_frame is not None and frame_idx >= end_frame:
                break
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if (frame_idx - start_frame) % step == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                yield frame_idx, frame_idx / source_fps, frame_rgb
                yielded += 1
                if max_frames > 0 and yielded >= max_frames:
                    break
            frame_idx += 1
    finally:
        cap.release()


def iter_image_frames(frame_dir: Path, sample_fps: float = 0.0, max_frames: int = 0) -> Iterator[Tuple[int, float, np.ndarray]]:
    paths = sorted(list(frame_dir.glob("*.jpg")) + list(frame_dir.glob("*.jpeg")) + list(frame_dir.glob("*.png")))
    if not paths:
        raise RuntimeError(f"No image frames found in: {frame_dir}")

    step = 1
    if sample_fps and sample_fps > 0:
        source_fps = 30.0
        step = max(int(round(source_fps / float(sample_fps))), 1)

    yielded = 0
    for i, path in enumerate(paths):
        if i % step != 0:
            continue
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Failed to read image frame: {path}")
        yield i, float(i), cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        yielded += 1
        if max_frames > 0 and yielded >= max_frames:
            break


def extract_holistic_keypoints_from_frames(
    frame_iter: Iterator[Tuple[int, float, np.ndarray]],
    model_complexity: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    import mediapipe as mp

    keypoints = []
    timestamps = []
    with mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=int(model_complexity),
        smooth_landmarks=True,
        refine_face_landmarks=True,
        min_detection_confidence=float(min_detection_confidence),
        min_tracking_confidence=float(min_tracking_confidence),
    ) as holistic:
        for _frame_idx, timestamp, frame_rgb in frame_iter:
            frame_rgb.flags.writeable = False
            results = holistic.process(frame_rgb)
            keypoints.append(holistic_result_to_vector(results))
            timestamps.append(float(timestamp))

    if not keypoints:
        raise RuntimeError("MediaPipe received no frames")
    return np.stack(keypoints, axis=0).astype(np.float32), np.asarray(timestamps, dtype=np.float32)


def extract_holistic_keypoints_from_video(
    video_path: Path,
    start: Optional[float] = None,
    end: Optional[float] = None,
    sample_fps: float = 10.0,
    max_frames: int = 0,
    model_complexity: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    frames = iter_video_frames(video_path, start=start, end=end, sample_fps=sample_fps, max_frames=max_frames)
    return extract_holistic_keypoints_from_frames(frames, model_complexity=model_complexity)


def extract_holistic_keypoints_from_frame_dir(
    frame_dir: Path,
    sample_fps: float = 0.0,
    max_frames: int = 0,
    model_complexity: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    frames = iter_image_frames(frame_dir, sample_fps=sample_fps, max_frames=max_frames)
    return extract_holistic_keypoints_from_frames(frames, model_complexity=model_complexity)


def sample_keypoints_sequence(sequence: np.ndarray, num_tokens: int) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 2 or sequence.shape[1] != KEYPOINT_DIM:
        raise RuntimeError(f"Expected keypoints shape [T,{KEYPOINT_DIM}], got {sequence.shape}")
    if sequence.shape[0] == 0:
        raise RuntimeError("Cannot sample an empty keypoint sequence")

    if sequence.shape[0] >= num_tokens:
        idx = np.linspace(0, sequence.shape[0] - 1, num=int(num_tokens), dtype=np.int64)
        return sequence[idx].astype(np.float32)

    pad = np.zeros((int(num_tokens) - sequence.shape[0], KEYPOINT_DIM), dtype=np.float32)
    return np.concatenate([sequence, pad], axis=0).astype(np.float32)
