"""yoga_coach -- real-time yoga posture feedback from a webcam.

The package is layered so that the interesting part (what counts as good
alignment) has no dependency on the camera stack:

``geometry`` / ``landmarks``
    coordinate maths and the per-frame skeleton
``metrics`` / ``checks`` / ``poses``
    measurements, one rule each, and the pose library built from them
``evaluator`` / ``session``
    scoring a frame, then smoothing a stream of frames into stable advice
``detector`` / ``render`` / ``console`` / ``voice`` / ``cli``
    MediaPipe, the on-screen overlay, and the command line

Only ``detector`` imports MediaPipe and only ``render``/``cli`` import
OpenCV, so the rules can be unit-tested against hand-built stick figures.
"""

from .checks import Check, CheckResult, Text
from .evaluator import PoseResult, evaluate, rank_poses
from .geometry import Point
from .landmarks import Skeleton
from .poses import POSES, POSES_BY_KEY, PoseSpec, get_pose
from .session import CoachSession, SessionState

__all__ = [
    "Check",
    "CheckResult",
    "CoachSession",
    "POSES",
    "POSES_BY_KEY",
    "Point",
    "PoseResult",
    "PoseSpec",
    "SessionState",
    "Skeleton",
    "Text",
    "evaluate",
    "get_pose",
    "rank_poses",
]

__version__ = "0.1.0"
