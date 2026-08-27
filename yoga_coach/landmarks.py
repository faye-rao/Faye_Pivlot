"""Landmark naming and the per-frame skeleton container.

MediaPipe Pose returns 33 landmarks in a fixed order.  Referring to them by
index makes the pose definitions unreadable, so everything downstream uses the
names below.  A :class:`Skeleton` is one frame's worth of landmarks plus the
helpers the checks need (scale normalisation, visibility gating, left/right
mirroring).
"""

from __future__ import annotations

from dataclasses import dataclass

from .geometry import Point, distance, midpoint

#: Landmark names in MediaPipe Pose order.  Index == position in this tuple.
LANDMARK_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)

NAME_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(LANDMARK_NAMES)}

#: Segments drawn by the renderer.  Torso and limbs only -- the face mesh
#: points add clutter without helping anyone fix their alignment.
SKELETON_EDGES: tuple[tuple[str, str], ...] = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_heel"),
    ("left_heel", "left_foot_index"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_heel"),
    ("right_heel", "right_foot_index"),
)

#: Landmarks below this visibility score are treated as "not seen".  Checks
#: that need them report ``None`` instead of a bogus angle.
DEFAULT_MIN_VISIBILITY = 0.5

#: The body parts framing is judged on, each existing as a left/right pair.
#: Face landmarks are excluded -- they stay visible even when the legs are out
#: of shot, which is exactly the case worth catching.
BODY_PARTS: tuple[str, ...] = (
    "shoulder",
    "elbow",
    "wrist",
    "hip",
    "knee",
    "ankle",
    "heel",
    "foot_index",
)

#: Chinese names for the parts, for telling someone what is out of frame.
PART_NAMES_ZH: dict[str, str] = {
    "shoulder": "肩膀",
    "elbow": "手肘",
    "wrist": "手腕",
    "hip": "髋部",
    "knee": "膝盖",
    "ankle": "脚踝",
    "heel": "脚跟",
    "foot_index": "脚尖",
}


def mirror_name(name: str) -> str:
    """``left_knee`` -> ``right_knee`` and vice versa."""
    if name.startswith("left_"):
        return "right_" + name[len("left_") :]
    if name.startswith("right_"):
        return "left_" + name[len("right_") :]
    return name


@dataclass
class Skeleton:
    """One frame of body landmarks in normalised image coordinates."""

    points: dict[str, Point]
    min_visibility: float = DEFAULT_MIN_VISIBILITY

    @classmethod
    def from_list(
        cls,
        landmarks,
        min_visibility: float = DEFAULT_MIN_VISIBILITY,
    ) -> "Skeleton":
        """Build from anything indexable with ``.x/.y/.z/.visibility`` fields.

        Accepts MediaPipe's ``NormalizedLandmark`` objects as well as plain
        tuples/dicts, which keeps the tests free of a MediaPipe dependency.
        """
        points: dict[str, Point] = {}
        for i, name in enumerate(LANDMARK_NAMES):
            if i >= len(landmarks):
                break
            lm = landmarks[i]
            if isinstance(lm, Point):
                points[name] = lm
                continue
            if isinstance(lm, dict):
                points[name] = Point(
                    float(lm["x"]),
                    float(lm["y"]),
                    float(lm.get("z", 0.0)),
                    float(lm.get("visibility", 1.0)),
                )
                continue
            points[name] = Point(
                float(lm.x),
                float(lm.y),
                float(getattr(lm, "z", 0.0) or 0.0),
                float(getattr(lm, "visibility", 1.0) or 0.0),
            )
        return cls(points=points, min_visibility=min_visibility)

    def get(self, name: str) -> Point | None:
        """Landmark by name, or ``None`` when it is missing or barely visible.

        Names of the form ``mid_<part>`` are synthesised on the fly from the
        left and right landmark of that part, so checks can talk about
        ``mid_hip`` or ``mid_shoulder`` as if they were real landmarks.
        """
        if name.startswith("mid_"):
            part = name[len("mid_") :]
            return self.mid("left_" + part, "right_" + part)
        point = self.points.get(name)
        if point is None or point.visibility < self.min_visibility:
            return None
        return point

    def require(self, *names: str) -> list[Point] | None:
        """All of ``names`` at once, or ``None`` if any one is unusable."""
        out: list[Point] = []
        for name in names:
            point = self.get(name)
            if point is None:
                return None
            out.append(point)
        return out

    def mid(self, left: str, right: str) -> Point | None:
        """Midpoint of a left/right pair, falling back to whichever is visible.

        Requiring both would make every mid-body point unavailable in a
        side-on pose, where the far side is occluded by the near one.  That
        takes the torso length with it -- and with it every distance measured
        in torso lengths -- so Downward Dog and Plank could not be scored at
        all from their own recommended camera angle.

        Using the one visible side is a good approximation there: seen from
        the side, the near shoulder and the body's midline project to nearly
        the same place.  Checks that genuinely need both sides (levelling the
        shoulders, levelling the hips) ask for the landmarks directly and
        still correctly report "not measurable".
        """
        a = self.get(left)
        b = self.get(right)
        if a is not None and b is not None:
            return midpoint(a, b)
        return a if a is not None else b

    def torso_length(self) -> float | None:
        """Shoulder-centre to hip-centre distance, the unit of scale.

        Every positional check is expressed as a multiple of this so that
        advice does not change when you step closer to the camera.
        """
        shoulders = self.mid("left_shoulder", "right_shoulder")
        hips = self.mid("left_hip", "right_hip")
        if shoulders is None or hips is None:
            return None
        length = distance(shoulders, hips)
        return length if length > 1e-6 else None

    def missing_parts(self) -> list[str]:
        """Body parts where *neither* side can be seen.

        Counting parts rather than landmarks is the whole point.  MediaPipe's
        visibility score answers "is this landmark unoccluded", not "is it in
        frame", and in any side-on pose -- Downward Dog, Plank, Chair -- the
        entire far side of the body is hidden behind the near side.  Scoring
        each landmark separately therefore reports a perfectly framed Downward
        Dog as half out of shot.  A knee is in frame if *either* knee is
        visible; a body that is genuinely cropped loses both.

        Face landmarks are excluded: they are almost always visible and would
        mask a body whose legs are out of shot.
        """
        missing = []
        for part in BODY_PARTS:
            if self.get(f"left_{part}") is None and self.get(f"right_{part}") is None:
                missing.append(part)
        return missing

    def coverage(self) -> float:
        """Fraction of the body parts (shoulders down) that are in frame.

        See :meth:`missing_parts` for why this counts parts, not landmarks.
        """
        return 1.0 - len(self.missing_parts()) / len(BODY_PARTS)
