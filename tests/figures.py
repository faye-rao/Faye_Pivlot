"""Hand-built stick figures for the tests.

Working from synthetic skeletons instead of recorded video keeps the tests
fast, deterministic and free of any MediaPipe or camera dependency, and it
lets a test say exactly what is wrong with a posture ("front knee 40 degrees
past the ankle") instead of hoping a video clip still contains that mistake.

Coordinates follow MediaPipe's convention: normalised to ``[0, 1]`` with x to
the right and **y downwards**.  ``left_*`` landmarks are drawn at the smaller
x, as if you were looking at the practitioner's back; none of the checks care
which way round that is, they use magnitudes or per-side measurements.
"""

from __future__ import annotations

from yoga_coach.geometry import Point
from yoga_coach.landmarks import LANDMARK_NAMES, Skeleton

#: A relaxed, well-aligned standing figure -- Mountain pose.
STANDING: dict[str, tuple[float, float]] = {
    "nose": (0.500, 0.100),
    "left_eye_inner": (0.480, 0.090),
    "left_eye": (0.475, 0.090),
    "left_eye_outer": (0.470, 0.090),
    "right_eye_inner": (0.520, 0.090),
    "right_eye": (0.525, 0.090),
    "right_eye_outer": (0.530, 0.090),
    "left_ear": (0.455, 0.100),
    "right_ear": (0.545, 0.100),
    "mouth_left": (0.485, 0.125),
    "mouth_right": (0.515, 0.125),
    "left_shoulder": (0.440, 0.250),
    "right_shoulder": (0.560, 0.250),
    "left_elbow": (0.420, 0.380),
    "right_elbow": (0.580, 0.380),
    "left_wrist": (0.410, 0.500),
    "right_wrist": (0.590, 0.500),
    "left_pinky": (0.405, 0.535),
    "right_pinky": (0.595, 0.535),
    "left_index": (0.410, 0.540),
    "right_index": (0.590, 0.540),
    "left_thumb": (0.420, 0.530),
    "right_thumb": (0.580, 0.530),
    "left_hip": (0.460, 0.520),
    "right_hip": (0.540, 0.520),
    "left_knee": (0.460, 0.720),
    "right_knee": (0.540, 0.720),
    "left_ankle": (0.460, 0.920),
    "right_ankle": (0.540, 0.920),
    "left_heel": (0.455, 0.940),
    "right_heel": (0.545, 0.940),
    "left_foot_index": (0.470, 0.960),
    "right_foot_index": (0.530, 0.960),
}


def figure(
    base: dict[str, tuple[float, float]] | None = None,
    /,
    hidden: tuple[str, ...] = (),
    **moved: tuple[float, float],
) -> Skeleton:
    """Build a :class:`Skeleton` from ``base`` with some joints moved.

    ``figure(left_knee=(0.5, 0.7))`` starts from :data:`STANDING`; ``hidden``
    names landmarks the detector could not see, which get visibility 0.
    """
    joints = dict(base if base is not None else STANDING)
    joints.update(moved)
    points = {
        name: Point(x, y, 0.0, 0.0 if name in hidden else 1.0)
        for name, (x, y) in joints.items()
    }
    return Skeleton(points=points)


def mirrored(base: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
    """The same posture facing the other way in the frame.

    Flips x and swaps the left/right labels, which is exactly what the
    detector reports when the practitioner turns round.  Nothing about the
    advice should change -- see ``TestMirrorInvariance``.
    """
    out: dict[str, tuple[float, float]] = {}
    for name, (x, y) in base.items():
        if name.startswith("left_"):
            name = "right_" + name[len("left_") :]
        elif name.startswith("right_"):
            name = "left_" + name[len("right_") :]
        out[name] = (1.0 - x, y)
    return out


def as_landmark_list(skeleton: Skeleton) -> list[Point]:
    """Flatten a skeleton back into MediaPipe's fixed-order list."""
    return [
        skeleton.points.get(name, Point(0.0, 0.0, 0.0, 0.0))
        for name in LANDMARK_NAMES
    ]


# --------------------------------------------------------------------------
# Poses other than Mountain.  Each is laid out so every check lands inside its
# target band; the geometry is spelled out in the comments so a future change
# to a band can be checked against the drawing rather than guessed at.
# --------------------------------------------------------------------------

#: Warrior II with the *right* leg forward.  Front thigh horizontal and shin
#: vertical (a 90 degree knee), back leg straight, arms level at shoulder
#: height, torso stacked over the hips.
WARRIOR_II_RIGHT: dict[str, tuple[float, float]] = {
    **STANDING,
    "nose": (0.560, 0.320),
    "left_ear": (0.535, 0.340),
    "right_ear": (0.585, 0.340),
    "left_eye": (0.545, 0.315),
    "right_eye": (0.575, 0.315),
    "left_eye_inner": (0.550, 0.315),
    "right_eye_inner": (0.570, 0.315),
    "left_eye_outer": (0.540, 0.315),
    "right_eye_outer": (0.580, 0.315),
    "mouth_left": (0.552, 0.335),
    "mouth_right": (0.568, 0.335),
    "left_shoulder": (0.420, 0.380),
    "right_shoulder": (0.480, 0.380),
    "left_elbow": (0.285, 0.380),
    "right_elbow": (0.615, 0.380),
    "left_wrist": (0.150, 0.380),
    "right_wrist": (0.750, 0.380),
    "left_pinky": (0.130, 0.385),
    "right_pinky": (0.770, 0.385),
    "left_index": (0.128, 0.380),
    "right_index": (0.772, 0.380),
    "left_thumb": (0.140, 0.375),
    "right_thumb": (0.760, 0.375),
    "left_hip": (0.420, 0.580),
    "right_hip": (0.480, 0.580),
    # front (right) leg: thigh 0.16 horizontal, shin 0.22 vertical -> 90 deg
    "right_knee": (0.640, 0.580),
    "right_ankle": (0.640, 0.800),
    "right_heel": (0.615, 0.810),
    "right_foot_index": (0.700, 0.805),
    # back (left) leg: hip -> ankle in a straight line, knee at its midpoint
    "left_knee": (0.300, 0.690),
    "left_ankle": (0.180, 0.800),
    "left_heel": (0.150, 0.810),
    "left_foot_index": (0.235, 0.805),
}

#: A side-on Plank: shoulders, hips and ankles on one line, shoulders stacked
#: over the wrists.
PLANK_SIDE: dict[str, tuple[float, float]] = {
    **STANDING,
    "nose": (0.230, 0.470),
    "left_ear": (0.265, 0.478),
    "right_ear": (0.262, 0.482),
    "left_eye": (0.245, 0.468),
    "right_eye": (0.243, 0.472),
    "left_eye_inner": (0.250, 0.468),
    "right_eye_inner": (0.248, 0.472),
    "left_eye_outer": (0.240, 0.468),
    "right_eye_outer": (0.238, 0.472),
    "mouth_left": (0.235, 0.490),
    "mouth_right": (0.233, 0.494),
    "left_shoulder": (0.300, 0.500),
    "right_shoulder": (0.303, 0.504),
    "left_elbow": (0.300, 0.600),
    "right_elbow": (0.303, 0.604),
    "left_wrist": (0.300, 0.700),
    "right_wrist": (0.303, 0.704),
    "left_pinky": (0.320, 0.712),
    "right_pinky": (0.323, 0.716),
    "left_index": (0.325, 0.706),
    "right_index": (0.328, 0.710),
    "left_thumb": (0.310, 0.695),
    "right_thumb": (0.313, 0.699),
    "left_hip": (0.550, 0.550),
    "right_hip": (0.553, 0.554),
    "left_knee": (0.700, 0.580),
    "right_knee": (0.703, 0.584),
    "left_ankle": (0.850, 0.610),
    "right_ankle": (0.853, 0.614),
    "left_heel": (0.880, 0.605),
    "right_heel": (0.883, 0.609),
    "left_foot_index": (0.870, 0.660),
    "right_foot_index": (0.873, 0.664),
}

#: Downward Dog seen from the side: an inverted V with the hands and feet on
#: the floor at y=0.90 and the hips high at y=0.32.
#:
#: The derived angles, for when a band changes and this has to be re-checked:
#: hip (shoulder-hip-knee) 66 degrees, back (wrist-shoulder-hip) 164, legs
#: 177, arms 180 -- every check comfortably inside its band.
DOWN_DOG_SIDE: dict[str, tuple[float, float]] = {
    **STANDING,
    # Head hangs below the shoulders, between the arms.
    "nose": (0.255, 0.700),
    "left_ear": (0.288, 0.678),
    "right_ear": (0.285, 0.682),
    "left_eye": (0.268, 0.694),
    "right_eye": (0.265, 0.698),
    "left_eye_inner": (0.272, 0.694),
    "right_eye_inner": (0.269, 0.698),
    "left_eye_outer": (0.263, 0.694),
    "right_eye_outer": (0.260, 0.698),
    "mouth_left": (0.252, 0.716),
    "mouth_right": (0.249, 0.720),
    "left_shoulder": (0.300, 0.620),
    "right_shoulder": (0.303, 0.624),
    "left_elbow": (0.250, 0.760),
    "right_elbow": (0.253, 0.764),
    "left_wrist": (0.200, 0.900),
    "right_wrist": (0.203, 0.904),
    "left_pinky": (0.180, 0.910),
    "right_pinky": (0.183, 0.914),
    "left_index": (0.175, 0.905),
    "right_index": (0.178, 0.909),
    "left_thumb": (0.192, 0.898),
    "right_thumb": (0.195, 0.902),
    "left_hip": (0.520, 0.320),
    "right_hip": (0.523, 0.324),
    "left_knee": (0.680, 0.600),
    "right_knee": (0.683, 0.604),
    "left_ankle": (0.820, 0.880),
    "right_ankle": (0.823, 0.884),
    "left_heel": (0.860, 0.900),
    "right_heel": (0.863, 0.904),
    "left_foot_index": (0.760, 0.900),
    "right_foot_index": (0.763, 0.904),
}

#: Downward Dog with the heels well clear of the floor -- about 18cm of lift
#: on an adult, i.e. up on the balls of the feet.  The toes stay put, which is
#: what makes them usable as the floor reference.
DOWN_DOG_HEELS_UP: dict[str, tuple[float, float]] = {
    **DOWN_DOG_SIDE,
    "left_heel": (0.860, 0.760),
    "right_heel": (0.863, 0.764),
}

#: A side-on Upward Dog, hands at the left and feet at the right.  The floor is
#: again y=0.900: the wrists are on it, the arms are 0.340 long and vertical, so
#: the shoulder is at (0.180, 0.560).  The torso is 0.290 long and tilted 35
#: degrees back from vertical, putting the hip at (0.346, 0.798) -- 0.31 torso
#: lengths clear of the wrist->ankle floor line.  The legs are straight and
#: 1.7 torsos long, which fixes the ankle at (0.843, 0.850).
UP_DOG_SIDE: dict[str, tuple[float, float]] = {
    **STANDING,
    # chest open, gaze forward and slightly up
    "nose": (0.115, 0.492),
    "left_ear": (0.152, 0.502),
    "right_ear": (0.155, 0.506),
    "left_eye": (0.128, 0.486),
    "right_eye": (0.131, 0.490),
    "left_eye_inner": (0.134, 0.487),
    "right_eye_inner": (0.137, 0.491),
    "left_eye_outer": (0.122, 0.485),
    "right_eye_outer": (0.125, 0.489),
    "mouth_left": (0.120, 0.510),
    "mouth_right": (0.123, 0.514),
    # arms vertical, elbow at the midpoint -> straight
    "left_shoulder": (0.180, 0.560),
    "right_shoulder": (0.183, 0.564),
    "left_elbow": (0.180, 0.730),
    "right_elbow": (0.183, 0.734),
    "left_wrist": (0.180, 0.900),
    "right_wrist": (0.183, 0.904),
    "left_pinky": (0.205, 0.912),
    "right_pinky": (0.208, 0.916),
    "left_index": (0.212, 0.906),
    "right_index": (0.215, 0.910),
    "left_thumb": (0.192, 0.896),
    "right_thumb": (0.195, 0.900),
    # pelvis lifted clear of the mat
    "left_hip": (0.346, 0.798),
    "right_hip": (0.349, 0.802),
    "left_knee": (0.595, 0.824),
    "right_knee": (0.598, 0.828),
    "left_ankle": (0.843, 0.850),
    "right_ankle": (0.846, 0.854),
    # tops of the feet on the floor, so the heel rides above the toes
    "left_heel": (0.862, 0.868),
    "right_heel": (0.865, 0.872),
    "left_foot_index": (0.900, 0.896),
    "right_foot_index": (0.903, 0.900),
}


#: Landmarks the far side of the body loses in any side-on pose.  MediaPipe
#: reports them with low visibility because they are behind the near side --
#: which is normal framing, not a cropped body.
FAR_SIDE_OCCLUDED: tuple[str, ...] = (
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "left_pinky",
    "left_index",
    "left_thumb",
    "left_hip",
    "left_knee",
    "left_ankle",
    "left_heel",
    "left_foot_index",
)

#: Tree with the *left* leg standing and the right foot on the inner thigh.
TREE_LEFT: dict[str, tuple[float, float]] = {
    **STANDING,
    "left_hip": (0.460, 0.520),
    "right_hip": (0.540, 0.520),
    "left_knee": (0.460, 0.720),
    "left_ankle": (0.460, 0.920),
    # lifted leg: knee swung out to the side, foot high on the inner thigh
    "right_knee": (0.720, 0.640),
    "right_ankle": (0.520, 0.660),
    "right_heel": (0.505, 0.675),
    "right_foot_index": (0.550, 0.690),
    # hands in prayer at the chest
    "left_elbow": (0.380, 0.400),
    "right_elbow": (0.620, 0.400),
    "left_wrist": (0.490, 0.330),
    "right_wrist": (0.510, 0.330),
    "left_pinky": (0.492, 0.300),
    "right_pinky": (0.508, 0.300),
    "left_index": (0.494, 0.295),
    "right_index": (0.506, 0.295),
    "left_thumb": (0.496, 0.310),
    "right_thumb": (0.504, 0.310),
}
