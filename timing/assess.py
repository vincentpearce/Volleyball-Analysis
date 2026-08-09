"""Phase 4: the core timing metric. Jump apex = temporal midpoint of takeoff
and landing (not tracked from a peak in the ankle signal -- Phase 2 already
detects takeoff/landing directly, and the midpoint is more robust than peak-
finding on a noisy smoothed signal). Timing offset = contact vs. that apex.

Always emits an entry per jump, even when contact wasn't found -- the raw
numbers matter more than the label (CLAUDE.md's explicit requirement), and a
missing contact frame is itself informative, not something to silently drop.
"""


def assess_hit(jump, fps, tolerance_ms):
    takeoff, landing = jump["takeoff_frame"], jump["landing_frame"]
    apex_frame = (takeoff + landing) / 2
    contact = jump.get("contact")

    result = {
        "takeoff_frame": takeoff,
        "landing_frame": landing,
        "landing_confirmed": jump["landing_confirmed"],
        "apex_frame": apex_frame,
        "contact_frame": None,
        "timing_offset_ms": None,
        "verdict": None,
    }
    if contact is None:
        return result

    contact_frame = contact["contact_frame"]
    offset_frames = contact_frame - apex_frame
    offset_ms = offset_frames / fps * 1000

    if abs(offset_ms) <= tolerance_ms:
        verdict = "on_time"
    elif offset_ms < 0:
        verdict = "early"
    else:
        verdict = "late"

    result.update({
        "contact_frame": contact_frame,
        "timing_offset_ms": offset_ms,
        "verdict": verdict,
    })
    return result


def assess_all(hitters, fps, tolerance_ms):
    results = []
    for hitter in hitters:
        for jump in hitter["jumps"]:
            hit = assess_hit(jump, fps, tolerance_ms)
            hit["track_id"] = hitter["track_id"]
            results.append(hit)
    return results
