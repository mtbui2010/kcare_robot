"""Helpers for place.py. Pure refactor of inline logic into named steps; flow
matches the original module 1:1."""

from kcare_robot.skills.arm import movej


def loc2text(loc):
    """Convert a `'caption@loc'` string into a Korean-readable display string."""
    splits = loc.split('@')
    return loc if len(splits) < 2 else '에'.join(splits[::-1])


def parse_place_input(inp):
    """Split `inp` on '>>' into `(target_loc, rev_loc)`. If `inp` is a non-str
    (e.g., already a pose) `target_loc` is None and `rev_loc` is `inp`."""
    if isinstance(inp, str):
        splits = inp.split('>>')
        if len(splits) == 2:
            return splits[0], splits[1]
        return None, splits[-1]
    return None, inp


def parse_card_loc(inp):
    """Split `inp` on '>>' the way collect_card/return_card do. Note: this
    keeps the original (slightly quirky) precedence — when the split has !=2
    parts, `target_loc` becomes `None` (not a tuple-fallback)."""
    splits = inp.split('>>')
    target_loc, rev_loc = (splits if len(splits) == 2 else None, splits[-1])
    return target_loc, rev_loc


def turn_signed(turn_angle, robot_mode):
    """Sign the turn angle for the current robot_mode: positive on left, negative
    on right. Returns `None` if `turn_angle` is `None`."""
    if turn_angle is None:
        return None
    return abs(turn_angle) if robot_mode == 'left' else -abs(turn_angle)


def turn_base(node, turn_angle):
    """Turn the base by `turn_angle`. No-op (returns success) when None."""
    if turn_angle is None:
        return {'isdone': True}
    return movej(node=node, dr0=turn_angle)
