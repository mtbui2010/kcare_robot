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

