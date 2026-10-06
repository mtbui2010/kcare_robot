"""KCare-specific helpers, moved out of ``robot_agent.utils``.

Everything here depends on this robot's conventions: the ``ENV`` location
schema (``room@furniture`` names with ``loc`` / ``height`` / ``label`` specs),
the lift geometry, the head camera intrinsics, and the Korean announcement
phrasing used by pick / place / move skills. Generic helpers (``text2voice``,
``exception_handler``, ``run_parallel_check``, …) stay in ``robot_agent.utils``.
"""

import os

import numpy as np

from robot_agent.env_names import aliases_of, resolve_env_name
from robot_agent.env_names import normalize as normalize_env_name
from robot_agent.skill_configs import EN2KR, ENV, KR2EN, LIFT_CONFIGS, MOBILE_CONFIGS
from robot_agent.utils import text2voice


# ---------------------------------------------------------------------------
# ENV lookups (room@furniture location schema)
# ---------------------------------------------------------------------------

def env_key(env_name, ENV=ENV):
    """Canonical ENV key for *env_name* (a key, an alias, or a ``@`` segment of
    either), or None. A name matching several locations raises
    ``AmbiguousLocation`` when ``MOBILE_CONFIGS['strict_loc']`` is set, else the
    first match is used with a warning."""
    return resolve_env_name(env_name, ENV, strict=bool(MOBILE_CONFIGS.get('strict_loc', False)))


def get_env_specs(env_name, ENV, recursive=False):
    """Specs dict for *env_name* from *ENV* (resolved through :func:`env_key`,
    so aliases work), matching progressively shorter ``a@b@c`` suffixes when
    *recursive*. Returns ``{}`` when unknown."""
    import copy
    from robot_agent.env_names import AmbiguousLocation
    try:
        if not env_name:
            return {}
        key = env_key(env_name, ENV)
        if key is not None:
            return copy.deepcopy(ENV[key])
        if recursive:
            return get_env_specs('@'.join(env_name.split('@')[1:]), ENV=ENV)
        return {}
    except AmbiguousLocation:
        raise       # strict mode: fail the skill instead of guessing a place
    except Exception:
        return {}


def get_closest_loc(node, ENV, threshold=0.6):
    """Name of the configured location nearest the mobile base, or None when
    the base is farther than *threshold* metres from all of them."""
    # The mobile_pose agent hands back {'pose': [x, y, rz]} here (also seen: a
    # ROS Pose, or a flat {'x', 'y'}); reading ret['x'] raised KeyError 'x'.
    from robot_agent.core.planning.loop import _read_robot_xy
    xy = _read_robot_xy(node)
    if xy is None:
        return None
    x, y = xy

    valid_locs = {k: [v['loc']['x'] - x, v['loc']['y'] - y]
                  for k, v in ENV.items()
                  if isinstance(v, dict) and isinstance(v.get('loc'), dict) and 'x' in v['loc']}
    if not valid_locs:
        return None
    dd = np.linalg.norm(np.array(list(valid_locs.values())), axis=-1)
    argmin = int(np.argmin(dd))
    if dd[argmin] > threshold:
        return None
    return list(valid_locs.keys())[argmin]


def get_lift_height(env, robot_mode):
    """Lift height to work at *env*: its configured height (else the mode's
    home height) plus a 10 cm clearance."""
    return env.get('height', LIFT_CONFIGS['home'][robot_mode]) + 0.1


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
# Pixel↔camera-frame conversion (Ixy2xyz / calc_normalvector / …) comes from
# ``visionserve.utils`` — only the arm/tool-frame helper lives here.

def get_dtool_next_state(node, dtool):
    """Arm pose after moving *dtool* = (dx, dy, dz) in the tool frame.

    Reads the current ``robot_pose`` agent, rotates the tool-frame offset into
    the world frame, and returns ``(x, y, z, roll°, pitch°, yaw°)``.
    """
    x, y, z, roll, pitch, yaw = node.agents['robot_pose'].get()['pose']
    dx, dy, dz = dtool

    Rx = [[1, 0, 0],
          [0, np.cos(roll), -np.sin(roll)],
          [0, np.sin(roll), np.cos(roll)]]
    Ry = [[np.cos(pitch), 0, np.sin(pitch)],
          [0, 1, 0],
          [-np.sin(pitch), 0, np.cos(pitch)]]
    Rz = [[np.cos(yaw), -np.sin(yaw), 0],
          [np.sin(yaw), np.cos(yaw), 0],
          [0, 0, 1]]

    R = np.dot(Rz, np.dot(Ry, Rx))
    dxw, dyw, dzw = R @ [dx, dy, dz]
    return (x + dxw, y + dyw, z + dzw,
            roll * 180 / np.pi, pitch * 180 / np.pi, yaw * 180 / np.pi)


# ---------------------------------------------------------------------------
# Korean / English wording
# ---------------------------------------------------------------------------

def correct_noun(noun, lang='ko', translate=False):
    """Word-mapped translation of *noun* (KR2EN / EN2KR), falling back to the
    LLM only when *translate* is set."""
    noun = noun.strip().lower()
    word_dict = EN2KR if lang == 'ko' else KR2EN
    if noun in word_dict:
        return word_dict[noun]

    if not translate:
        return noun

    from robot_agent.utils import _llm
    return _llm().chat(prompt=f'''
                        Translate the follwing noun to {lang}, make it short, direct and native.
                        Ouput result only: {noun}
                        ''')


def correct_loc(loc, lang='korean'):
    """LLM translation of a location phrase into *lang*."""
    from robot_agent.utils import _llm
    return _llm().chat(prompt=f'''
                        Translate the follwing location to {lang}, make it short, direct and native.
                        Ouput result only: {loc}
                        ''')


def loc2text(loc, lang='ko'):
    """``'kitchen@table'`` → spoken form (reversed order, word-mapped)."""
    if not isinstance(loc, str):
        return None
    return ' '.join([correct_noun(el, lang=lang) for el in loc.split('@')[::-1]])


# ---------------------------------------------------------------------------
# Skill announcements (module-level state carries the pick/place subject
# between the "-ing" and "-ed" calls of one skill run)
# ---------------------------------------------------------------------------

caption_out = None


def announce_picking(caption_in, lang='ko'):
    global caption_out
    caption_out = correct_noun(caption_in, lang=lang)
    if lang == 'ko':
        text2voice(f'{caption_out} 잡을게요', run_thread=False, lang=lang)
    else:
        text2voice(f'grasping {caption_out}', run_thread=False, lang=lang)


def announce_picked(lang='ko'):
    global caption_out
    if lang == 'ko':
        text2voice(f'{caption_out} 잡았어요', run_thread=False, lang=lang)
    else:
        text2voice(f'{caption_out} grasped', run_thread=False, lang=lang)


loc_text_out = None


def announce_placing(inp=None, to_wipe=False, lang='ko'):
    if not isinstance(inp, str):
        return

    env = get_env_specs(inp, ENV=ENV)
    # A place named by one of its aliases is read back as the user said it;
    # the label (etri's Korean names for English keys) is for the key itself.
    label = None if normalize_env_name(inp) in aliases_of(env) else env.get('label', None)
    if inp is None and label is None:
        return

    global loc_text_out

    if to_wipe:
        text2voice('음료수를 쏟은 것을 닦을게요' if lang == 'ko' else "I'll wipe up the spilled drink", lang=lang)
    else:
        loc_text_out = loc2text(inp, lang=lang) if label is None else label
        if loc_text_out is not None:
            text2voice(f'{loc_text_out}에 내려 놓을게요' if lang == 'ko' else f'I will put down at {loc_text_out}',
                       run_thread=False, lang=lang)


def announce_placed(inp, to_wipe=False, lang='ko'):
    if not isinstance(inp, str):
        return

    global loc_text_out
    if to_wipe:
        text2voice('음료수를 쏟은 것을 닦았어요' if lang == 'ko' else 'spilled drink was wiped up.', lang=lang)
    else:
        text2voice(f'{loc_text_out}에 내려 놓았어요' if lang == 'ko' else f'Put down at {loc_text_out}',
                   run_thread=False, lang=lang)


def announce_moving(env_name):
    pass


def announce_arrived():
    pass


# ── Text on debug images (Korean too) ─────────────────────────────────────────
# cv2.putText only has ASCII glyphs: object names like 신라면 came out as
# "??????" on the detection log image. Non-ASCII text is drawn with Pillow and
# a Korean-capable font shipped in configs/fonts (OFL), so it works in any
# container without installing fonts.
_FONT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'fonts', 'NanumGothic-Bold.ttf')
_fonts: dict = {}


def put_text(im, text, org, scale=0.6, color=(255, 255, 0), thick=2, outline=(0, 0, 0)):
    """cv2.putText-like text with a dark outline, in place on an RGB/BGR uint8
    image. `org` is the left end of the baseline, as for cv2.putText."""
    import cv2
    text = str(text)
    if text.isascii():
        cv2.putText(im, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, outline, thick + 3)
        cv2.putText(im, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)
        return im
    try:
        from PIL import Image, ImageDraw, ImageFont
        size = max(10, int(round(34 * scale)))
        if size not in _fonts:
            _fonts[size] = ImageFont.truetype(_FONT_FILE, size)
        pil = Image.fromarray(im)
        ImageDraw.Draw(pil).text(org, text, font=_fonts[size], fill=tuple(int(c) for c in color),
                                 anchor='ls', stroke_width=max(1, thick), stroke_fill=tuple(outline))
        im[...] = np.asarray(pil)
    except Exception:                       # no Pillow / font: at least the ASCII part
        safe = text.encode('ascii', 'replace').decode()
        cv2.putText(im, safe, org, cv2.FONT_HERSHEY_SIMPLEX, scale, outline, thick + 3)
        cv2.putText(im, safe, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)
    return im
