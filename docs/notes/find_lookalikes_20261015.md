# find: telling look-alikes apart (2026-10-15)

Experiment notes moved out of CLAUDE.md. Content unchanged.

grounding-dino scores "a bag" and hardly the colour: at the dressroom, `find::black bag` boxed the beige bag
(0.45 vs the black bag at 0.39–0.42, 2 of 3 live frames); with only the blue bag in view, every colour scored
0.83–0.90 on it. Two options, both off by default (`recognition.py`):

- `others='beige bag, blue bag'` — the other names go into the same detector request; each box goes to the name
  it matches best, and a box overlapping (IoU ≥ 0.5) a higher-scoring box of another name is dropped. No extra
  time. ask's options can be passed as they are (`검은색=black bag, …`; the asked name is skipped). Live: 9 / 9
  right on the arm camera (black bag 0.68–0.79). Weak spot: an absent object can still land on another thing of
  that colour (beige bag → the beige robe, 0.50). `find_grasp` (pick's wrist close-up) uses it softly
  (`others_strict=False`): a claimed box is kept when no other box of the asked name is left — up close the
  black bag scored "blue bag" 0.43–0.45 vs "black bag" 0.35–0.39 on the same box and pick failed in fine_move.
- `verify=True` (`verify_above`, default 0.75; 1 = every box, at most 3) — the QA VLM (`hri._qa_config`, the
  `vlm` connection) answers yes / no on each box's crop. ~0.8 s per box. Live: 9 / 9 right; on the head frame
  with only the blue bag, black / beige → not found. Was broken until 2026-10-15 (imported the deleted `qa.py`).

Scenario 8: `find::inputs={answer}, others="검은색=black bag, 베이지색=beige bag, 파란색=blue bag", verify=True`.
Captures and scripts: `kaair_real_capture/find_bag_20261015/`.
