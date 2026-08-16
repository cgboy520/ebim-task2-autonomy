# Room-scene official verdicts, 2026-08-05

Written by the organizers' own `scripts/evaluation/task2` service
(`/isaac/eval_camera/evaluate`) on the L4 rig, room scene
(`scene_room.py --record --headless`), target at its stock slot.

| file stamp | gated IoU | case | note |
|---|---|---|---|
| `20260805_112808_252802` | **0.6939** | `both_liner_dominant` | accepted at attempt 1/5, scene left in its scored state |

Decoded with `evidence/lay_geometry.py`:

```
   pad     56.0 x  63.0 px =   89.5 x 100.6 mm
   target  63.0 x  59.0 px =  100.6 x  94.2 mm  -> plate turned ~42 deg
   centre offset +2.4 , -12.8 mm
   if the pad were perfectly centred:      0.8384
```

Chain telemetry for the same attempt: 94.3 mm laid at a 20.3 mm width,
centred on the plate at the arc's release point; the tray-lay ramp then
moved lay and plate together by 50 mm and 42 deg.

Pre-fix room record: 0.3522 / 0.1928 / 0.0000, plus two wrong-face lays
before the orientation fix.
