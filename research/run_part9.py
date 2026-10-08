"""第九部分重現：python3 run_part9.py"""
from lowtf_breakout import lonbrk_tf, orb, summ as s1
from orb_fade import bars, fade, summ as s2

print("=== A. LONBRK 訊號，M1/M3/M5 進出場 ===")
for tf in (1, 3, 5):
    for ent, st in (("touch", "range"), ("close", "range"), ("close", "swing"), ("close", "atr")):
        print(f"M{tf} {ent:5s} {st:5s}", s1(lonbrk_tf(tf, ent, st)))
print("\n=== B. 開盤區間突破（順勢）===")
for name, oh in (("倫敦 10:00", (10, 0)), ("COMEX 15:20", (15, 20)), ("紐約 15:30", (15, 30)), ("美股 16:30", (16, 30))):
    for orm in (15, 30, 60):
        for tf, ent in ((1, "touch"), (3, "close"), (5, "close")):
            print(f"{name} OR{orm} M{tf} {ent}", s1(orb(oh, orm, tf, win_min=120, exit_h=19 if oh[0] < 16 else 22, entry=ent)))
print("\n=== C. 紐約開盤 OR15 假突破反向 ===")
for tf in (1, 3, 5, 15):
    d = bars(tf)
    sp = "2026-08-15" if tf < 15 else "2026-06-01"
    for mode in ("touch", "fail"):
        for tg in (True, False):
            print(f"M{tf} {mode} target={tg}", s2(fade(d, tf, (15, 30), 15, mode=mode, target=tg), sp))
