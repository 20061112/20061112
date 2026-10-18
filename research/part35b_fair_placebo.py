"""第三十五部分補充：公平的安慰劑，跑全部設定（不挑）。

週線上升趨勢線只在多頭裡存在；原本的安慰劑從全部日子抽，會抽到空頭，對碰線不公平地有利。
這裡隨機日子只從「當時存在上升趨勢線、且策略在 1 倍」的日子抽（同次數、同出場規則），每組 50 次。
全部 16 組設定 × 4 個指數，抱 52 週、2 倍；同時列前後半差，避免只看挑出來的好組。
"""
import numpy as np
import pandas as pd
from part35_zigzag_long import Book, CFG, BOOKS

out = open("results_part35b.txt", "w")


def P(s=""):
    print(s); out.write(s + "\n"); out.flush()


P("第三十五部分補充：公平安慰劑（隨機日子只從存在上升趨勢線、策略 1 倍的日子抽），全部設定、抱 52 週、2 倍\n")
rng = np.random.default_rng(351)
allrows = []
for name, fut, ratio, start in BOOKS:
    Bk = Book(name, fut, ratio, start)
    b = Bk.ev(windows=False)
    P(f"{name}（基準 IRR {b['irr'] * 100:.1f}%）")
    for th, k, tol in CFG:
        trig, brk, touch, wbrk, line, lows = Bk.signal(th, k, tol)
        mask, cnt = Bk.hold_mask(trig, brk, 52 * 5)
        if cnt == 0:
            continue
        has_line = np.zeros(len(Bk.S), bool)
        for w in range(len(line)):
            d = Bk.wk_to_day[w]
            if d < len(Bk.S) and line[w] == line[w]:
                e = Bk.wk_to_day[w + 1] if w + 1 < len(line) else len(Bk.S)
                has_line[d:e] = True
        lm = mask & ~Bk.ON
        m = Bk.ev(windows=False, regime_arr=Bk.ON | lm, L_arr=np.where(Bk.ON, 3.0, np.where(lm, 2.0, 1.0)))
        cand = np.flatnonzero(has_line & ~Bk.ON & (np.arange(len(Bk.S)) > Bk.i0))
        vals = []
        for _ in range(50):
            t = np.zeros(len(Bk.S), bool); t[rng.choice(cand, min(cnt, len(cand)), replace=False)] = True
            mk, _ = Bk.hold_mask(t, brk, 52 * 5)
            l2 = mk & ~Bk.ON
            vals.append(Bk.ev(windows=False, regime_arr=Bk.ON | l2, L_arr=np.where(Bk.ON, 3.0, np.where(l2, 2.0, 1.0)))["irr"])
        vals = np.array(vals)
        pct = (vals < m["irr"]).mean() * 100
        d1, d2 = m["h1"] - b["h1"], m["h2"] - b["h2"]
        allrows.append(dict(idx=name, d=m["irr"] - b["irr"], pct=pct, both=d1 > 0.002 and d2 > 0.002, cnt=cnt))
        P(f"  th {th:.0%} {k}點 tol {tol:.0%}（{cnt:2d} 次）：IRR 差 {(m['irr'] - b['irr']) * 100:+5.1f}%｜前/後半 {d1 * 100:+5.1f}/{d2 * 100:+5.1f}%｜"
          f"公平安慰劑 {pct:3.0f}%")
    P("")
R = pd.DataFrame(allrows)
P("總結（每個指數全部有碰線的設定）")
for name, g in R.groupby("idx", sort=False):
    P(f"  {name:5s} {len(g):2d} 組：IRR 差中位 {g.d.median() * 100:+.1f}%｜公平安慰劑中位 {g.pct.median():.0f}%、≥ 90% 的組數 {(g.pct >= 90).sum()}｜"
      f"兩半都變好 {g.both.sum()}")
P(f"  全部 {len(R)} 組：公平安慰劑中位 {R.pct.median():.0f}%、≥ 90% 的 {(R.pct >= 90).mean() * 100:.0f}%（純隨機約 10%）；兩半都變好 {R.both.mean() * 100:.0f}%")
out.close()
