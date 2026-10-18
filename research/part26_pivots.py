"""第二十六部分：Pivot（樞紐點）—— 一開始就用多年度、兩個商品，標準規則不調參，附假價位對照。

資料：XAUUSD M1 2023/6 ~ 2026/10（data/XAUUSD_M1_2023_2026.csv）、NAS100 M1 2023/10 ~ 2026/10（三個檔案接起來）。
交易日 = broker 01:00 ~ 23:59（= MT5 日線）；pivot 用前一個交易日的高 / 低 / 收。
  經典：P = (H+L+C)/3，R1 = 2P−L，S1 = 2P−H，R2 = P+(H−L)，S2 = P−(H−L)，R3 = H+2(P−L)，S3 = L−2(H−P)
  Camarilla：rng = H−L；H3/L3 = C ± 1.1·rng/4，H4/L4 = C ± 1.1·rng/2
規則（全部是常見的教科書版本，參數一個都沒調；每個規則每天每個方向最多一筆；22:00 後不進場；當天收盤平倉）：
  F1 R1/S1 反轉：當天第一次從 P 那側碰到 R1 → 在 R1 做空，停損 R2，目標 P（S1 鏡像）
  F2 R2/S2 反轉：碰到 R2 → 做空，停損 R3，目標 R1
  B1 R1/S1 突破：M5 收盤站上 R1 → 收盤價做多，停損 P，目標 R2
  B2 R2/S2 突破：M5 收盤站上 R2 → 做多，停損 R1，目標 R3
  P1 P 回測：開盤在 P 之上，第一次回到 P → 在 P 做多，停損 S1，目標 R1（開在下方鏡像）
  C1 Camarilla H3/L3 反轉：碰到 H3 → 做空，停損 H4，目標前日收盤 C
  C2 Camarilla H4/L4 突破：M5 收盤站上 H4 → 做多，停損 H3，沒有目標（抱到收盤）
對照（安慰劑）：所有價位一起平移 ±0.25 × ATR10（兩個方向都做）再跑同樣規則。pivot 若真的有支撐壓力 / 突破意義，
真價位應該明顯比假價位好。成本：進場 K 棒點差。同一根 K 棒同時碰停損和目標 → 停損。
輸出：results_part26.txt
"""
import numpy as np
import pandas as pd
from load import load

SRC = {"XAUUSD": (["data/XAUUSD_M1_2023_2026.csv"], 0.01),
       "NAS100": (["data/NAS100_M1_2023_2024.csv", "data/NAS100_M1_2024_2025.csv", "data/NAS100_M1_2025_2026.csv"], 0.1)}
SIDES = {"F1": (-1, 1), "F2": (-1, 1), "C1": (-1, 1), "B1": (1, -1), "B2": (1, -1), "C2": (1, -1)}   # day_trades 每個清單的方向
RULES = ["F1", "F2", "B1", "B2", "P1", "C1", "C2"]
NAMES = {"F1": "R1/S1 反轉（停 R2，目標 P）", "F2": "R2/S2 反轉（停 R3，目標 R1）", "B1": "R1/S1 突破（停 P，目標 R2）",
         "B2": "R2/S2 突破（停 R1，目標 R3）", "P1": "P 回測（停 S1，目標 R1）", "C1": "Camarilla H3/L3 反轉（停 H4，目標 C）",
         "C2": "Camarilla H4/L4 突破（停 H3，抱到收盤）"}


def load_sym(sym):
    files, pt = SRC[sym]
    d = pd.concat([load(f) for f in files])
    d = d[~d.index.duplicated()].sort_index()
    d["sp"] = d.spread * pt
    d["day"] = (d.index - pd.Timedelta(hours=1)).normalize()
    return d


def levels(H, L, C, shift=0.0):
    P = (H + L + C) / 3
    lv = dict(P=P, R1=2 * P - L, S1=2 * P - H, R2=P + (H - L), S2=P - (H - L), R3=H + 2 * (P - L), S3=L - 2 * (H - P),
              H3=C + 1.1 * (H - L) / 4, L3=C - 1.1 * (H - L) / 4, H4=C + 1.1 * (H - L) / 2, L4=C - 1.1 * (H - L) / 2, C=C)
    return {k: v + shift for k, v in lv.items()}


def exit_trade(O, Hh, Ll, Cc, S, i, side, ep, stop, tgt, first_bar_entry):
    """i = 進場那根（觸價單：同根檢查停損；收盤單：i 是下一根）。回傳 R。"""
    risk = side * (ep - stop)
    if risk <= 0:
        return None
    if tgt is not None and side * (tgt - ep) <= 0:       # 進場價已經超過目標（跳空）→ 沒有目標，抱到收盤
        tgt = None
    hs = (Ll[i:] <= stop) if side == 1 else (Hh[i:] >= stop)
    ht = np.zeros_like(hs) if tgt is None else ((Hh[i:] >= tgt) if side == 1 else (Ll[i:] <= tgt))
    if first_bar_entry and tgt is not None:
        ht[0] = False                                     # 觸價進場那根不給目標
    js = np.argmax(hs) if hs.any() else len(hs); jt = np.argmax(ht) if ht.any() else len(ht)
    if js <= jt and js < len(hs):
        k = i + js
        px = stop if (first_bar_entry and js == 0) else (min(O[k], stop) if side == 1 else max(O[k], stop))
    elif jt < len(ht):
        px = tgt
    else:
        px = Cc[-1]
    return (side * (px - ep) - S[i]) / risk


def day_trades(g, lv, open_px):
    O, Hh, Ll, Cc, S = g.open.values, g.high.values, g.low.values, g.close.values, g.sp.values
    mins = ((g.index - g.day.iloc[0]).total_seconds().values // 60).astype(int)
    ok = mins < 22 * 60                                   # 22:00 前進場（mins 從 00:00 起算）
    prevC = np.r_[O[0], Cc[:-1]]
    m5 = np.where(((mins + 1) % 5 == 0) & ok)[0]
    out = {}

    def touch(level, frm):                                # 第一次從 frm 那側碰到 level
        if frm == "below":
            idx = np.where(ok & (Hh >= level) & (prevC < level))[0]
        else:
            idx = np.where(ok & (Ll <= level) & (prevC > level))[0]
        return idx[0] if len(idx) else None

    def fade(lv_name, stop_name, tgt_name, side):         # side = 進場方向
        i = touch(lv[lv_name], "below" if side == -1 else "above")
        if i is None:
            return None
        ep = max(O[i], lv[lv_name]) if side == -1 else min(O[i], lv[lv_name])
        return exit_trade(O, Hh, Ll, Cc, S, i, side, ep, lv[stop_name], lv[tgt_name], True)

    def brk(lv_name, stop_name, tgt_name, side):
        c = Cc[m5]
        w = m5[c > lv[lv_name]] if side == 1 else m5[c < lv[lv_name]]
        if not len(w) or w[0] + 1 >= len(Cc):
            return None
        i = w[0]
        return exit_trade(O, Hh, Ll, Cc, S, i + 1, side, Cc[i], lv[stop_name], None if tgt_name is None else lv[tgt_name], False)

    out["F1"] = [fade("R1", "R2", "P", -1), fade("S1", "S2", "P", 1)]
    out["F2"] = [fade("R2", "R3", "R1", -1), fade("S2", "S3", "S1", 1)]
    out["B1"] = [brk("R1", "P", "R2", 1), brk("S1", "P", "S2", -1)]
    out["B2"] = [brk("R2", "R1", "R3", 1), brk("S2", "S1", "S3", -1)]
    if open_px > lv["P"]:
        out["P1"] = [fade("P", "S1", "R1", 1)]
    elif open_px < lv["P"]:
        out["P1"] = [fade("P", "R1", "S1", -1)]
    out["C1"] = [fade("H3", "H4", "C", -1), fade("L3", "L4", "C", 1)]
    out["C2"] = [brk("H4", "H3", None, 1), brk("L4", "L3", None, -1)]
    return out


def run(d):
    days = [k for k, g in d.groupby("day") if len(g) >= 1000]
    G = {k: g for k, g in d.groupby("day") if k in set(days)}
    DD = pd.DataFrame({k: dict(o=g.open.iloc[0], h=g.high.max(), l=g.low.min(), c=g.close.iloc[-1]) for k, g in G.items()}).T
    tr = np.maximum(DD.h - DD.l, np.maximum((DD.h - DD.c.shift()).abs(), (DD.l - DD.c.shift()).abs()))
    atr = tr.rolling(10).mean().shift(1)
    rows = []
    for a, b in zip(days[:-1], days[1:]):
        if (b - a).days > 4 or not np.isfinite(atr[b]):
            continue
        prev = DD.loc[a]
        for kind, sh in (("real", 0.0), ("fake+", 0.25), ("fake-", -0.25)):
            lv = levels(prev.h, prev.l, prev.c, sh * atr[b])
            for rule, rs in day_trades(G[b], lv, DD.o[b]).items():
                for j, r in enumerate(rs):
                    if r is not None and np.isfinite(r):
                        side = SIDES[rule][j] if rule in SIDES else (1 if DD.o[b] > lv["P"] else -1)
                        rows.append(dict(day=b, kind=kind, rule=rule, side=side, R=r))
    return pd.DataFrame(rows)


def tst(x):
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 else np.nan


def pf(x):
    return x[x > 0].sum() / max(-x[x <= 0].sum(), 1e-12)


if __name__ == "__main__":
    out = open("results_part26.txt", "w")
    def P(s=""): print(s); out.write(s + "\n"); out.flush()
    P("第二十六部分：Pivot（經典 + Camarilla），標準規則不調參；每格 = 每筆 R（筆數）。假價位 = 價位平移 ±0.25 ATR10 的平均。")
    allT = {}
    for sym in SRC:
        T = run(load_sym(sym)); T["year"] = T.day.dt.year; allT[sym] = T
        T.to_csv(f"part26_trades_{sym}.csv", index=False, float_format="%.4f")
        years = sorted(T.year.unique())
        P(f"\n## {sym}（{T.day.min().date()} ~ {T.day.max().date()}）")
        P(f"  {'規則':36s} {'真價位 全期':>34s} {'假價位':>16s} {'真−假':>7s} | 逐年（真價位每筆 R）")
        for rule in RULES:
            x = T[(T.kind == "real") & (T.rule == rule)].R
            f = T[(T.kind != "real") & (T.rule == rule)].R
            yr = "  ".join(f"{y}: {T[(T.kind == 'real') & (T.rule == rule) & (T.year == y)].R.mean():+.3f}" for y in years)
            P(f"  {rule} {NAMES[rule]:33s} {x.mean():+.3f}R t={tst(x):+5.2f} PF {pf(x):.2f} 勝 {(x > 0).mean() * 100:2.0f}% ({len(x):4d})"
              f"   {f.mean():+.3f}R ({len(f):4d})  {x.mean() - f.mean():+.3f} | {yr}")
    P("\n## 兩個商品合計（真價位）與「真 − 假」差異的 t（以日為單位配對）")
    for rule in RULES:
        cells = []
        for sym, T in allT.items():
            x = T[(T.kind == "real") & (T.rule == rule)].groupby("day").R.sum()
            f = T[(T.kind != "real") & (T.rule == rule)].groupby("day").R.sum() / 2
            dif = x.reindex(x.index.union(f.index), fill_value=0) - f.reindex(x.index.union(f.index), fill_value=0)
            cells.append(f"{sym} 真−假 每天 {dif.mean():+.3f}R t={tst(dif):+.2f}")
        P(f"  {rule} {NAMES[rule]:33s} " + " | ".join(cells))
    P("\n## 多 / 空拆開（真價位每筆 R；兩個商品這段期間都大漲，突破類若只有多單賺就是 beta）")
    for rule in RULES:
        cells = []
        for sym, T in allT.items():
            x = T[(T.kind == "real") & (T.rule == rule)]
            cells.append(f"{sym} 多 {x[x.side == 1].R.mean():+.3f} ({(x.side == 1).sum()}) 空 {x[x.side == -1].R.mean():+.3f} ({(x.side == -1).sum()})")
        P(f"  {rule} {NAMES[rule]:33s} " + " | ".join(cells))
    out.close()
