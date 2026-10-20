"""第四十二部分 B：在 Fabio 框架內改良 —— 出場（保本 / 移動停損 / 抱到收盤）、IVB 進場位置與過濾條件。

紀律：研究期（XAU 2023/6~2025/9、NAS 2023/10~2025/9）分成前後兩半，只用研究期挑；
      保留期 2025/10 ~ 2026/10 參數凍結後只跑一次（part42_holdout.py）。
成交：一律用較保守的限價成交（價格要穿過掛單價 THROUGH 才算成交），每筆扣點差。
→ results_part42_improve.txt
"""
import numpy as np
import pandas as pd
from part41_fabio import load_symbol, profile, stats, fmt, NY_OPEN, NY_CLOSE, LDN_OPEN

RESEARCH_END = pd.Timestamp("2025-10-01")
THROUGH = {"XAUUSD": 0.1, "NAS100": 1.0}
SPLIT = {"XAUUSD": pd.Timestamp("2024-08-01"), "NAS100": pd.Timestamp("2024-10-01")}  # 研究期前後兩半


def exit_sim(d, i, side, entry, stop, rule, last_m=NY_CLOSE, same_bar=False):
    """rule：('rr', k) 固定 k R 停利；('be', k, b) k R 停利 + 浮盈 b R 後保本；
             ('trail', b, g) 浮盈 b R 後停損跟在最大浮盈 − g R，沒有停利；('close', b) 只有停損 + b R 後保本，last_m 平倉。"""
    H, L, C, M = d["H"], d["L"], d["C"], d["M"]
    R = abs(entry - stop)
    st, best = stop, 0.0
    kind = rule[0]
    tgt = entry + side * rule[1] * R if kind in ("rr", "be") else None
    if same_bar and (side > 0 and L[i] <= stop or side < 0 and H[i] >= stop):
        return -1.0, i
    for k in range(i if same_bar else i + 1, len(H)):
        if same_bar and k == i:
            continue
        if M[k] >= last_m:
            return side * (C[k - 1] - entry) / R, k
        lo_r = side * ((L[k] if side > 0 else H[k]) - entry) / R    # 這根最不利
        hi_r = side * ((H[k] if side > 0 else L[k]) - entry) / R    # 這根最有利
        st_r = side * (st - entry) / R
        if lo_r <= st_r:
            return st_r, k
        if tgt is not None and hi_r >= rule[1]:
            return rule[1], k
        best = max(best, hi_r)
        # 收盤後才更新停損（同一根內不假設先後順序）
        if kind == "be" and best >= rule[2] or kind == "close" and best >= rule[1]:
            st = entry if side * (entry - st) > 0 else st
        if kind == "trail" and best >= rule[1]:
            st = entry + side * max(best - rule[2], 0.0) * R if side * (entry + side * max(best - rule[2], 0) * R - st) > 0 else st
    return side * (C[-1] - entry) / R, len(H) - 1


EXITS = [("rr", 2.0), ("rr", 3.0), ("rr", 4.0), ("be", 2.0, 1.0), ("be", 3.0, 1.0), ("be", 4.0, 1.5),
         ("trail", 1.0, 1.0), ("trail", 1.5, 1.5), ("trail", 2.0, 2.0), ("close", 1.0), ("close", 1.5)]


def ename(e):
    return {"rr": "停利{}R", "be": "停利{}R+{}R保本", "trail": "{}R後移動停損(距{}R)", "close": "抱到收盤+{}R保本"}[e[0]].format(*e[1:])


def ivb_setup(day, d, prev, hist, sym, entry_lv="vah", stop_lv="poc", buf_f=0.02, or_len=30):
    """回傳 IVB 進場（不含出場）與特徵；限價需穿價 THROUGH 才成交。
    entry_lv：vah（框上緣，原版）/ mid（框中間）/ edge（開盤區間高點，突破點回測）
    stop_lv：poc（原版）/ val（價值區另一側）"""
    O, H, L, C, M = d["O"], d["H"], d["L"], d["C"], d["M"]
    orm = (M >= NY_OPEN) & (M < NY_OPEN + or_len)
    if orm.sum() < 0.8 * or_len:
        return None
    oh, ol = H[orm].max(), L[orm].min()
    p = profile(H[orm], L[orm], d["V"][orm])
    if p is None:
        return None
    buf, th = buf_f * (oh - ol), THROUGH[sym]
    side, brk = 0, None
    for i in np.where(M >= NY_OPEN + or_len)[0]:
        if M[i] >= NY_CLOSE:
            return None
        if side == 0:
            if M[i] >= 19 * 60 + 30:
                return None
            if (M[i] + 1) % 5 == 0:
                side = 1 if C[i] > oh else (-1 if C[i] < ol else 0)
                brk = i if side else None
            continue
        if M[i] >= 21 * 60:
            return None
        hi_box, poc, lo_box = (p["vah"], p["poc"], p["val"]) if side > 0 else (p["val"], p["poc"], p["vah"])
        lv = {"vah": hi_box, "mid": (hi_box + poc) / 2, "edge": oh if side > 0 else ol}[entry_lv]
        stop = {"poc": poc, "val": lo_box}[stop_lv] - side * buf
        if side * (C[i] - lo_box) < 0:
            return None
        if side * (lv - stop) <= 0:
            return None
        if (L[i] <= lv - th) if side > 0 else (H[i] >= lv + th):
            entry = min(O[i], lv) if side > 0 else max(O[i], lv)
            # 特徵（只用進場當下已知的資訊）
            past = [h for h in hist[-20:]]
            or_z = (np.log(oh - ol) - np.mean(np.log(past))) / np.std(np.log(past)) if len(past) >= 10 else np.nan
            ldn = (M >= LDN_OPEN) & (M < NY_OPEN)
            ldn_dir = np.sign(C[ldn][-1] - O[ldn][0]) * side if ldn.sum() > 100 else 0
            pva = 0
            if prev is not None:
                pm = (prev["M"] >= NY_OPEN) & (prev["M"] < NY_CLOSE)
                pp = profile(prev["H"][pm], prev["L"][pm], prev["V"][pm]) if pm.sum() > 300 else None
                if pp is not None:
                    o = O[orm][0]
                    pva = side * (1 if o > pp["vah"] else (-1 if o < pp["val"] else 0))
            return dict(day=day, i=i, side=side, entry=entry, stop=stop, R_pts=abs(entry - stop), cost=d["S"][i],
                        or_z=or_z, brk_m=M[brk], wait=M[i] - M[brk], ldn_dir=ldn_dir, pva=pva, dow=day.dayofweek,
                        or_rng=oh - ol)
    return None


def run_ivb(days, sym, **kw):
    keys = sorted(days)
    hist, rows = [], []
    for n, day in enumerate(keys):
        d = days[day]
        s = ivb_setup(day, d, days[keys[n - 1]] if n else None, hist, sym, **kw)
        if s:
            rows.append((s, d))
        orm = (d["M"] >= NY_OPEN) & (d["M"] < NY_OPEN + kw.get("or_len", 30))
        if orm.sum():
            hist.append(d["H"][orm].max() - d["L"][orm].min())
    return rows


def with_exit(rows, rule):
    out = []
    for s, d in rows:
        r, k = exit_sim(d, s["i"], s["side"], s["entry"], s["stop"], rule, same_bar=True)
        out.append(dict(s, R=r - s["cost"] / s["R_pts"], R_gross=r, t_out=d["M"][k]))
    return pd.DataFrame(out)


def halves(g, sym):
    g = g[g.day < RESEARCH_END]
    a, b = g[g.day < SPLIT[sym]], g[g.day >= SPLIT[sym]]
    return g, a.R.mean(), b.R.mean()


def main():
    out = []
    for sym in ("XAUUSD", "NAS100"):
        days = load_symbol(sym)
        out.append(f"\n################ {sym}（研究期，限價穿價 {THROUGH[sym]} 才成交） ################")
        # 1) 進場位置 × 停損 × 出場
        out.append("\n[1] IVB 進場位置 × 停損 × 出場：研究期全部 / 前半 / 後半（淨 R）")
        res = []
        cache = {}
        for elv in ("vah", "mid", "edge"):
            for slv in ("poc", "val"):
                rows = run_ivb(days, sym, entry_lv=elv, stop_lv=slv)
                cache[(elv, slv)] = rows
                for e in EXITS:
                    g, a, b = halves(with_exit(rows, e), sym)
                    s = stats(g.R)
                    res.append(dict(進場=elv, 停損=slv, 出場=ename(e), n=s["n"], 期望=s["avg"], t=s["t"], 前半=a, 後半=b,
                                    做多=g[g.side > 0].R.mean(), 做空=g[g.side < 0].R.mean(), 回撤=s["dd"]))
        rd = pd.DataFrame(res)
        out.append(rd.round(3).to_string(index=False))
        both = rd[(rd.前半 > 0) & (rd.後半 > 0)]
        out.append(f"→ 兩半皆正 {len(both)}/{len(rd)} 組")
        for col in ("進場", "停損", "出場"):
            out.append(f"  依{col}平均：" + ", ".join(f"{k} {v:+.3f}" for k, v in rd.groupby(col).期望.mean().items()))
        # 2) 過濾條件：用原版進場（vah/poc）+ 研究期最穩的出場
        base_rows = cache[("vah", "poc")]
        sub = rd[(rd.進場 == "vah") & (rd.停損 == "poc")].copy()
        sub["minh"] = sub[["前半", "後半"]].min(axis=1)
        best_e = EXITS[[ename(e) for e in EXITS].index(sub.sort_values("minh").iloc[-1].出場)]
        out.append(f"\n[2] 過濾條件（進場 vah / 停損 poc / 出場 {ename(best_e)}；前半 | 後半 | 筆數）")
        g = with_exit(base_rows, best_e)
        g = g[g.day < RESEARCH_END]
        g["half"] = np.where(g.day < SPLIT[sym], "前", "後")
        g["or_zq"] = pd.cut(g.or_z, [-9, -0.5, 0.5, 9], labels=["小區間", "中", "大區間"])
        g["brk_q"] = pd.cut(g.brk_m, [0, 17 * 60 + 15, 18 * 60, 24 * 60], labels=["17:15前突破", "17:15~18:00", "18:00後"])
        g["wait_q"] = pd.cut(g.wait, [-1, 10, 30, 999], labels=["≤10分回踩", "10~30分", ">30分"])
        g["side_s"] = np.where(g.side > 0, "做多", "做空")
        g["ldn_s"] = g.ldn_dir.map({1: "順倫敦方向", -1: "逆倫敦方向", 0: "倫敦持平"})
        g["pva_s"] = g.pva.map({1: "開在前日VA外(順)", -1: "開在前日VA外(逆)", 0: "開在前日VA內"})
        g["dow_s"] = g.dow.map(dict(enumerate("一二三四五"))).radd("週")
        for f in ("side_s", "or_zq", "brk_q", "wait_q", "ldn_s", "pva_s", "dow_s"):
            tb = g.groupby([f, "half"], observed=True).R.mean().unstack()
            nn = g.groupby(f, observed=True).size()
            out.append("  " + " ; ".join(f"{k}: {tb.loc[k].get('前', np.nan):+.2f} | {tb.loc[k].get('後', np.nan):+.2f} | {nn[k]}"
                                         for k in tb.index))
    txt = "\n".join(out)
    print(txt)
    open("results_part42_improve.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
