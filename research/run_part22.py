"""第二十二部分：推動段 → 震盪（旗形）→ 順勢突破。輸出 results_part22.txt、part22_*.csv、part22_*.png。

研究流程（吸取第十部分樣本外失敗的教訓，一開始就用多年資料）：
  研究期 2023/1 ~ 2024/12：看特徵、選規則、選出場
  驗證期 2025/1 ~ 2026/10：只跑一次，不回頭調整
週期：M15（2023/1~2026/10）、H1（由 M15 合成）、M5（由 M1 合成，2023/6~2026/10）。
"""
import sys
import numpy as np
import pandas as pd
from load import load
from backtest import resample
from part22_flag import events, simulate, non_overlap, summ, formations, placebo_boxes

CUT = pd.Timestamp("2025-01-01")
FEATS = ["pole_atr", "pole_er", "body_ratio", "ke_ratio", "zc_rate", "area", "retr", "box_atr",
         "bbw_ratio", "bbw_pct", "vol_ratio", "brk_body", "brk_dist", "day_er", "m"]
CATS = ["htf", "day_dir", "sess", "monday", "dir"]
EXITS = [("box", 1.0, False), ("box", 2.0, False), ("box", 3.0, False), ("box", None, False), ("box", None, True),
         ("mid", 2.0, False), ("mid", 3.0, False), ("mid", None, False)]


def sess(h):
    return np.select([h < 10, h < 15, h < 20], ["1 亞洲01-09", "2 倫敦10-14", "3 紐約15-19"], "4 尾盤20-23")


def fmt(s):
    return f"n={s['n']:5d} 平均 {s['mean']:+.3f}R t={s['t']:+.2f} 勝率 {s['win']:.2f} PF {s['pf']:.2f}"


def year_line(E, col="R"):
    g = E.groupby(E.time.dt.year)[col]
    return "  ".join(f"{y}:{v.mean():+.3f}({len(v)})" for y, v in g)


def section_direction(df, name, kw, out, pstep=1):
    """測試 A：旗面成形後，順旗桿方向突破的比例是否高於隨機漫步基準。"""
    F = formations(df, **kw).dropna(subset=["up_first"])
    F["excess"] = F.up_first - F.exp_up
    out(f"\n## A. {name}：旗面成形後先往哪邊突破（{len(F)} 個旗面）")
    out("  順旗桿方向先突破的比例 vs 隨機漫步基準（現價在箱體中的位置）；excess = 實際 − 基準")
    for y, g in F.groupby(F.time.dt.year):
        se = g.excess.std() / np.sqrt(len(g))
        out(f"  {y}: n={len(g):5d} 實際 {g.up_first.mean():.3f} 基準 {g.exp_up.mean():.3f} excess {g.excess.mean():+.3f} (t={g.excess.mean() / se:+.2f})")
    se = F.excess.std() / np.sqrt(len(F))
    out(f"  全期: 實際 {F.up_first.mean():.3f} 基準 {F.exp_up.mean():.3f} excess {F.excess.mean():+.3f} (t={F.excess.mean() / se:+.2f})")
    # 安慰劑：任意 5 根箱體（不要求旗桿）。同一年、同一個「現價在箱體中的位置」分組相減
    P = placebo_boxes(df, L=kw.get("Mmin", 4) + 1, step=pstep)
    P["excess"] = P.up_first - P.exp_up
    bins = np.linspace(0, 1, 11)
    F["pb"] = np.digitize(F.exp_up, bins[1:-1])
    P["pb"] = np.digitize(P.exp_up, bins[1:-1])
    ref = P.groupby([P.time.dt.year, "pb"]).excess.mean()
    F["placebo"] = [ref.get((y, b), np.nan) for y, b in zip(F.time.dt.year, F.pb)]
    F["vs_placebo"] = F.excess - F.placebo
    out(f"  安慰劑（任意 {kw.get('Mmin', 4) + 1} 根箱體，{len(P)} 個）excess {P.excess.mean():+.3f}；"
        "按現價位置分組：")
    for b, g in P.groupby(pd.cut(P.exp_up, [0, .2, .4, .6, .8, 1]), observed=True):
        f = F[(F.exp_up > b.left) & (F.exp_up <= b.right)]
        out(f"    位置 {b}: 安慰劑 excess {g.excess.mean():+.3f} (n={len(g)}) | 旗面 {f.excess.mean():+.3f} (n={len(f)})")
    out("  ★ 旗面 − 同年同位置的安慰劑（= 旗桿本身帶來的順勢傾向）：")
    for y, g in F.groupby(F.time.dt.year):
        out(f"    {y}: {g.vs_placebo.mean():+.3f} (t={g.vs_placebo.mean() / g.vs_placebo.std() * np.sqrt(len(g)):+.2f})")
    v = F.vs_placebo.dropna()
    out(f"    全期: {v.mean():+.3f} (t={v.mean() / v.std() * np.sqrt(len(v)):+.2f})")
    out("  按特徵三分位（旗面 − 安慰劑；括號為 2023-24 / 2025-26）：")
    F["excess"] = F.vs_placebo
    for f in ["pole_atr", "pole_er", "body_ratio", "ke_ratio", "area", "retr", "box_atr", "bbw_ratio"]:
        q = pd.qcut(F[f], 3, labels=["低", "中", "高"], duplicates="drop")
        parts = []
        for k, g in F.groupby(q, observed=True):
            a, b = g[g.time < CUT].excess.mean(), g[g.time >= CUT].excess.mean()
            parts.append(f"{k} {g.excess.mean():+.3f} ({a:+.3f}/{b:+.3f})")
        out(f"    {f:10s} " + " | ".join(parts))
    return F


def tercile_table(E, out, col="R"):
    """每個特徵：研究期定三分位，看 2023、2024、2025、2026 各年的平均 R。"""
    res = E[E.time < CUT]
    out("  特徵（三分位以研究期切）         低           中           高      | 研究期 高−低 2023 / 2024 | 驗證期 高−低")
    rows = []
    for f in FEATS:
        try:
            qs = res[f].quantile([1 / 3, 2 / 3]).to_numpy()
        except Exception:
            continue
        if qs[0] == qs[1]:
            continue
        q = np.digitize(E[f], qs)
        E2 = E.assign(q=q)
        mean = E2.groupby("q")[col].mean()
        yr = E2.groupby([E2.time.dt.year, "q"])[col].mean().unstack()
        d23 = yr.loc[2023, 2] - yr.loc[2023, 0] if 2023 in yr.index else np.nan
        d24 = yr.loc[2024, 2] - yr.loc[2024, 0] if 2024 in yr.index else np.nan
        v = E2[E2.time >= CUT]
        dv = v[v.q == 2][col].mean() - v[v.q == 0][col].mean()
        rows.append(dict(feat=f, lo=qs[0], hi=qs[1], d23=d23, d24=d24, dval=dv))
        out(f"  {f:10s} [{qs[0]:7.3f},{qs[1]:7.3f}] {mean.get(0, np.nan):+.3f}  {mean.get(1, np.nan):+.3f}  {mean.get(2, np.nan):+.3f}"
            f"   |   {d23:+.3f} / {d24:+.3f}            | {dv:+.3f}")
    for f in CATS:
        g = E.groupby(f)[col]
        yr = E.groupby([E.time.dt.year, f])[col].mean().unstack()
        out(f"  {f:10s} " + "  ".join(f"{k}:{v.mean():+.3f}({len(v)})" for k, v in g)
            + "   逐年: " + "; ".join(f"{y}[" + ",".join(f"{x:+.2f}" for x in r.to_numpy()) + "]" for y, r in yr.iterrows()))
    return pd.DataFrame(rows)


def run_tf(df, name, kw, out):
    H = kw.get("H", 32)
    E = events(df, **{k: v for k, v in kw.items() if k != "H"})
    E["sess"] = sess(E.hour.to_numpy())
    E["monday"] = (E.dow == 0).astype(int)
    out(f"\n\n# {name}：突破事件 {len(E)} 個（{E.time.min():%Y-%m-%d} ~ {E.time.max():%Y-%m-%d}），"
        f"每交易日約 {len(E) / E.time.dt.normalize().nunique():.2f} 個")
    out(f"  旗桿中位 {E.pole_atr.median():.1f} ATR、旗面中位 {E.m.median():.0f} 根、箱體中位 {E.box_atr.median():.2f} ATR")

    # B. 全部事件、各種出場，逐年
    out("\n## B. 全部「記錄 HH/LL → 收盤突破」事件，各種出場（逐年平均 R，括號為筆數）")
    sims = {}
    for st, tp, ms in EXITS:
        key = f"{st}/{'量測' if ms else (f'{tp:g}R' if tp else '無停利')}"
        S = simulate(df, E, stop=st, tp=tp, H=H, measured=ms)
        sims[key] = S
        e2 = E.assign(R=S.R)
        out(f"  {key:10s} 全期 {fmt(summ(S.R))} | {year_line(e2)}")
    # 不受停損設計影響的前瞻報酬（ATR）
    for hz in (4, 16, 32):
        S = simulate(df, E, stop="box", tp=None, H=hz)
        out(f"  抱 {hz:2d} 根（只有箱體停損）每筆 {S.pnl_atr.mean():+.3f} ATR | " + year_line(E.assign(R=S.pnl_atr)))

    base = sims["box/2R"]
    E["R"] = base.R
    E["bars"] = base.bars

    # C. 特徵三分位
    out("\n## C. 特徵三分位（出場：箱體停損 + 2R + 最長 H 根）")
    T = tercile_table(E, out)

    # D. 使用者假說：教科書旗形（事先寫死，不調參）
    out("\n## D. 事先寫死的「教科書旗形」規則（不調參）與對照組")
    rules = {
        "全部事件": np.ones(len(E), bool),
        "D1 有推動段 pole≥4ATR": E.pole_atr >= 4,
        "D2 D1 + 旗面K變小 body_ratio≤0.6": (E.pole_atr >= 4) & (E.body_ratio <= 0.6),
        "D3 D2 + 動能衰減 ke_ratio≤0.3": (E.pole_atr >= 4) & (E.body_ratio <= 0.6) & (E.ke_ratio <= 0.3),
        "D4 D3 + 回檔≤50%": (E.pole_atr >= 4) & (E.body_ratio <= 0.6) & (E.ke_ratio <= 0.3) & (E.retr <= 0.5),
        "D5 D4 + 布林收縮 bbw_ratio<0.8": (E.pole_atr >= 4) & (E.body_ratio <= 0.6) & (E.ke_ratio <= 0.3) & (E.retr <= 0.5) & (E.bbw_ratio < 0.8),
        "D6 D4 + 積分黏在突破側 area>0": (E.pole_atr >= 4) & (E.body_ratio <= 0.6) & (E.ke_ratio <= 0.3) & (E.retr <= 0.5) & (E.area > 0),
        "對照 無推動段 pole<3": E.pole_atr < 3,
        "對照 旗面K沒變小 body_ratio>1": E.body_ratio > 1,
        "對照 深回檔 retr>0.8": E.retr > 0.8,
    }
    for k, msk in rules.items():
        g = E[msk.to_numpy() if hasattr(msk, "to_numpy") else msk]
        out(f"  {k:34s} {fmt(summ(g.R))} | {year_line(g)}")

    # E. 事先登記的選擇程序（只用 2023~2024）
    out("\n## E. 事先登記的選擇程序（只看 2023~2024）")
    out("  規則：特徵的『高−低』三分位差在 2023 與 2024 同號、兩年平均 |差| ≥ 0.10R → 保留；"
        "保留的特徵各刪掉較差的那個三分位（刪低或刪高）。")
    keep = T[(np.sign(T.d23) == np.sign(T.d24)) & (((T.d23 + T.d24) / 2).abs() >= 0.10)]
    msk = np.ones(len(E), bool)
    for _, r in keep.iterrows():
        if (r.d23 + r.d24) > 0:
            msk &= (E[r.feat] >= r.lo).to_numpy()
            out(f"    保留 {r.feat}：刪掉 < {r.lo:.3f}（研究期差 {r.d23:+.3f}/{r.d24:+.3f}；驗證期差 {r.dval:+.3f}）")
        else:
            msk &= (E[r.feat] <= r.hi).to_numpy()
            out(f"    保留 {r.feat}：刪掉 > {r.hi:.3f}（研究期差 {r.d23:+.3f}/{r.d24:+.3f}；驗證期差 {r.dval:+.3f}）")
    if keep.empty:
        out("    沒有任何特徵過關。")
        return E, None
    sel = E[msk]
    # 出場也只用研究期選
    best, bestv = None, -9
    for key, S in sims.items():
        r = S.R[msk & (E.time < CUT).to_numpy()]
        if r.mean() > bestv:
            best, bestv = key, r.mean()
    out(f"  研究期最好的出場：{best}（研究期 {bestv:+.3f}R）")
    sel = sel.assign(R=sims[best].R[msk], bars=sims[best].bars[msk])
    for lab, g in (("研究期 2023-24", sel[sel.time < CUT]), ("驗證期 2025-26", sel[sel.time >= CUT])):
        out(f"  {lab}（可重疊）  {fmt(summ(g.R))}")
        k = non_overlap(g, g.bars.to_numpy())
        out(f"  {lab}（一次一筆）{fmt(summ(g.R.to_numpy()[k]))}")
    out("  逐年：" + year_line(sel))
    return E, sel


def main():
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)
        sys.stdout.flush()

    m15 = load("data/XAUUSD_M15_2023_2026.csv")
    h1 = resample(m15, "1h")
    m1 = load("data/XAUUSD_M1_2023_2026.csv")
    m5 = resample(m1, "5min")
    del m1

    out("# 第二十二部分：推動段 → 震盪 → 順勢突破（研究期 2023~2024 / 驗證期 2025~2026/10）")
    cfgs = [("M15", m15, dict(K=24, Mmin=3, Mmax=24, atr_n=96)),
            ("H1", h1, dict(K=24, Mmin=3, Mmax=24, atr_n=120, H=24)),
            ("M5", m5, dict(K=24, Mmin=3, Mmax=24, atr_n=288, H=48))]
    for name, df, kw in cfgs:
        fk = {k: v for k, v in kw.items() if k in ("K", "Mmax", "atr_n")}
        A = section_direction(df, name, dict(fk, Mmin=4), out, pstep={"M15": 2, "H1": 1, "M5": 5}[name])
        A.to_csv(f"part22_formations_{name}.csv", index=False)
    for name, df, kw in cfgs:
        E, sel = run_tf(df, name, kw, out)
        E.to_csv(f"part22_events_{name}.csv", index=False)
        if sel is not None:
            sel.to_csv(f"part22_selected_{name}.csv", index=False)
    with open("results_part22.txt", "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
