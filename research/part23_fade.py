"""第二十三部分：第二十二部分的反向做法 — 旗形「做反向」能不能賺？
輸出 results_part23.txt。研究期 2023~2024 選變體，驗證期 2025~2026/10 只看一次。

使用第二十二部分已存的事件（part22_events_*.csv、part22_formations_*.csv，同樣的資料與合成方式，t 是 K 棒序號）。
四種反向做法（以「旗桿向上」為例；向下鏡像）：
  F1 突破就反手：收盤突破 HH 那根 → 以收盤價做空；停損 = 突破後最高價 + 0.1 ATR 以外（用突破 K 高點）
  F2 突破就反手：同 F1，停損固定 entry + k × ATR
  F3 假突破：突破後 W 根內收盤跌回 HH 以下 → 做空；停損 = 突破後最高點 + 0.1 ATR
  F4 旗面內逆勢：旗面成形（HH 後 4 根沒創高）時直接做空；停損 = HH + 0.1 ATR
停利：1R / 2R / 箱體中點 / 箱底；限時 H 根收盤出場。同一根同時碰到停損與停利 → 停損。每筆扣進場點差。
"""
import numpy as np
import pandas as pd
from load import load
from backtest import resample
from part22_flag import _atr, summ

CUT = pd.Timestamp("2025-01-01")
TFS = {"M15": dict(atr_n=96, H=32), "H1": dict(atr_n=120, H=24), "M5": dict(atr_n=288, H=48)}


def sim(h, l, c, t, d, entry, sl, tgt, sp, H):
    """t 收盤進場，之後 H 根內先碰停損或停利；都沒碰就第 H 根收盤出場。回傳 R。"""
    n = len(c)
    ix = np.minimum(t[:, None] + np.arange(1, H + 1)[None, :], n - 1)
    fav = np.where(d[:, None] == 1, h[ix], -l[ix])
    adv = np.where(d[:, None] == 1, l[ix], -h[ix])
    hsl = adv <= (sl * d)[:, None]
    htp = fav >= (tgt * d)[:, None]
    big = 10 ** 6
    fsl = np.where(hsl.any(1), hsl.argmax(1), big)
    ftp = np.where(htp.any(1), htp.argmax(1), big)
    px = c[ix][:, -1]
    px = np.where(ftp < fsl, tgt, px)
    px = np.where((fsl <= ftp) & (fsl < big), sl, px)
    risk = (entry - sl) * d
    return ((px - entry) * d - sp) / risk


def line(T):
    s = summ(T.R)
    yr = "  ".join(f"{y}:{g.R.mean():+.3f}({len(g)})" for y, g in T.groupby(T.time.dt.year))
    return f"n={s['n']:5d} {s['mean']:+.3f}R t={s['t']:+.2f} 勝率 {s['win']:.2f} PF {s['pf']:.2f} | {yr}"


def targets(entry, d, risk, box_mid, box_far):
    return {"1R": entry + d * risk, "2R": entry + d * 2 * risk, "箱體中點": box_mid, "箱底": box_far}


def run(df, name, atr_n, H, out):
    h, l, c = (df[k].to_numpy(float) for k in ("high", "low", "close"))
    sp_all = df.spread.to_numpy(float) * 0.01
    atr = _atr(h, l, c, atr_n)
    E = pd.read_csv(f"part22_events_{name}.csv", parse_dates=["time"])
    res = {}

    # 原本的順勢（對照）：part22 box/2R
    res["順勢 箱體停損 2R（第二十二部分）"] = E[["time"]].assign(R=E.R)

    t = E.t.to_numpy()
    d = -E.dir.to_numpy()              # 反向
    entry = E.entry.to_numpy()
    A = E.atr.to_numpy()
    hh, blo, mid = E.hh.to_numpy(), E.box_lo.to_numpy(), E.mid.to_numpy()
    sp = E.sp.to_numpy()
    # F1：停損 = 突破 K 棒極值 + 0.1 ATR
    brk_ext = np.where(d == -1, h[t], l[t])
    variants = {"F1 突破K極值": brk_ext - d * 0.1 * A}
    for k in (0.5, 1.0, 2.0):
        variants[f"F2 {k:g}ATR"] = entry - d * k * A
    for vn, sl in variants.items():
        risk = (entry - sl) * d
        for tn, tg in targets(entry, d, risk, mid, blo).items():
            ok = (tg - entry) * d > 0
            R = sim(h, l, c, t[ok], d[ok], entry[ok], sl[ok], tg[ok], sp[ok], H)
            res[f"{vn} / {tn}"] = pd.DataFrame({"time": E.time[ok].to_numpy(), "R": R})

    # F3：假突破 — 突破後 W 根內第一次收盤回到 HH 內側
    for W in (3, 6):
        rows = []
        for i, (ti, di, hhi) in enumerate(zip(t, E.dir.to_numpy(), hh)):
            seg = c[ti + 1:ti + 1 + W] * di
            back = np.flatnonzero(seg < hhi * di)
            if len(back) == 0:
                continue
            tb = ti + 1 + back[0]
            ext = (h[ti:tb + 1].max() if di == 1 else l[ti:tb + 1].min())
            rows.append((i, tb, ext))
        if not rows:
            continue
        ii, tb, ext = (np.array(x) for x in zip(*rows))
        dd = -E.dir.to_numpy()[ii]
        en = c[tb]
        sl = ext - dd * 0.1 * atr[tb]
        risk = (en - sl) * dd
        for tn, tg in targets(en, dd, risk, mid[ii], blo[ii]).items():
            ok = ((tg - en) * dd > 0) & (risk > 0)
            R = sim(h, l, c, tb[ok], dd[ok], en[ok], sl[ok], tg[ok], sp_all[tb][ok], H)
            res[f"F3 假突破 W={W} / {tn}"] = pd.DataFrame({"time": df.index[tb[ok]], "R": R})

    # F4：旗面內逆勢（第二十二部分的 formations；由 exp_up 與 box_atr 還原箱體價位）
    Fm = pd.read_csv(f"part22_formations_{name}.csv", parse_dates=["time"])
    t0 = Fm.t.to_numpy()
    dirp = Fm.dir.to_numpy()
    A0 = atr[t0]
    box = Fm.box_atr.to_numpy() * A0
    c0 = c[t0] * dirp
    blo_d = c0 - Fm.exp_up.to_numpy() * box
    hh_d = blo_d + box
    dd = -dirp
    en = c[t0]
    sl = hh_d * dirp + dirp * 0.1 * A0          # HH 外 0.1 ATR（對做空 = 上方）
    risk = (en - sl) * dd
    for pos_name, pos_ok in (("全部", np.ones(len(Fm), bool)), ("現價在箱體上半部", Fm.exp_up.to_numpy() > 0.5)):
        for tn, tg in targets(en, dd, risk, (hh_d - box / 2) * dirp, blo_d * dirp).items():
            ok = pos_ok & ((tg - en) * dd > 0) & (risk > 0)
            R = sim(h, l, c, t0[ok], dd[ok], en[ok], sl[ok], tg[ok], sp_all[t0][ok], H)
            res[f"F4 旗面內逆勢 {pos_name} / {tn}"] = pd.DataFrame({"time": Fm.time[ok].to_numpy(), "R": R})

    out(f"\n# {name}（最長持有 {H} 根）")
    for k, T in res.items():
        out(f"  {k:34s} {line(T)}")
    # 只用研究期挑最好的反向變體
    cand = {k: T for k, T in res.items() if not k.startswith("順勢")}
    best = max(cand, key=lambda k: cand[k][cand[k].time < CUT].R.mean())
    T = cand[best]
    out(f"  ▶ 研究期（2023~24）最好的反向變體：{best}")
    out(f"     研究期 {line(T[T.time < CUT])}")
    out(f"     驗證期 {line(T[T.time >= CUT])}")
    pos = sum(cand[k][cand[k].time < CUT].R.mean() > 0 for k in cand)
    posv = sum(cand[k][cand[k].time >= CUT].R.mean() > 0 for k in cand)
    out(f"  反向變體共 {len(cand)} 個：研究期為正 {pos} 個、驗證期為正 {posv} 個")


def main():
    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    out("# 第二十三部分：旗形做反向（研究期 2023~2024 / 驗證期 2025~2026/10）")
    m15 = load("data/XAUUSD_M15_2023_2026.csv")
    run(m15, "M15", out=out, **TFS["M15"])
    run(resample(m15, "1h"), "H1", out=out, **TFS["H1"])
    run(resample(load("data/XAUUSD_M1_2023_2026.csv"), "5min"), "M5", out=out, **TFS["M5"])
    with open("results_part23.txt", "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
