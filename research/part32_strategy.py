"""第三十二部分：插針 KDE 價位反轉 → 實際策略規則與完整績效，並測小時框（M1 執行）。

策略規則（每一組設定都相同）：
  1. 價位：pin TF 的插針（影線 >= 50% 全長且 >= 0.8 ATR）做 KDE，峰值 = 價位（kde_levels.py）。
     模式：rolling N 天（每天開盤前重算）、expanding、fixed（只用 1/2~5/31 的插針）。
  2. 進場：價格先離開價位 >= 1 ATR（執行 TF 的 ATR14），之後回到價位 0.1 ATR 內 →
     在 價位 ± 0.1 ATR 掛限價順勢「反向」進場（從上往下碰 = 做多，從下往上碰 = 做空）。
  3. 停損 1 ATR；停利 tp × ATR（tp = 1 或 2）；最多持有 24 根執行 TF K 棒，到時收盤平倉。
     碰觸那一根若同時碰到停損算停損（保守）；之後同一根同時碰停損與停利算停損。
  4. 一次只持有一筆（有部位時新的碰觸不做）。每筆扣當根點差。
  5. 可選 proven 濾網：同價位（±h）最近一次已揭曉的碰觸有先反彈 1 ATR 才做。
績效以 R（= 1 ATR 停損）計，另換算成美元（0.01 手 = 1 oz，每 $1 價差 = $1）。
placebo：價位整體隨機平移 3~6 個頻寬，同樣規則、10 次平均。
"""
import numpy as np
import pandas as pd
from load import load
from td import resample, atr
from kde_levels import find_pins
import part30_kde as P30
from part30_kde import touches, outcomes, build_levels, shift_levels, SPLIT
from part31_invalidate import apply_rules, add_h

H = 24


def trades(x, A, ev, tp, rule_mask=None):
    h, l, c = (x[k].to_numpy(float) for k in ("high", "low", "close"))
    spr = x.spread.to_numpy(float) * 0.01
    n = len(c)
    ev = ev if rule_mask is None else ev[rule_mask]
    ev = ev.sort_values(["t", "strength"], ascending=[True, False])
    out, busy = [], -1
    for t, d, L, a in zip(ev.t.to_numpy(), ev.dir.to_numpy(), ev.L.to_numpy(), ev.atr.to_numpy()):
        if t <= busy:
            continue
        ref = L + d * P30.TOL * a
        stop, tgt = ref - d * a, ref + d * tp * a
        end = min(t + H, n - 1)
        ex, px = end, c[end]
        for j in range(t, end + 1):
            if (l[j] <= stop) if d == 1 else (h[j] >= stop):
                ex, px = j, stop
                break
            if j > t and ((h[j] >= tgt) if d == 1 else (l[j] <= tgt)):
                ex, px = j, tgt
                break
        busy = ex
        R = ((px - ref) * d - spr[t]) / a
        out.append((x.index[t], x.index[ex], d, ref, a, R, R * a))
    return pd.DataFrame(out, columns=["entry", "exit", "dir", "price", "atr", "R", "usd"])


def stats(tr, ndays):
    if len(tr) == 0:
        return dict(n=0)
    R = tr.R.to_numpy()
    eq = np.cumsum(R)
    dd = (np.maximum.accumulate(np.r_[0, eq]) - np.r_[0, eq]).max()
    usd = tr.usd.to_numpy()
    eqd = np.cumsum(usd)
    ddu = (np.maximum.accumulate(np.r_[0, eqd]) - np.r_[0, eqd]).max()
    pf = R[R > 0].sum() / -R[R < 0].sum() if (R < 0).any() else np.inf
    pfu = usd[usd > 0].sum() / -usd[usd < 0].sum() if (usd < 0).any() else np.inf
    daily = tr.groupby(tr.entry.dt.normalize()).R.sum()
    hold = (tr.exit - tr.entry).dt.total_seconds().median() / 60
    return dict(n=len(R), per_day=len(R) / ndays, win=(R > 0).mean(), avgR=R.mean(),
                t=R.mean() / R.std(ddof=1) * np.sqrt(len(R)), PF=pf, totR=R.sum(), maxDD_R=dd,
                RoverDD=R.sum() / dd if dd > 0 else np.nan, PF_usd=pfu, tot_usd=usd.sum(),
                maxDD_usd=ddu, avg_win=R[R > 0].mean(), avg_loss=R[R < 0].mean(),
                max_loss_streak=max_streak(R < 0), hold_min=hold,
                worst_day=daily.min())


def max_streak(b):
    m = cur = 0
    for v in b:
        cur = cur + 1 if v else 0
        m = max(m, cur)
    return m


def main():
    m1 = load("data/XAUUSD_M1_2026.csv")
    exec_cache, pin_cache = {}, {}
    START = pd.Timestamp("2026-02-02")
    CFG = [
        # (exec TF, pin TF, mode, N, bw)
        ("5min", "15min", "expanding", None, 0.5),
        ("5min", "15min", "rolling", 20, 0.5),
        ("5min", "15min", "fixed", None, 0.5),
        ("5min", "1h", "expanding", None, 0.5),
        ("5min", "1h", "rolling", 5, 0.5),
        ("5min", "1h", "fixed", None, 0.5),
        # 小時框：M1 執行
        ("1min", "5min", "rolling", 5, 0.5),
        ("1min", "5min", "rolling", 20, 0.5),
        ("1min", "15min", "rolling", 20, 0.5),
        ("1min", "15min", "expanding", None, 0.5),
        ("1min", "15min", "fixed", None, 0.5),
        ("1min", "1h", "expanding", None, 0.5),
        ("1min", "1h", "fixed", None, 0.5),
    ]
    rows, best_trades = [], {}
    P30.RNG = np.random.default_rng(2)
    for etf, ptf, mode, N, bw in CFG:
        if etf not in exec_cache:
            x = resample(m1, etf)
            A = atr(x)
            day = x.index.normalize()
            sl = {}
            for d in day.unique():
                if d < START:
                    continue
                idx = np.flatnonzero(day == d)
                sl[d] = (idx[0], idx[-1] + 1)
            exec_cache[etf] = (x, A, sl)
        x, A, sl = exec_cache[etf]
        if ptf not in pin_cache:
            dd = resample(m1, ptf)
            pin_cache[ptf] = find_pins(dd, atr(dd))
        pins = pin_cache[ptf]
        lv = build_levels(pins, list(sl), mode, N, bw)
        days_used = [d for d in sl if d in lv]
        name = f"exec {etf} | pins {ptf} {mode}{'' if N is None else N}"
        sets = [("KDE", lv)] + [("placebo", shift_levels(lv)) for _ in range(10)]
        for kind, L in sets:
            ev = outcomes(x, touches(x, A, sl, L))
            ev, masks = apply_rules(ev, add_h(ev, L, x))
            for rule in ("base", "proven"):
                for tp in (1, 2):
                    tr = trades(x, A, ev, tp, masks[rule])
                    key = (name, rule, tp, kind)
                    if kind == "KDE":
                        best_trades[(name, rule, tp)] = tr
                        for grp, g in (("all", tr), ("H1", tr[tr.entry < SPLIT]), ("H2", tr[tr.entry >= SPLIT])):
                            nd = len([d for d in days_used if (grp == "all") or (grp == "H1") == (d < SPLIT)])
                            s = stats(g, max(nd, 1))
                            rows.append(dict(cfg=name, rule=rule, tp=tp, grp=grp, kind="KDE", **s))
                    else:
                        s = stats(tr, max(len(days_used), 1))
                        rows.append(dict(cfg=name, rule=rule, tp=tp, grp="all", kind="placebo", **s))
        print(name, "done", flush=True)
    res = pd.DataFrame(rows)
    num = res.select_dtypes("number").columns
    pl = res[res.kind == "placebo"].groupby(["cfg", "rule", "tp"])[["n", "per_day", "win", "avgR", "PF"]].mean()
    pl.columns = [c + "_pl" for c in pl.columns]
    kde = res[res.kind == "KDE"].drop(columns="kind")
    kde = kde.merge(pl.reset_index(), on=["cfg", "rule", "tp"], how="left")
    kde.to_csv("part32_summary.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 300)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 40)
    cols = ["cfg", "rule", "tp", "grp", "n", "per_day", "win", "avgR", "t", "PF", "totR", "maxDD_R",
            "RoverDD", "tot_usd", "maxDD_usd", "max_loss_streak", "hold_min", "PF_pl", "avgR_pl"]
    txt = kde[cols].round(3).to_string()
    open("results_part32.txt", "w").write(txt + "\n")
    print(txt)
    pd.to_pickle(best_trades, "/tmp/claude-0/-home-user-20061112/ec30ee8c-5c52-5218-87f6-11d2155d3637/scratchpad/p32_trades.pkl")


if __name__ == "__main__":
    main()
