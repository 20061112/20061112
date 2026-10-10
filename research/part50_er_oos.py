"""第五十部分：ER 回檔順勢策略（FINAL_STRATEGY_ER.md、第 29 部分）的樣本外驗證。

規則完全沿用 part29_final.py（A M15 40/20/5、B M30 20/10/3、C M15 20/10/3；停利 1.5R / 2R / 1.5R；
最長持有 80 / 20 / 40 根；停損上限 $100；不做星期一；週五 20 點後不開新單；週五最後一根平倉；隔夜 $0.7/盎司/晚）。
資料：data/XAUUSD_M15_2023_2026.csv（M30、H1、H4 由 M15 合成）。2023~2025 為樣本外，2026 為樣本內（參考）。
三種部位：1 單位 / 1R 且同向加 1 / 同向 2 單位 + 1R 加 1。
"""
import numpy as np
import pandas as pd
from load import load
from part25_exits import prep, entries
from part25_ext import tf_er_done
from part27_sizing import sim
from part28_final import nights
from part29_optimize import LEGS

EXIT = {"A": (1.5, 80), "B": (2, 20), "C": (1.5, 40)}


def build(base):
    cache, rows = {}, []
    for leg, (tp, hold) in EXIT.items():
        tf, L, M, S, a, cool = LEGS[leg]
        B = cache.setdefault(tf, prep(base, tf))
        if "h1" not in B:
            B["h1"], B["h4"] = tf_er_done(base, B, "1h"), tf_er_done(base, B, "4h")
        step = B["idx"][1] - B["idx"][0]
        t, side = entries(B, L, M, S, a, cool)
        for k, s in zip(t, side):
            ts = B["idx"][k] + step
            if ts.dayofweek == 0 or (ts.dayofweek == 4 and ts.hour >= 20):
                continue
            r = sim(B, k, s, M, hold // 2, tp=tp, cap_usd=100, cap_mode="cap", flat="week")
            if not r:
                continue
            nt = nights(B["idx"], k + 1, r["j"])
            sw = 0.7 * nt
            rows.append(dict(leg=leg, time=B["idx"][k], xtime=B["idx"][r["j"]], side=s,
                             htf=bool((s * B["h1"][k] > 0) and (s * B["h4"][k] > 0)), added=r["added"], reason=r["reason"],
                             stop_usd=r["Rusd"], R=r["R"] - sw / r["Rusd"],
                             addR=(r["addR"] - sw / r["Rusd"]) if r["added"] else 0.0,
                             usd=r["usd"] - sw, add_usd=(r["add_usd"] - sw) if r["added"] else 0.0))
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


def stats(T, R, usd, days):
    d = pd.Series(R.to_numpy(), index=T.xtime.dt.date).groupby(level=0).sum().reindex(days, fill_value=0.0)
    du = pd.Series(usd.to_numpy(), index=T.xtime.dt.date).groupby(level=0).sum().reindex(days, fill_value=0.0)
    eq = d.cumsum().to_numpy()
    mdd = -(eq - np.maximum.accumulate(np.r_[0, eq])[1:]).min()
    v = R.to_numpy()
    return dict(n=len(T), win=np.mean(v > 0), pf=v[v > 0].sum() / -v[v < 0].sum(), avg=v.mean(),
                t=v.mean() / v.std() * np.sqrt(len(v)), tot=d.sum(), mdd=mdd, sh=d.mean() / d.std() * np.sqrt(252),
                usd=du.sum(), top5=d.sort_values(ascending=False).head(5).sum() / d.sum())


def main():
    base = load("data/XAUUSD_M15_2023_2026.csv")
    days_all = sorted(set(base.index.date))
    T = build(base)
    T.to_csv("part50_er_trades.csv", index=False)
    out = []
    P = out.append
    P("=" * 120)
    P("第五十部分：ER 回檔順勢策略樣本外（規則同 part29_final.py；2023~2025 樣本外，2026 樣本內）")
    P("=" * 120)
    periods = [("2023（樣本外）", 2023, 2024), ("2024（樣本外）", 2024, 2025), ("2025（樣本外）", 2025, 2026),
               ("2023~2025 樣本外合計", 2023, 2026), ("2026（樣本內）", 2026, 2027)]
    sizing = [("1 單位", False, False), ("1R 且同向加 1", False, True), ("同向 2 單位 + 1R 加 1", True, True)]
    for lab, y0, y1 in periods:
        S = T[(T.xtime.dt.year >= y0) & (T.xtime.dt.year < y1)]
        days = [d for d in days_all if y0 <= d.year < y1]
        P(f"\n【{lab}】{len(days)} 個交易日，{len(S)} 筆（{len(S) / len(days):.2f}/天）")
        for sl, x2, add in sizing:
            w = np.where(S.htf & x2, 2.0, 1.0)
            wa = np.where(S.htf & add & S.added, 1.0, 0.0)
            R = S.R * w + S.addR * wa
            usd = S.usd * w + S.add_usd * wa
            s = stats(S, R, usd, days)
            P(f"  {sl:<18} 勝率 {s['win']:.0%} PF {s['pf']:.2f} 每筆 {s['avg']:+.3f}R（t={s['t']:+.1f}） 總 {s['tot']:+6.1f}R "
              f"回撤 {s['mdd']:5.1f}R 獲利/回撤 {s['tot'] / s['mdd']:5.1f} Sharpe {s['sh']:+.2f} 每盎司 ${s['usd']:+,.0f} 最好5天占 {s['top5']:.0%}")
        g = S.groupby("leg").R.agg(["size", "mean", "sum"])
        P("  各組（1 單位）：" + "  ".join(f"{k} {int(r['size'])} 筆 {r['mean']:+.3f}R 總 {r['sum']:+.1f}R" for k, r in g.iterrows()))
    P("\n逐月 R（1 單位）")
    mo = T.groupby(T.xtime.dt.strftime("%Y-%m")).R.sum()
    for y in sorted(set(i[:4] for i in mo.index)):
        P(f"  {y}: " + "  ".join(f"{k[5:]}:{v:+.1f}" for k, v in mo.items() if k.startswith(y)))
    P(f"  正報酬月份 {np.mean(mo > 0):.0%}（{int((mo > 0).sum())}/{len(mo)}）")
    txt = "\n".join(out)
    print(txt)
    open("results_part50.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
