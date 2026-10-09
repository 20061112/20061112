"""第二十九部分：整合最佳化（固定：週五最後一根平倉、不做星期一、掉期 $0.7/盎司/晚、週三 ×3）

只用前半段（1/2~5/31）挑參數，後半段（6/1~10/8）只看結果。
候選都是前面已經有證據的選項，不加新概念：
  每組（leg）  : A M15 40/20/5 a.3 冷卻 5、B M30 20/10/3 a.3 冷卻 3、C M15 20/10/3 a.4 冷卻 10、D M30 14/7/2 a.4 冷卻 7
                 停利 tp ∈ {1.5, 2}R × 最長持有 ∈ {L, 2L} 根 → 每組各自用前半 Sharpe 挑
  組合層級     : 組合 ⊆ {A,B,C,D}（至少 2 組）
                 停損上限 ∈ {無, $100（拉近）}
                 週五幾點後不開新單 ∈ {不限, 20 點, 18 點, 16 點}（broker 時間；反正會被週五平倉切掉）
                 部位 ∈ {1 單位, 同向 2 單位, 1R 且同向加 1, 同向 2 單位 + 1R 加 1}
                 亞洲時段（訊號收盤 0~8 點）權重 ∈ {1, 0.5}
挑選標準：前半段每日 R 的 Sharpe（年化）。
→ results_part29.txt、part29_grid.csv、part29_best_trades.csv
"""
import itertools
import numpy as np
import pandas as pd
from part23_ma_cycle import load_m1, SPLIT
from part25_exits import prep, entries
from part25_ext import tf_er_done
from part27_sizing import sim
from part28_final import nights

LEGS = {"A": ("15min", 40, 20, 5, 0.3, 5), "B": ("30min", 20, 10, 3, 0.3, 3),
        "C": ("15min", 20, 10, 3, 0.4, 10), "D": ("30min", 14, 7, 2, 0.4, 7)}
SWAP = 0.7


def pmetrics(T, R, days):
    """組合指標：R 已經是加權後每筆的 R。"""
    if len(T) == 0:
        return {}
    dr = pd.Series(R.to_numpy(), index=T.xtime.dt.date).groupby(level=0).sum()
    out = {}
    for lab, dd in (("全", days), ("前", [d for d in days if pd.Timestamp(d) < SPLIT]),
                    ("後", [d for d in days if pd.Timestamp(d) >= SPLIT])):
        s = dr.reindex(dd, fill_value=0)
        eq = s.cumsum().to_numpy()
        mdd = -(eq - np.maximum.accumulate(np.r_[0, eq])[1:]).min()
        out[f"{lab}Sharpe"] = s.mean() / s.std() * np.sqrt(252) if s.std() > 0 else np.nan
        out[f"{lab}R"] = s.sum()
        out[f"{lab}MDD"] = mdd
    v = R.to_numpy()
    out["PF"] = v[v > 0].sum() / -v[v < 0].sum()
    out["n"] = len(v)
    out["每天"] = len(v) / len(days)
    out["P_MDD"] = out["全R"] / out["全MDD"] if out["全MDD"] > 0 else np.nan
    return out


def main():
    m1 = load_m1()
    days = sorted(set(m1.index.date))
    cache = {}
    tables = {}
    for leg, (tf, L, M, S, a, cool) in LEGS.items():
        B = cache.setdefault(tf, prep(m1, tf))
        if "h1" not in B:
            B["h1"], B["h4"] = tf_er_done(m1, B, "1h"), tf_er_done(m1, B, "4h")
        step = B["idx"][1] - B["idx"][0]
        t, side = entries(B, L, M, S, a, cool)
        keep = B["idx"][t].dayofweek != 0
        t, side = t[keep], side[keep]
        for tp, hold, cap in itertools.product((1.5, 2), (L, 2 * L), (None, 100)):
            rows = []
            for k, s in zip(t, side):
                r = sim(B, k, s, M, hold // 2, tp=tp, cap_usd=cap, cap_mode="cap" if cap else None, flat="week")
                if not r:
                    continue
                sw = SWAP * nights(B["idx"], k + 1, r["j"])
                ts = B["idx"][k] + step                                    # 訊號 K 棒收盤時間
                rows.append(dict(leg=leg, time=B["idx"][k], xtime=B["idx"][r["j"]], hour=ts.hour, dow=ts.dayofweek,
                                 htf=(s * B["h1"][k] > 0) and (s * B["h4"][k] > 0), added=r["added"],
                                 R=r["R"] - sw / r["Rusd"], addR=(r["addR"] - sw / r["Rusd"]) if r["added"] else 0.0,
                                 usd=r["usd"] - sw, add_usd=(r["add_usd"] - sw) if r["added"] else 0.0, Rusd=r["Rusd"]))
            tables[(leg, tp, hold, cap)] = pd.DataFrame(rows)

    out = []
    P = out.append
    P("=== 1. 每組各自挑停利 / 最長持有（前半 Sharpe，固定 1 單位、無上限）===")
    rows, pick = [], {}
    for leg, (tf, L, *_ ) in LEGS.items():
        best = None
        for tp, hold in itertools.product((1.5, 2), (L, 2 * L)):
            T = tables[(leg, tp, hold, None)]
            m = pmetrics(T, T.R, days)
            rows.append(dict(組=leg, tp=tp, 最長=hold, n=m["n"], PF=round(m["PF"], 2), 前Sharpe=round(m["前Sharpe"], 2),
                             後Sharpe=round(m["後Sharpe"], 2), 前R=round(m["前R"], 1), 後R=round(m["後R"], 1)))
            if best is None or m["前Sharpe"] > best[0]:
                best = (m["前Sharpe"], tp, hold)
        pick[leg] = best[1:]
    P(pd.DataFrame(rows).to_string(index=False))
    P("前半挑出：" + "、".join(f"{k} tp{v[0]} 最長{v[1]}根" for k, v in pick.items()))

    P("\n=== 2. 組合層級掃描（每組用上面挑出的 tp / 最長持有）===")
    SIZES = {"1 單位": (False, False), "同向 2 單位": (True, False), "1R 且同向加 1": (False, True),
             "同向 2 + 1R 加 1": (True, True)}
    grid = []
    for r_ in range(2, 5):
        for combo in itertools.combinations(LEGS, r_):
            for cap in (None, 100):
                base = pd.concat([tables[(g, *pick[g], cap)] for g in combo]).sort_values("time").reset_index(drop=True)
                for cut in (None, 20, 18, 16):
                    T = base if cut is None else base[~((base.dow == 4) & (base.hour >= cut))]
                    for sz, (x2, add) in SIZES.items():
                        for asia in (1.0, 0.5):
                            w = np.where(T.htf & x2, 2.0, 1.0) * np.where(T.hour <= 8, asia, 1.0)
                            wa = np.where(T.htf & add & T.added, 1.0, 0.0) * np.where(T.hour <= 8, asia, 1.0)
                            R = T.R * w + T.addR * wa
                            m = pmetrics(T, R, days)
                            usd = T.usd * w + T.add_usd * wa
                            grid.append(dict(組合="+".join(combo), 上限=cap or "無", 週五截止=cut or "不限", 部位=sz,
                                             亞洲權重=asia, **{k: round(float(v), 2) for k, v in m.items()},
                                             每筆美元=round(float(usd.mean()), 2), 總美元=round(float(usd.sum()))))
    G = pd.DataFrame(grid)
    G.to_csv("part29_grid.csv", index=False)
    G["後排名%"] = G.後Sharpe.rank(pct=True).round(2)
    cols = ["組合", "上限", "週五截止", "部位", "亞洲權重", "n", "每天", "PF", "全Sharpe", "全R", "全MDD", "P_MDD",
            "前Sharpe", "後Sharpe", "後R", "後MDD", "後排名%", "每筆美元", "總美元"]
    P(f"共 {len(G)} 種組合。前半 Sharpe 最高 15 名（看它們在後半的表現）：")
    P(G.sort_values("前Sharpe", ascending=False)[cols].head(15).to_string(index=False))
    rk = G.前Sharpe.rank()
    P(f"\n前半 Sharpe 與後半 Sharpe 的等級相關：{rk.corr(G.後Sharpe.rank()):.2f}；"
      f"前半前 10% 的組合，後半 Sharpe 平均 {G[rk >= rk.quantile(0.9)].後Sharpe.mean():.2f}，全部平均 {G.後Sharpe.mean():.2f}")

    P("\n=== 3. 每個選項的平均效果（所有其他選項平均，看前後兩半是否一致）===")
    for col in ("上限", "週五截止", "部位", "亞洲權重"):
        P(G.groupby(col)[["前Sharpe", "後Sharpe", "前R", "後R", "全MDD", "P_MDD"]].mean().round(2).to_string())
        P("")
    G["組數"] = G.組合.str.count(r"\+") + 1
    P(G.groupby("組數")[["前Sharpe", "後Sharpe", "全R", "全MDD", "P_MDD"]].mean().round(2).to_string())
    P(G.assign(含D=G.組合.str.contains("D"))
      .groupby("含D")[["前Sharpe", "後Sharpe", "全R", "全MDD", "P_MDD"]].mean().round(2).to_string())

    best = G.sort_values("前Sharpe", ascending=False).iloc[0]
    ref = G[(G.組合 == "A+B+C") & (G.上限 == 100) & (G.週五截止 == "不限") & (G.部位 == "同向 2 + 1R 加 1") & (G.亞洲權重 == 1.0)].iloc[0]
    P("\n=== 4. 前半最佳 vs 上一版建議（A+B+C、上限 100、同向 2 + 1R 加 1）===")
    P(pd.DataFrame([best[cols], ref[cols]], index=["前半最佳", "上一版"]).T.to_string())
    open("results_part29.txt", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
