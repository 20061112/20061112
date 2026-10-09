"""第二十八部分：加入掉期成本、停損上限（只拉近不跳單）、週五收盤平倉 → results_part28.txt

主體：A/B/C、tp2、最長 2L 根、不做星期一、週五最後一根 K 棒強制平倉（不過週末）。
掉期：每跨過一次 broker 日界線扣 swap 美元/盎司；週三 → 週四那次算 3 倍（三倍掉期日）。
      多空都扣（保守）；1 手 = 100 盎司，$30~70/手/晚 = $0.30~0.70/盎司/晚。加碼單同樣扣主單的晚數（保守）。
停損上限：停損距離 > 上限時把停損拉近到「進場價 ∓ 上限」，照樣開單。
部位：固定 1 單位 vs 同向（H1+H4 ER20 同向）進場 2 單位 + 到 +1R 再加 1 單位。
"""
import numpy as np
import pandas as pd
from part23_ma_cycle import load_m1
from part25_exits import CFGS, prep, entries, metrics
from part25_ext import tf_er_done
from part27_sizing import sim


def nights(idx, e, j):
    """e..j 之間跨過幾次日界線（週三 → 週四算 3 次）。"""
    d = idx[e:j + 1]
    if len(d) < 2:
        return 0
    days = pd.DatetimeIndex(d.normalize().unique())
    n = 0
    for a, b in zip(days[:-1], days[1:]):
        n += 3 if a.dayofweek == 2 else 1
    return n


def main():
    m1 = load_m1()
    days = sorted(set(m1.index.date))
    cache, sigs = {}, {}
    for name, (tf, L, M, S, a, cool) in CFGS.items():
        B = cache.setdefault(tf, prep(m1, tf))
        if "h1" not in B:
            B["h1"], B["h4"] = tf_er_done(m1, B, "1h"), tf_er_done(m1, B, "4h")
        t, side = entries(B, L, M, S, a, cool)
        keep = B["idx"][t].dayofweek != 0
        sigs[name] = (B, t[keep], side[keep], L, M)

    def book(cap=None, swap=0.0):
        rows = []
        for name, (B, t, side, L, M) in sigs.items():
            for k, s in zip(t, side):
                r = sim(B, k, s, M, L, cap_usd=cap, cap_mode="cap" if cap else None, flat="week")
                if not r:
                    continue
                nt = nights(B["idx"], k + 1, r["j"])
                sw = swap * nt
                r["usd"] -= sw
                r["R"] -= sw / r["Rusd"]
                if r["added"]:
                    r["add_usd"] -= sw
                    r["addR"] -= sw / r["Rusd"]
                htf = (s * B["h1"][k] > 0) and (s * B["h4"][k] > 0)
                rows.append(dict(leg=name, time=B["idx"][k], xtime=B["idx"][r["j"]], bars=r["j"] - k, htf=htf,
                                 nights=nt, swap=sw, **r))
        return pd.DataFrame(rows)

    def row(T, scale):
        u0 = np.where(T.htf & scale, 2.0, 1.0)
        ua = np.where(T.htf & scale & T.added, 1.0, 0.0)
        R = T.R * u0 + T.addR * ua
        usd = T.usd * u0 + T.add_usd * ua
        m = metrics(T.assign(R=R), days)
        return dict(n=m["n"], 每天=m["每天"], 總R=m["總R"], avgR=round(float(R.mean()), 3), PF=m["PF"], MDD_R=m["MDD"],
                    P_MDD=m["P_MDD"], Sharpe=m["Sharpe"], 每筆美元=round(float(usd.mean()), 2),
                    總美元=round(float(usd.sum())), 最大單筆虧損=round(float(usd.min()), 1),
                    掉期合計=round(float((T.swap * (u0 + ua)).sum())), 前半R=round(float(R[T.time < "2026-06-01"].sum()), 1),
                    後半R=round(float(R[T.time >= "2026-06-01"].sum()), 1))

    out = []
    P = out.append
    T0 = book()
    P(f"週五平倉版：{len(T0)} 筆，跨日 {(T0.nights > 0).sum()} 筆（{(T0.nights > 0).mean():.0%}），"
      f"平均過夜 {T0.nights[T0.nights > 0].mean():.2f} 晚（含週三 ×3），最多 {T0.nights.max()} 晚")

    P("\n=== 1. 掉期成本的影響（無停損上限）===")
    rows = []
    for sw in (0.0, 0.3, 0.5, 0.7, 1.0):
        T = book(swap=sw)
        for lab, sc in (("固定 1 單位", False), ("同向 2 單位 + 1R 加 1", True)):
            rows.append(dict(掉期_每盎司每晚=sw, 部位=lab, **row(T, sc)))
    P(pd.DataFrame(rows).to_string(index=False))

    P("\n=== 2. 停損上限（超過就拉近，照樣開單；掉期 $0.7/盎司/晚）===")
    P("停損距離分位（美元/盎司）：" + ", ".join(f"{q:.0%} ${T0.R0usd.quantile(q):.0f}" for q in (.5, .75, .8, .85, .9, .95, .99)))
    rows = []
    for cap in (None, 120, 100, 90, 80, 70, 60, 50):
        T = book(cap=cap, swap=0.7)
        hit = int((T.R0usd > cap).sum()) if cap else 0
        for lab, sc in (("固定 1 單位", False), ("同向 2 單位 + 1R 加 1", True)):
            rows.append(dict(上限=cap or "無", 被拉近=hit, 部位=lab, **row(T, sc)))
    P(pd.DataFrame(rows).to_string(index=False))

    P("\n=== 3. 被拉近的單本身（上限 $80 vs 不設上限，掉期 0.7）===")
    Tn, Tc = book(swap=0.7), book(cap=80, swap=0.7)
    big = Tn.R0usd > 80
    P(f"停損 > $80 的 {big.sum()} 筆：不設上限 每筆 ${Tn.usd[big].mean():+.1f}、勝率 {(Tn.usd[big] > 0).mean():.0%}；"
      f"拉近到 $80 後 每筆 ${Tc.usd[big.to_numpy()].mean():+.1f}、勝率 {(Tc.usd[big.to_numpy()] > 0).mean():.0%}")

    P("\n=== 4. 最終建議版逐月（上限 $80 拉近、掉期 0.7、週五平倉、同向 2 單位 + 1R 加 1；美元為每單位 1 盎司）===")
    u0 = np.where(Tc.htf, 2.0, 1.0)
    ua = np.where(Tc.htf & Tc.added, 1.0, 0.0)
    Tc = Tc.assign(Rw=Tc.R * u0 + Tc.addR * ua, usdw=Tc.usd * u0 + Tc.add_usd * ua)
    mo = Tc.groupby(Tc.xtime.dt.to_period("M")).agg(筆數=("Rw", "size"), R=("Rw", "sum"), 美元=("usdw", "sum")).round(1)
    P(mo.T.to_string())
    Tc.to_csv("part28_final_trades.csv", index=False)
    open("results_part28.txt", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
