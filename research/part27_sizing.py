"""第二十七部分：加倉組合、最大停損上限、過夜檢查 → results_part27.txt

主體：第二十五 / 二十六部分 A/B/C、tp2、最長 2L 根、不做星期一。
1. 部位：
   base      固定 1 單位
   x2        H1+H4 同向 → 進場 2 單位
   add       到 +1R 且 H1+H4 同向 → 加 1 單位（加碼單停損 = 原進場價、目標 = 原 2R）
   x2+add    同向時進場 2 單位，到 +1R 再加 1 單位（共 3 單位）
2. 最大停損上限（每盎司美元）：
   skip  : 停損距離 > 上限就不做
   cap   : 停損距離 > 上限就把停損拉近到「進場價 ∓ 上限」（R 跟著變小）
3. 過夜：持倉跨過 broker 交易日（23:59 → 隔天 01:00）或週末的比例與損益；
   flat  : 每天最後一根 K 棒收盤強制平倉（不過夜）
   flatwk: 只在星期五最後一根強制平倉（不過週末）
損益單位：
   R 指標 = 每單位固定風險 1R（實盤以帳戶 % 固定風險下單）；加倉時以「單位數」加權
   美元 = 每單位 1 盎司（0.01 手）
"""
import numpy as np
import pandas as pd
from part23_ma_cycle import load_m1
from part25_exits import CFGS, prep, entries, metrics
from part25_ext import tf_er_done


def sim(B, k, s, M, L, tp=2, cap_usd=None, cap_mode=None, flat=None, buf=0.1):
    o, h, l, c, A, spr, idx = B["o"], B["h"], B["l"], B["c"], B["A"], B["spr"], B["idx"]
    n = len(c)
    e = k + 1
    ext = l[k - M + 1:k + 1].min() if s == 1 else h[k - M + 1:k + 1].max()
    stop = ext - s * buf * A[k]
    ep = o[e]
    R0 = s * (ep - stop)
    if not R0 > 0:
        return None
    if cap_usd is not None and R0 > cap_usd:
        if cap_mode == "skip":
            return None
        stop = ep - s * cap_usd
    R = s * (ep - stop)
    tgt = ep + s * tp * R
    end = min(e + 2 * L, n)
    a_on, a_ep, a_x = False, None, None
    reason = "time"
    for j in range(e, end):
        if a_on is True and ((l[j] <= ep) if s == 1 else (h[j] >= ep)):
            a_x = min(ep, o[j]) if s == 1 else max(ep, o[j])
            a_on = None
        if (l[j] <= stop) if s == 1 else (h[j] >= stop):
            xp = stop if j == e else (min(stop, o[j]) if s == 1 else max(stop, o[j]))
            reason = "stop"
            break
        if (h[j] >= tgt) if s == 1 else (l[j] <= tgt):
            xp, reason = tgt, "tp"
            break
        if a_on is False and ((h[j] >= ep + R) if s == 1 else (l[j] <= ep - R)):
            a_on, a_ep = True, ep + s * R
        last_of_day = j + 1 >= n or idx[j + 1].date() != idx[j].date()
        if flat == "day" and last_of_day:
            xp, reason = c[j], "flat"
            break
        if flat == "week" and last_of_day and (j + 1 >= n or idx[j + 1].dayofweek < idx[j].dayofweek):
            xp, reason = c[j], "flat"
            break
    else:
        j = end - 1
        xp = c[j]
    if a_on is True:
        a_x = xp
    cost = spr[e]
    main_usd = s * (xp - ep) - cost
    add_usd = (s * (a_x - a_ep) - cost) if a_ep is not None else 0.0
    overnight = idx[j].date() != idx[k].date()
    weekend = (idx[j] - idx[k]).days >= 2 or idx[j].dayofweek < idx[e].dayofweek
    return dict(j=j, R=main_usd / R, addR=add_usd / R, added=a_ep is not None, usd=main_usd, add_usd=add_usd,
                Rusd=R, R0usd=R0, reason=reason, overnight=overnight, weekend=weekend)


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

    def book(**kw):
        rows = []
        for name, (B, t, side, L, M) in sigs.items():
            for k, s in zip(t, side):
                r = sim(B, k, s, M, L, **kw)
                if r:
                    al = (s * B["h1"][k] > 0) and (s * B["h4"][k] > 0)
                    rows.append(dict(leg=name, time=B["idx"][k], xtime=B["idx"][r["j"]], bars=r["j"] - k,
                                     htf=al, **r))
        return pd.DataFrame(rows)

    def scheme(T, init2, add):
        u0 = np.where(T.htf & init2, 2.0, 1.0)
        ua = np.where(T.htf & add & T.added, 1.0, 0.0)
        R = T.R * u0 + T.addR * ua
        usd = T.usd * u0 + T.add_usd * ua
        m = metrics(T.assign(R=R), days)
        return dict(總R=m["總R"], PF=m["PF"], MDD_R=m["MDD"], P_MDD=m["P_MDD"], Sharpe=m["Sharpe"],
                    最多單位=int((u0 + ua).max()), 平均單位=round(float((u0 + ua).mean()), 2),
                    每筆美元=round(float(usd.mean()), 2), 總美元=round(float(usd.sum())), 前半R=round(float(R[T.time < "2026-06-01"].sum()), 1),
                    後半R=round(float(R[T.time >= "2026-06-01"].sum()), 1))

    out = []
    P = out.append
    T = book()
    P("=== 1. 部位方式（A+B+C、tp2、不做星期一）===")
    rows = [dict(方式=lab, **scheme(T, i2, ad)) for lab, i2, ad in
            (("固定 1 單位", False, False), ("同向進場 2 單位", True, False), ("+1R 且同向加 1", False, True),
             ("同向 2 單位 + 1R 再加 1", True, True))]
    P(pd.DataFrame(rows).to_string(index=False))

    # 同時持有的單位數（風險）
    u0 = np.where(T.htf, 2.0, 1.0)
    ev = pd.concat([pd.Series(u0, index=T.time), pd.Series(-u0, index=T.xtime)]).sort_index().cumsum()
    P(f"同向 2 單位 + 加碼：同時最多持有 {ev.max():.0f} 單位初始風險（加碼單停損在原進場價，不增加初始風險）")

    P("\n=== 2. 最大停損上限（每盎司美元）===")
    P("每筆停損距離分布（美元/盎司）：" + T.groupby("leg").R0usd.describe(percentiles=[.5, .75, .9])[["50%", "75%", "90%", "max"]]
      .round(1).to_string().replace("\n", " | "))
    rows = []
    for cap in (None, 80, 60, 50, 40, 30):
        for mode in (("skip", "cap") if cap else (None,)):
            Tc = book(cap_usd=cap, cap_mode=mode)
            m = metrics(Tc, days)
            rows.append(dict(上限=cap or "無", 方式=mode or "-", n=m["n"], avgR=m["avgR"], PF=m["PF"], MDD_R=m["MDD"],
                             P_MDD=m["P_MDD"], Sharpe=m["Sharpe"], 每筆美元=round(float(Tc.usd.mean()), 2),
                             最大單筆虧損美元=round(float(Tc.usd.min()), 1), 總美元=round(float(Tc.usd.sum())),
                             前半=m["前半"], 後半=m["後半"]))
    P(pd.DataFrame(rows).to_string(index=False))

    P("\n=== 3. 過夜 ===")
    P(f"原版：{len(T)} 筆中 {T.overnight.sum()} 筆跨日（{T.overnight.mean():.0%}）、{T.weekend.sum()} 筆跨週末；"
      f"跨日單 avgR {T.R[T.overnight].mean():+.3f}、當日平倉單 {T.R[~T.overnight].mean():+.3f}")
    P("各組跨日比例：" + T.groupby("leg").overnight.mean().round(2).to_string().replace("\n", " | "))
    rows = []
    for lab, fl in (("原版（可過夜）", None), ("不過週末", "week"), ("不過夜（每天收盤平倉）", "day")):
        Tf = book(flat=fl)
        m = metrics(Tf, days)
        rows.append(dict(方式=lab, n=m["n"], avgR=m["avgR"], PF=m["PF"], MDD_R=m["MDD"], P_MDD=m["P_MDD"],
                         Sharpe=m["Sharpe"], 每筆美元=round(float(Tf.usd.mean()), 2), 強制平倉=int((Tf.reason == "flat").sum()),
                         前半=m["前半"], 後半=m["後半"]))
        if fl == "day":
            Tday = Tf
    P(pd.DataFrame(rows).to_string(index=False))
    P("\n不過夜版本 + 部位方式：")
    P(pd.DataFrame([dict(方式=lab, **scheme(Tday, i2, ad)) for lab, i2, ad in
                    (("固定 1 單位", False, False), ("同向 2 單位 + 1R 再加 1", True, True))]).to_string(index=False))
    open("results_part27.txt", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
