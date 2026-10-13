"""第二十三部分：只用 M5 / M15 / M30 / H1 重做相鄰兩根 K 棒研究，並結合第二十二部分的發現：
1. 大小用「時段標準化」（÷ 同一時刻過去 20 天平均），排除開盤效應
2. 「變大」改成倍數分組（第二十二部分 D1 顯示放大倍數越大越明顯）
3. 依時段（亞洲 / 倫敦 / 紐約）分開
4. 加入「預期下一根大小」（大小可預測）作為過濾，讓報酬能蓋過點差
5. 嚴格流程：所有格子只在發現期（2023/6~2024/12）排名挑選，驗證期（2025/1~2026/10）只看一次
"""
import numpy as np
import pandas as pd
from load import load

SPLIT = pd.Timestamp("2025-01-01")
TFS = [("M5", "5min", 5), ("M15", "15min", 15), ("M30", "30min", 30), ("H1", "1h", 60)]
AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "tickvol": "sum", "spread": "mean"}
HS = (1, 3, 6)


def pct(x):
    return f"{100 * x:5.1f}%"


def tstat(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 2 else np.nan


def session(h):
    return np.select([h < 9, h < 15], ["亞洲", "倫敦"], "紐約")   # 伺服器時間


def tod_norm(s):
    """÷ 同一時刻（時:分）過去 20 天平均；只用過去資料。"""
    return s / s.groupby(s.index.time).transform(lambda x: x.shift().rolling(20, min_periods=10).mean())


def build(m1, rule, mins):
    df = m1.resample(rule, label="left", closed="left").agg(AGG).dropna()
    f = pd.DataFrame(index=df.index)
    body, rng_ = (df.close - df.open).abs(), df.high - df.low
    f["dir"] = np.sign(df.close - df.open)
    atr = rng_.rolling(20).mean()
    gap = df.index.to_series().diff().dt.total_seconds().div(60)
    f["ok"] = (gap <= 2 * mins) & (gap > 0)
    f["same"] = (f.dir == f.dir.shift())
    for k, s in (("body", body), ("rng", rng_)):
        f[f"{k}_raw"] = s / s.shift()                       # 這根 ÷ 前一根
        n = tod_norm(s)
        f[f"{k}_tod"] = n / n.shift()                        # 時段標準化後 這根 ÷ 前一根
    f["body_atr"] = body / atr.shift()
    # 預期下一根大小：下一個時刻過去 20 天的平均全長 ÷ ATR20（事前可知）
    exp_next = rng_.groupby(rng_.index.time).transform(lambda x: x.shift().rolling(20, min_periods=10).mean())
    f["exp_next"] = (exp_next.shift(-1) / atr)
    f["cost_atr"] = df.spread * 0.01 / atr
    f["cost_usd"] = df.spread * 0.01
    f["atr"] = atr
    t = df.index.to_series()
    for h in HS:
        span = (t.shift(-h) - t).dt.total_seconds().div(60)
        valid = span <= h * mins + 65                         # 中間不跨週末（容許每日休市 1 小時）
        f[f"r{h}"] = ((df.close.shift(-h) - df.close) * f.dir / atr).where(valid)
        f[f"usd{h}"] = ((df.close.shift(-h) - df.close) * f.dir).where(valid)
    f["sess"] = session(f.index.hour)
    f["per"] = np.where(f.index < SPLIT, "發現", "驗證")
    f["year"] = f.index.year
    f = f[f.ok & (f.dir != 0) & (f.dir.shift() != 0) & f.atr.notna()].copy()
    f["exp_hi"] = f.exp_next >= f.groupby("per").exp_next.transform("median")   # 預期下一根比一般大
    return f


RB = [0, .5, 1, 1.5, 2, 3, np.inf]
RL = ["<0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", ">3"]


def basic(f, name):
    print(f"\n## {name}（n={len(f):,}）")
    for k, lab in (("body", "實體"), ("rng", "全長")):
        for norm, nl in (("raw", "原始"), ("tod", "時段標準化")):
            r = f[f"{k}_{norm}"]
            big, sml = r > 1, r < 1
            s = f.same
            print(f"  {lab}{nl:5s}: 變大 {pct(big.mean())} 變小 {pct(sml.mean())} | 變大+順 {pct((big & s).mean())} 變大+逆 {pct((big & ~s).mean())} "
                  f"變小+順 {pct((sml & s).mean())} 變小+逆 {pct((sml & ~s).mean())} | 順勢率 {pct(s.mean())}")
    # 依倍數 × 順逆 → 下一根（全期間，描述用）
    print("  實體倍數（時段標準化）× 這根與前一根同向/反向 → 下一根與這根同色率 / 下一根順這根方向報酬(ATR) / 扣點差")
    q = pd.cut(f.body_tod, RB, labels=RL)
    for s, sl in ((True, "同向"), (False, "反向")):
        row = []
        for lab in RL:
            m = (q == lab) & (f.same == s)
            sub = f[m]
            cont = (np.sign(sub.r1) > 0).mean()
            row.append(f"{lab}:{pct(cont)} {sub.r1.mean():+.3f}/{(sub.r1 - sub.cost_atr).mean():+.3f}")
        print(f"    {sl}: " + " | ".join(row))


def grid(F):
    rows = []
    for tf, f in F.items():
        for size in ("body", "rng"):
            for norm in ("raw", "tod"):
                q = pd.cut(f[f"{size}_{norm}"], RB, labels=RL)
                for same in (True, False):
                    for lab in RL:
                        base = (q == lab) & (f.same == same)
                        for sess in ("全部", "亞洲", "倫敦", "紐約"):
                            m = base if sess == "全部" else base & (f.sess == sess)
                            for ef in ("不限", "預期大"):
                                mm = m if ef == "不限" else m & f.exp_hi
                                sub = f[mm]
                                for h in HS:
                                    d, v = sub[sub.per == "發現"], sub[sub.per == "驗證"]
                                    if len(d) < 300 or len(v) < 100:
                                        continue
                                    rd, rv = d[f"r{h}"], v[f"r{h}"]
                                    rows.append(dict(tf=tf, size=size, norm=norm, rel="同向" if same else "反向", ratio=lab, sess=sess, exp=ef, h=h,
                                                     n_d=len(d), n_v=len(v), d=rd.mean(), t_d=tstat(rd), v=rv.mean(), t_v=tstat(rv),
                                                     cost_d=d.cost_atr.mean(), cost_v=v.cost_atr.mean()))
    return pd.DataFrame(rows)


def trade(f, cond, side, h, mins):
    """不重疊：持倉中出現的訊號略過。side=+1 順這根方向，-1 反向。回傳每筆淨報酬（ATR）與美元。"""
    sub = f[cond].dropna(subset=[f"r{h}"])
    out, last_exit = [], pd.Timestamp(0)
    for t, r in sub.iterrows():
        if t < last_exit:
            continue
        out.append((t, side * r[f"r{h}"] - r.cost_atr, side * r[f"usd{h}"] - r.cost_usd))
        last_exit = t + pd.Timedelta(minutes=mins * h)
    return pd.DataFrame(out, columns=["t", "R", "usd"]).set_index("t")


def line(T):
    if len(T) == 0:
        return "n=0"
    g, l = T.R[T.R > 0].sum(), -T.R[T.R < 0].sum()
    yr = T.groupby(T.index.year).R.mean()
    return (f"n={len(T):5d} 淨 {T.R.mean():+.3f}ATR (t{tstat(T.R):+4.1f}) 勝率 {pct((T.R > 0).mean())} PF {g / l:.2f} | 逐年 "
            + " ".join(f"{y}:{v:+.3f}" for y, v in yr.items()))


def main():
    m1 = load("data/XAUUSD_M1_2023_2026.csv")
    F = {name: build(m1, rule, mins) for name, rule, mins in TFS}
    mins_of = {n: m for n, _, m in TFS}
    print(f"資料 {m1.index[0]} ~ {m1.index[-1]}；發現期 < {SPLIT.date()} ≤ 驗證期；時段（伺服器時間）亞洲 1-9 / 倫敦 9-15 / 紐約 15-24")

    print("\n# A. 基本機率：原始 vs 時段標準化")
    for name, f in F.items():
        basic(f, name)

    print("\n\n# B. 全格子掃描（時框 4 × 實體/全長 × 原始/標準化 × 同/反向 × 倍數 6 × 時段 4 × 預期大小 2 × 持有 1/3/6 根）")
    G = grid(F)
    G["net_d"] = G.d - G.cost_d
    G["net_v"] = G.v - G.cost_v
    print(f"  有效格子 {len(G)} 個（發現期 n≥300、驗證期 n≥100）")
    # 全局檢查：發現期方向 → 驗證期是否同號
    for thr in (0, 2, 3, 4):
        sel = G[G.t_d.abs() >= thr]
        agree = (np.sign(sel.d) == np.sign(sel.v)).mean()
        print(f"  發現期 |t|≥{thr}: {len(sel):5d} 格，驗證期同號 {pct(agree)}；扣點差後兩段都賺（照發現期方向做）{pct(((np.sign(sel.d) * sel.d - sel.cost_d > 0) & (np.sign(sel.d) * sel.v - sel.cost_v > 0)).mean())}")
    print("\n  發現期扣點差後淨值最高的 20 格（方向照發現期：順 = 順這根做，反 = 反這根做），驗證期只看一次：")
    G["side"] = np.sign(G.d)
    G["netd_side"] = G.side * G.d - G.cost_d
    G["netv_side"] = G.side * G.v - G.cost_v
    G["tv_side"] = G.side * G.t_v
    top = G[G.t_d.abs() >= 3].sort_values("netd_side", ascending=False).head(20)
    for _, r in top.iterrows():
        print(f"   {r.tf:3s} {r['size']:4s} {r.norm:3s} {r.rel} 倍數{r.ratio:>6s} {r.sess} {r.exp:3s} 持有{r.h} 做{'順' if r.side > 0 else '反'} | "
              f"發現 n={r.n_d:5d} {r.netd_side:+.3f} (t{abs(r.t_d):.1f}) → 驗證 n={r.n_v:5d} {r.netv_side:+.3f} (t{r.tv_side:+.1f})")
    G.to_csv("part23_grid.csv", index=False)

    print("\n\n# C. 事先登記的假設（來自第二十二部分 D1 的「實體變大＋同向 → 延續」），搬到日內：")
    print("   規則：實體（時段標準化）≥ 前一根 1.5 倍 且 同向 → 順這根方向，持有 h 根，扣點差；不重疊")
    for name, f in F.items():
        for h in HS:
            base = (f.body_tod >= 1.5) & f.same
            for ef in ("不限", "預期大"):
                c = base if ef == "不限" else base & f.exp_hi
                T = trade(f, c, 1, h, mins_of[name])
                print(f"  {name:3s} 持有{h} {ef:3s}: {line(T)}")

    print("\n\n# D. B 段發現期前 5 名，當成交易（不重疊、扣點差），逐年")
    for _, r in top.head(5).iterrows():
        f = F[r.tf]
        q = pd.cut(f[f"{r['size']}_{r.norm}"], RB, labels=RL)
        c = (q == r.ratio) & (f.same == (r.rel == "同向"))
        if r.sess != "全部":
            c &= f.sess == r.sess
        if r.exp == "預期大":
            c &= f.exp_hi
        T = trade(f, c, int(r.side), int(r.h), mins_of[r.tf])
        print(f"  {r.tf} {r['size']} {r.norm} {r.rel} {r.ratio} {r.sess} {r.exp} 持有{r.h} 做{'順' if r.side > 0 else '反'}")
        print(f"     發現期 {line(T[T.index < SPLIT])}")
        print(f"     驗證期 {line(T[T.index >= SPLIT])}")


if __name__ == "__main__":
    main()
