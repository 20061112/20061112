"""第八部分探索 2：前一天方向的延續性、與 FADE 排除時段的動能（M15，2/2~10/8，broker 時間）。"""
import numpy as np
import pandas as pd
from load import load

d = load("data/XAUUSD_M15_full.csv")
pc = d.close.shift()
tr = pd.concat([d.high - d.low, (d.high - pc).abs(), (d.low - pc).abs()], axis=1).max(axis=1)
d["atr"] = tr.rolling(14).mean()
day = (d.index - pd.Timedelta(hours=1)).normalize()           # broker 交易日 01:00~23:59
D = d.groupby(day).agg(open=("open", "first"), close=("close", "last"), high=("high", "max"), low=("low", "min"),
                       atr=("atr", "mean"), n=("close", "size"))
D = D[D.n >= 60]
D["ret"] = D.close - D.open
D["prev_dir"] = np.sign(D.ret.shift(1))
D.loc[D.index.dayofweek == 0, "prev_dir"] = 0                # 星期一沒有前一天（同 FADE 規格）


def seg_ret(h0, h1):
    """每個交易日 broker h0:00 → h1:00 的報酬（點）"""
    out = {}
    for k, g in d.groupby(day):
        a, b = g.between_time(f"{h0:02d}:00", f"{h0:02d}:14"), g.between_time(f"{h1 - 1:02d}:45", f"{h1 - 1:02d}:59")
        if len(a) and len(b):
            out[k] = b.close.iloc[-1] - a.open.iloc[0]
    return pd.Series(out)


if __name__ == "__main__":
    x = D[D.prev_dir != 0]
    r = x.prev_dir * x.ret
    print(f"交易日 {len(D)}；有前一天的 {len(x)} 天")
    print(f"全天順前一天方向：平均 {r.mean():+.2f} 點，中位 {r.median():+.2f}，勝率 {(r > 0).mean():.2f}，t={r.mean() / r.std() * np.sqrt(len(r)):.2f}")
    half = x.index < pd.Timestamp("2026-06-01")
    print(f"  2~5 月 {r[half].mean():+.2f}（{half.sum()} 天）  6~10 月 {r[~half].mean():+.2f}（{(~half).sum()} 天）")
    print("\n各時段順前一天方向的平均報酬（點），t 值，前後半")
    for h0, h1, name in ((1, 10, "亞洲 01-10"), (10, 13, "倫敦開盤 10-13（FADE 排除）"), (13, 15, "13-15"),
                         (15, 19, "倫敦紐約重疊 15-19"), (19, 21, "19-21"), (21, 23, "COMEX 收盤後 21-23（FADE 排除）")):
        s = seg_ret(h0, h1).reindex(x.index).dropna()
        rr = s * x.prev_dir.reindex(s.index)
        hh = rr.index < pd.Timestamp("2026-06-01")
        print(f"  {name:28s} {rr.mean():+6.2f}  t={rr.mean() / rr.std() * np.sqrt(len(rr)):+.2f}  勝率 {(rr > 0).mean():.2f}  前 {rr[hh].mean():+.2f} 後 {rr[~hh].mean():+.2f}")
    print("\n亞洲盤區間突破：10:00~13:00 第一次突破亞洲(01-10)高/低點，順突破方向持有到 19:00")
    res = []
    for k, g in d.groupby(day):
        asia = g.between_time("01:00", "09:59")
        lon = g.between_time("10:00", "12:59")
        if len(asia) < 30 or len(lon) < 10:
            continue
        hi, lo = asia.high.max(), asia.low.min()
        for t, row in lon.iterrows():
            s = 1 if row.high > hi else -1 if row.low < lo else 0
            if s:
                ep = hi if s == 1 else lo
                end = g[g.index < t.normalize() + pd.Timedelta(hours=19)]
                stop = lo if s == 1 else hi                       # 停損 = 亞洲區間另一側
                after = g[(g.index > t) & (g.index < t.normalize() + pd.Timedelta(hours=19))]
                hit = (after.low <= stop).any() if s == 1 else (after.high >= stop).any()
                xp = stop if hit else end.close.iloc[-1]
                res.append(dict(day=k, side=s, pnl=s * (xp - ep), R=abs(ep - stop), with_prev=s * D.prev_dir.get(k, 0)))
                break
    B = pd.DataFrame(res)
    B["pnl_R"] = B.pnl / B.R
    hh = B.day < pd.Timestamp("2026-06-01")
    print(f"  {len(B)} 天：每筆 {B.pnl.mean():+.2f} 點 / {B.pnl_R.mean():+.3f}R，t={B.pnl.mean() / B.pnl.std() * np.sqrt(len(B)):.2f}，前 {B.pnl[hh].mean():+.2f} 後 {B.pnl[~hh].mean():+.2f}")
    for v, nm in ((1, "順前一天"), (-1, "逆前一天")):
        z = B[B.with_prev == v]
        print(f"  {nm}：{len(z)} 天，每筆 {z.pnl.mean():+.2f} 點 / {z.pnl_R.mean():+.3f}R")
