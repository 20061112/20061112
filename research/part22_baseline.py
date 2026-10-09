"""第二十二部分：TD 訊號 vs 隨機進場基準（同樣的停損結構、同樣方向比例）。

基準：隨機挑 K 棒、方向隨機，停損 = 前 9 根極值外 0.1 ATR，其餘與 part22_td.simulate 相同。
另外列出未扣點差的 R，以及點差佔 R 的比例，分辨「訊號沒用」與「成本吃掉」。
"""
import numpy as np
import pandas as pd
from load import load
from td import resample, atr, td_signals
from part22_td import simulate, SPLIT

TFS = ["1min", "5min", "15min", "30min", "1h", "4h"]
CFGS = [(1.0, 10), (2.0, 20), (3.0, 40)]
rng = np.random.default_rng(0)


def random_sigs(df, A, m):
    l, h = df.low.to_numpy(), df.high.to_numpy()
    i = np.sort(rng.choice(np.arange(20, len(df) - 2), m, replace=False))
    d = rng.choice([1, -1], m)
    lo = pd.Series(l).rolling(9).min().to_numpy()
    hi = pd.Series(h).rolling(9).max().to_numpy()
    ext = np.where(d == 1, lo[i], hi[i])
    s = pd.DataFrame(dict(i=i, dir=d, ext=ext))
    return s[~np.isnan(A[s.i])].reset_index(drop=True)


def run(df, sig, A, tp, H):
    """回傳 (淨 R, 毛 R, 點差/R)。"""
    df0 = df.copy()
    df0["spread"] = 0
    net = simulate(df, sig, A, tp, H)
    gross = simulate(df0, sig, A, tp, H)
    return net, gross


def main():
    m1 = load("data/XAUUSD_M1_2026.csv")
    L = []
    for tf in TFS:
        df = resample(m1, tf)
        A = atr(df)
        sig = td_signals(df)
        sig = sig[~np.isnan(A[sig.i])].reset_index(drop=True)
        o = df.open.to_numpy()
        spr = df.spread.to_numpy() * 0.01
        L.append(f"\n=== {tf} ===")
        groups = {"S9 all": sig[sig.kind == "S9"], "S9 perf": sig[(sig.kind == "S9") & sig.perf],
                  "C13": sig[sig.kind == "C13"]}
        rnd = random_sigs(df, A, min(20000, len(df) // 4))
        groups["RANDOM"] = rnd
        for g, s in groups.items():
            s = s.reset_index(drop=True)
            ii = np.minimum(s.i.to_numpy() + 1, len(df) - 1)
            risk = (o[ii] - (s.ext.to_numpy() - s.dir.to_numpy() * 0.1 * A[s.i.to_numpy()])) * s.dir.to_numpy()
            cost = np.nanmedian(spr[ii] / np.where(risk > 0, risk, np.nan))
            parts = []
            for tp, H in CFGS:
                net, gross = run(df, s, A, tp, H)
                parts.append(f"tp{tp:.0f}/H{H}: 毛 {np.nanmean(gross):+.3f} 淨 {np.nanmean(net):+.3f}")
            L.append(f"  {g:<8} n={len(s):6d} 點差/R 中位數={cost:.3f} | " + " | ".join(parts))
    txt = "\n".join(L)
    print(txt)
    open("results_part22_baseline.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
