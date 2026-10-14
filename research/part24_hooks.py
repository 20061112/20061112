"""第二十四部分 (1)：各指標「超買/超賣後拐頭」再進場。
輸出：python3 part24_hooks.py > results_part24_hooks.txt

指標：RSI14、Stoch %K(14,3)、CCI20、MACD 柱狀體、MACD 線、OBV z 值（OBV 與 50 根均線差 / 50 根標準差）。
極端區兩種定義：
  固定門檻：RSI 30/20、Stoch 20/10、CCI −100/−200（MACD、OBV 沒有固定門檻）
  百分位  ：指標在過去 500 根中的百分位 <= 10% / 3%（六個指標都可用）
拐頭：指標在 j-1 為局部低點（ind[j-1] < ind[j-2] 且 ind[j] > ind[j-1]），且 ind[j-1] 在極端區 → j+1 開盤做反轉。
對照：「不等拐頭」= 指標第一次進入極端區那根收盤後直接進場。
出場統一：停損 1.5 ATR、停利 3 ATR、32 根（part24_common.sim）。
另外依 EMA200（前一根收盤）分成「順勢拐頭」（多頭中超賣拐頭買、空頭中超買拐頭賣）與「逆勢拐頭」。
頂部（超買）用 mirror 後同一段程式。
"""
import numpy as np
import pandas as pd
from part22_divergence import get_frames, indicators, mirror, SPLIT
from part24_common import sim, split_line, daily_regime


def obv_z(X):
    s = pd.Series(X["obv"])
    return ((s - s.rolling(50).mean()) / s.rolling(50).std()).to_numpy()


def series_for(X):
    """回傳做多空間（找底部）的指標序列：值越小越超賣。"""
    macd = pd.Series(X["macd"])
    sig = macd.ewm(span=9, adjust=False).mean()
    return {"rsi": X["rsi"], "stoch": X["stoch"], "cci": X["cci"],
            "macdh": (macd - sig).to_numpy(), "macd": X["macd"], "obvz": X["obvz"]}


FIXED = {"rsi": (30, 20), "stoch": (20, 10), "cci": (-100, -200)}


def events(X, idx, real_side, ema_tr, regime):
    """X 為做多空間（找超賣）。real_side = 這個空間做多對應的原始方向。"""
    out = []
    S = series_for(X)
    n = len(X["c"])
    for name, v in S.items():
        pct = pd.Series(v).rolling(500, min_periods=300).rank(pct=True).to_numpy()
        levels = []
        if name in FIXED:
            levels += [(f"<= {FIXED[name][0]}", v <= FIXED[name][0]), (f"<= {FIXED[name][1]}", v <= FIXED[name][1])]
        levels += [("百分位<=10%", pct <= 0.10), ("百分位<=3%", pct <= 0.03)]
        for lab, ext in levels:
            ext = np.nan_to_num(ext, nan=0).astype(bool)
            hook = np.zeros(n, bool)
            hook[2:] = (v[1:-1] < v[:-2]) & (v[2:] > v[1:-1]) & ext[1:-1]
            first = np.zeros(n, bool)
            first[1:] = ext[1:] & ~ext[:-1]
            for kind, mask in (("拐頭", hook), ("直接", first)):
                for t in np.where(mask)[0]:
                    if t < 600:
                        continue
                    r = sim(X, t, 1)
                    if r is None:
                        continue
                    ei, xi, pnl = r
                    out.append(dict(ind=name, lvl=lab, kind=kind, side=real_side, entry_time=idx[ei], exit_time=idx[xi],
                                    pnl_R=pnl, trend=real_side * ema_tr[t], regime=real_side * regime[t]))
    return out


def main():
    for tf, df in get_frames().items():
        X = indicators(df)
        X["obvz"] = obv_z(X)
        idx = df.index
        ema = df.close.ewm(span=200, adjust=False).mean()
        ema_tr = np.sign(df.close - ema).shift(1).to_numpy()
        reg = daily_regime(df)
        M = mirror(X)
        M["obvz"] = -X["obvz"]
        E = pd.DataFrame(events(X, idx, 1, ema_tr, reg) + events(M, idx, -1, ema_tr, reg))
        print(f"\n{'=' * 100}\n## {tf}（反轉方向；研究期 2023~24 | 驗證期 2025~26/10 | 逐年）")
        for ind in ["rsi", "stoch", "cci", "macdh", "macd", "obvz"]:
            print(f"\n### {ind}")
            for lvl in E[E.ind == ind].lvl.unique():
                for kind in ("直接", "拐頭"):
                    split_line(f"{lvl} {kind}", E[(E.ind == ind) & (E.lvl == lvl) & (E.kind == kind)])
                x = E[(E.ind == ind) & (E.lvl == lvl) & (E.kind == "拐頭")]
                split_line(f"{lvl} 拐頭 順EMA200", x[x.trend == 1])
                split_line(f"{lvl} 拐頭 逆EMA200", x[x.trend == -1])
                split_line(f"{lvl} 拐頭 做多", x[x.side == 1])
                split_line(f"{lvl} 拐頭 做空", x[x.side == -1])
        print(f"\n### {tf} 彙總：所有指標 × 門檻，拐頭相對直接進場的改善（不重疊後平均 R，全期）")
        from part22_divergence import nonoverlap
        rows = []
        for (ind, lvl), g in E.groupby(["ind", "lvl"]):
            a = nonoverlap(g[g.kind == "直接"]).pnl_R.mean()
            b = nonoverlap(g[g.kind == "拐頭"]).pnl_R.mean()
            rows.append((ind, lvl, a, b, b - a))
        S = pd.DataFrame(rows, columns=["ind", "lvl", "直接", "拐頭", "改善"])
        print(S.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
