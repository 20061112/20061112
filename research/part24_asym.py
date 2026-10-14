"""第二十四部分 (2)：為什麼多空不一致？空頭趨勢中也一樣嗎？
輸出：python3 part24_asym.py > results_part24_asym.txt

沒有 2023 以前的資料（repo 各分支都沒有，外部資料來源被網路政策擋住），
所以用 2023~2026 內部的「日線空頭」期間來檢驗：
  日線狀態 = part24_common.daily_regime（SMA50 與其 20 天斜率，只用前一天資料）。

1. 各年、各狀態的天數，以及「任意 K 棒」往後 16 根的平均漂移（ATR）。
2. 第二十三部分的順勢單（C1S2）與統一出場版本，依「方向 × 日線狀態」拆開：
   重點是「空頭狀態中的順勢空」是否和「多頭狀態中的順勢多」一樣好。
3. 扣掉同狀態、同方向的平均漂移後的超額前瞻報酬。
4. 機制檢查：大漲 / 大跌之後的走勢（大跌是否較快被買回）。
"""
import numpy as np
import pandas as pd
from part22_divergence import get_frames, indicators, mirror, detect, nonoverlap, SPLIT, DIVS
from part23_trend_hook import trend_trades
from part24_common import sim, fwd, daily_regime, fmt, tstat, both_spaces, split_line

REG = {1: "日線多頭", -1: "日線空頭", 0: "盤整"}


def main():
    F = get_frames()
    for tf in ("M15", "H1"):
        df = F[tf]
        X = indicators(df)
        idx = df.index
        reg = daily_regime(df)
        c, atr = X["c"], X["atr"]
        print(f"\n{'=' * 100}\n## {tf}")

        # 1. 狀態天數與漂移
        d = pd.DataFrame({"reg": reg, "y": idx.year}, index=idx)
        days = d.groupby([d.index.normalize()]).first()
        print("\n### 1. 日線狀態天數（逐年）")
        print(days.groupby(["y", "reg"]).size().unstack(fill_value=0).rename(columns=REG).to_string())
        f16 = np.full(len(c), np.nan)
        f16[:-16] = (c[16:] - c[:-16]) / atr[:-16]
        print("\n  任意 K 棒往後 16 根的平均報酬（做多方向，ATR）")
        for r in (1, 0, -1):
            m = reg == r
            print(f"  {REG[r]:5s}: 全期 {np.nanmean(f16[m]):+.3f}  研究期 {np.nanmean(f16[m & (idx < SPLIT)]):+.3f}  驗證期 {np.nanmean(f16[m & (idx >= SPLIT)]):+.3f}  （K 棒數 {m.sum()}）")
        drift = {r: np.nanmean(f16[reg == r]) for r in (1, 0, -1)}

        # 2. 順勢單依方向 × 狀態
        A, U, Fw = [], [], []
        for Y, s_short in both_spaces(X):
            ev = detect(Y)
            for tr in trend_trades(Y, ev, idx, s_short):   # trend_trades 內部會存成 -side = 順勢方向
                A.append(tr)
            for e in ev:
                t = e["t"]
                real = -s_short                         # 順勢 = 在 Y 中做多
                r = sim(Y, t, 1)
                if r is not None:
                    U.append(dict(cond=e["cond"], side=real, entry_time=idx[r[0]], exit_time=idx[r[1]], pnl_R=r[2], reg=reg[t]))
                ff = fwd(Y, t, 16, 1)
                Fw.append(dict(cond=e["cond"], side=real, reg=reg[t], f16=ff, time=idx[t],
                               excess=ff - real * drift.get(reg[t], np.nan) if np.isfinite(reg[t]) else np.nan))
        A = pd.DataFrame(A)
        A["reg"] = pd.Series(reg, index=idx).reindex(A.entry_time).to_numpy()
        U, Fw = pd.DataFrame(U), pd.DataFrame(Fw)

        for title, T in (("C1S2（第二十三部分的順勢單）", A[A.strat == "C1S2"]), ("統一出場（下一根開盤，1.5 ATR 停損 / 3 ATR 停利）", U)):
            print(f"\n### 2. {title}：方向 × 日線狀態   研究期 | 驗證期 | 逐年")
            for cnd in ["base", "push", "speed", "cci", "obv"]:
                x = T[T.cond == cnd]
                for side, sn in ((1, "多"), (-1, "空")):
                    for r in (1, 0, -1):
                        split_line(f"{cnd} 做{sn} {REG[r]}", x[(x.side == side) & (x.reg == r)], 18)

        print("\n### 3. 前瞻 16 根（順勢方向，ATR，可重疊）與扣掉同狀態漂移後的超額")
        for cnd in ["base", "push", "speed", "cci", "obv"]:
            for side, sn in ((1, "多"), (-1, "空")):
                for r in (1, -1):
                    x = Fw[(Fw.cond == cnd) & (Fw.side == side) & (Fw.reg == r)]
                    print(f"  {cnd:5s} 做{sn} {REG[r]}: n={len(x):5d} f16 {x.f16.mean():+.3f} (t{tstat(x.f16):+.1f})  超額 {x.excess.mean():+.3f} (t{tstat(x.excess):+.1f})")

        # 4. 機制：大漲 / 大跌之後
        print("\n### 4. 機制：過去 16 根漲跌 >= 3 ATR 之後，再往後 16 根（做多方向，ATR）")
        p16 = np.full(len(c), np.nan)
        p16[16:] = (c[16:] - c[:-16]) / atr[16:]
        for r in (1, 0, -1):
            for lab, m in (("大漲後", p16 >= 3), ("大跌後", p16 <= -3)):
                mm = m & (reg == r)
                print(f"  {REG[r]:5s} {lab}: n={mm.sum():6d}  後 16 根 {np.nanmean(f16[mm]):+.3f}  （同狀態平均 {drift[r]:+.3f}）")


if __name__ == "__main__":
    main()
