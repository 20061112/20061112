"""第二十六部分：H1 背離順勢單（D+E 盤面過濾、C1S2）——獲利 vs 超額，以及六個版本同時開的相關性。
輸出：python3 part26_corr.py > results_part26_corr.txt

1. 獲利 vs 超額：
   獲利 = 每筆實際損益（R，已扣點差）
   超額 = 獲利 − 持倉期間「同方向、同日線狀態、同年份」的平均漂移（每根 H1 的平均報酬 × 持有根數 ÷ 停損距離）
2. 六個版本（推進、OBV、速度衰減、RSI、CCI、Stoch）+ 無背離 base：
   - 交易重疊：A 的交易中，有多少筆和 B 是同一根進場、同方向（= 幾乎同一筆單）
   - 每日損益（以出場日加總 R）的相關係數
   - 六個同時開（每個版本每筆 1R）vs 同一筆訊號只開一次（去重）的組合績效
"""
import numpy as np
import pandas as pd
from part22_divergence import get_frames, nonoverlap, SPLIT
from part24_common import daily_regime, tstat
from part25_regime_drift import cont_trades

SIX = ["push", "obv", "speed", "rsi", "cci", "stoch"]
NAME = {"base": "無背離", "push": "推進", "obv": "OBV", "speed": "速度衰減", "rsi": "RSI", "cci": "CCI", "stoch": "Stoch", "macd": "MACD"}


def stats(daily):
    eq = daily.cumsum()
    dd = (eq.cummax() - eq).max()
    return f"總 {daily.sum():+6.1f}R  最大回撤 {dd:5.1f}R  總/回撤 {daily.sum() / dd:4.1f}  日Sharpe {daily.mean() / daily.std() * np.sqrt(252):4.2f}"


def main():
    df = get_frames()["H1"]
    A, _ = cont_trades(df)
    A = A[(A.side == A.reg) & (A.side == A.ema)]
    pos = pd.Series(np.arange(len(df)), index=df.index)
    c = df.close.to_numpy()
    atr = pd.Series(np.maximum(df.high - df.low, np.maximum((df.high - df.close.shift()).abs(),
                                                            (df.low - df.close.shift()).abs()))).rolling(14).mean().to_numpy()
    reg = daily_regime(df)
    r1 = np.r_[np.diff(c), np.nan] / atr                    # 每根 H1 往後一根的報酬（ATR）
    yr = df.index.year
    drift = pd.DataFrame({"r": r1, "reg": reg, "y": yr}).groupby(["reg", "y"]).r.mean()

    T = {}
    for cnd in ["base"] + SIX + ["macd"]:
        x = nonoverlap(A[A.cond == cnd]).copy()
        i0, i1 = pos.reindex(x.entry_time).to_numpy(), pos.reindex(x.exit_time).to_numpy()
        bars = i1 - i0 + 1
        d = np.array([drift.get((r, y), np.nan) for r, y in zip(x.reg, x.entry_time.dt.year)])
        x["drift_R"] = x.side * d * bars / x.R_atr
        x["excess_R"] = x.pnl_R - x.drift_R
        x["day"] = x.exit_time.dt.normalize()
        T[cnd] = x

    print("## 1. 獲利 vs 超額（H1，D+E，C1S2，不重疊）")
    print(f"  {'版本':8s} {'筆數':>5s} | {'獲利 研究':>16s} {'驗證':>16s} | {'漂移貢獻':>8s} | {'超額 研究':>16s} {'驗證':>16s} | 超額逐年")
    for k, x in T.items():
        a, b = x[x.entry_time < SPLIT], x[x.entry_time >= SPLIT]
        ey = " ".join(f"{v:+.2f}" for v in x.groupby(x.entry_time.dt.year).excess_R.mean())
        print(f"  {NAME[k]:8s} {len(x):5d} | {a.pnl_R.mean():+.3f} (t{tstat(a.pnl_R):+.1f})   {b.pnl_R.mean():+.3f} (t{tstat(b.pnl_R):+.1f})  |"
              f" {x.drift_R.mean():+.3f}    | {a.excess_R.mean():+.3f} (t{tstat(a.excess_R):+.1f})   {b.excess_R.mean():+.3f} (t{tstat(b.excess_R):+.1f})  | {ey}")
    for side, sn in ((1, "多"), (-1, "空")):
        x = pd.concat([T[k][T[k].side == side] for k in SIX])
        print(f"  六個合併 做{sn}：獲利 {x.pnl_R.mean():+.3f}R、漂移 {x.drift_R.mean():+.3f}R、超額 {x.excess_R.mean():+.3f}R（{len(x)} 筆，含重複）")

    keys = ["base"] + SIX
    print("\n## 2a. 交易重疊：列的交易中，有幾 % 和欄是同一筆（同進場時間、同方向）")
    sig = {k: set(zip(T[k].entry_time, T[k].side)) for k in keys}
    M = pd.DataFrame({NAME[b]: {NAME[a]: len(sig[a] & sig[b]) / len(sig[a]) for a in keys} for b in keys})
    print((M * 100).round(0).astype(int).to_string())

    print("\n## 2b. 每日損益（出場日加總 R）相關係數")
    days = pd.date_range(df.index.min().normalize(), df.index.max().normalize(), freq="D")
    D = pd.DataFrame({NAME[k]: T[k].groupby("day").pnl_R.sum().reindex(days, fill_value=0) for k in keys})
    D = D[D.index.dayofweek < 5]                           # 全部平日（含沒交易的日子），Sharpe 才不會高估
    C = D[(D != 0).any(axis=1)].corr()                      # 相關係數只用有交易的日子
    print(C.round(2).to_string())
    six = [NAME[k] for k in SIX]
    cc = C.loc[six, six].to_numpy()
    print(f"  六個版本兩兩相關係數平均 {cc[np.triu_indices(6, 1)].mean():.2f}（最低 {cc[np.triu_indices(6, 1)].min():.2f}、最高 {cc[np.triu_indices(6, 1)].max():.2f}）")
    print(f"  六個與無背離 base 的相關平均 {C.loc[six, '無背離'].mean():.2f}")

    print("\n## 2c. 同時開的組合")
    for k in keys:
        print(f"  {NAME[k]:8s} 單獨    : {stats(D[NAME[k]])}")
    print(f"  六個同時開（各 1R）: {stats(D[six].sum(axis=1))}")
    print(f"  六個同時開（各 1/6 R）: {stats(D[six].sum(axis=1) / 6)}")
    U = pd.concat([T[k] for k in SIX]).sort_values("entry_time").drop_duplicates(["entry_time", "side"])
    U = nonoverlap(U)
    du = U.groupby("day").pnl_R.sum().reindex(D.index, fill_value=0)
    print(f"  六個去重（同一訊號只開一次、一次一筆）: {stats(du)}  筆數 {len(U)}")
    every = pd.concat([T[k] for k in SIX])
    cnt = every.groupby(["entry_time", "side"]).size()
    print(f"  同一筆訊號被幾個版本同時觸發：1 個 {(cnt == 1).mean():.0%}、2~3 個 {((cnt >= 2) & (cnt <= 3)).mean():.0%}、4~6 個 {(cnt >= 4).mean():.0%}")


if __name__ == "__main__":
    main()
