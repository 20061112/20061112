"""檢查新環境的回測結果是否和參考結果一致。

用法：python3 verify.py
  1. 檢查資料檔 SHA-256 是否為原始檔
  2. 重新跑 h1_six_divergence.backtest
  3. 和 reference/trades_reference.csv 逐筆比對（版本、方向、進場時間、出場時間、損益 R）
全部一致 → 印出 PASS。
"""
import hashlib
import os
import sys
import pandas as pd
import h1_six_divergence as S

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "XAUUSD_M15_2023_2026.csv")
SHA = "cc76476b32b8bdd974305e3ce8d3d112dafcb52d836c52e2b11d16ecee2b5629"
TOL = 1e-3


def main():
    ok = True
    h = hashlib.sha256(open(DATA, "rb").read()).hexdigest()
    print(f"[1] 資料 SHA-256 {'一致' if h == SHA else '不一致（資料檔不同，結果不會相同）'}")
    ok &= h == SHA

    h1 = S.to_h1(S.load_m15(DATA))
    T, _ = S.backtest(h1)
    T["cond"] = T.cond.map(S.NAME)
    T["side"] = T.side.map({1: "多", -1: "空"})
    R = pd.read_csv(os.path.join(HERE, "reference", "trades_reference.csv"), parse_dates=["entry_time", "exit_time"])
    print(f"[2] 筆數：新 {len(T)} / 參考 {len(R)}；組合 R：新 {T.combo_R.sum():+.3f} / 參考 {R.combo_R.sum():+.3f}")
    ok &= len(T) == len(R) and abs(T.combo_R.sum() - R.combo_R.sum()) < 0.01

    key = ["cond", "side", "entry_time"]
    m = T[key + ["exit_time", "pnl_R"]].merge(R[key + ["exit_time", "pnl_R"]], on=key, how="outer",
                                               suffixes=("_new", "_ref"), indicator=True)
    miss = m[m._merge != "both"]
    diff = m[(m._merge == "both") & (((m.pnl_R_new - m.pnl_R_ref).abs() > TOL) | (m.exit_time_new != m.exit_time_ref))]
    print(f"[3] 逐筆比對：只在一邊出現 {len(miss)} 筆；進場相同但出場 / 損益不同 {len(diff)} 筆")
    if len(miss) or len(diff):
        print(pd.concat([miss, diff]).head(20).to_string())
    ok &= len(miss) == 0 and len(diff) == 0
    print("\nPASS：結果與參考完全一致" if ok else "\nFAIL：結果不一致，請檢查資料檔與套件版本")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
