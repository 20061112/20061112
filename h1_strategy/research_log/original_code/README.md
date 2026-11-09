# 原始研究程式（只供追溯）

重現策略請用上一層的 `h1_six_divergence.py`。這裡是研究時實際用的程式，相依關係：

```
part32_final.py（定案輸出）
 ├── part30_exits.py（進場、出場）── part22_divergence.py（指標、偵測、nonoverlap）、part24_common.py（日線狀態）
 └── part28_h1_combo.py（淨值、報表）── part25_regime_drift.py ── part23_trend_hook.py
part31_risk.py（最大停損 / 風控測試）、part26_corr.py（相關性）
load.py（讀檔）、backtest.py（resample）
```

注意：`part22_divergence.get_frames()` 會同時讀取 `data/XAUUSD_M15_2023_2026.csv` 與
`data/XAUUSD_M1_2023_2026.csv`（70 MB，用來合成 M5；本策略用不到，所以沒有放進套件）。
要直接執行這些檔案，請把它們放在原 repo 的 `research/` 目錄（分支 `claude/dazzling-einstein-6dzhk0`）。
