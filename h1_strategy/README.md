# XAUUSD H1「六合一」背離順勢策略 — 重現套件

這個資料夾是完整、獨立的重現套件：一個 M15 資料檔 + 一支 Python 程式，就能重跑整個回測，並逐筆比對參考結果。
不需要 repo 裡的其他研究程式。

| 文件 | 內容 |
|---|---|
| [`STRATEGY.md`](STRATEGY.md) | **策略規則（完整定義）、績效、美元換算、風險、實盤流程** |
| [`RESEARCH_SUMMARY.md`](RESEARCH_SUMMARY.md) | 為什麼是這個策略：研究過程、測過但沒採用的東西 |
| `README.md`（本檔） | 怎麼在新環境重現 |

---

## 1. 檔案清單
```
h1_strategy/
├── README.md                 本檔：重現步驟
├── STRATEGY.md               策略規則與績效
├── RESEARCH_SUMMARY.md       研究過程摘要
├── requirements.txt          Python 套件
├── h1_six_divergence.py      回測主程式（單檔，含全部規則）
├── verify.py                 和參考結果逐筆比對
├── data/
│   └── XAUUSD_M15_2023_2026.csv    MT5 匯出的 M15（2023/1/3 ~ 2026/10/9），5.5 MB
├── reference/                參考結果（verify.py 用）
│   ├── trades_reference.csv  逐筆 962 筆
│   ├── stats_reference.txt   統計
│   ├── equity_reference.png  淨值曲線
│   └── research_version/     研究程式（research/part32_final.py）的原始輸出，見第 6 節
└── research_log/             研究紀錄與原始研究程式（只供追溯，重現不需要）
    ├── RESEARCH_README.md    研究全紀錄（第一 ~ 三十二部分）
    ├── FINAL_STRATEGY_H1_research.md
    └── original_code/        研究用的原始程式
```

## 2. 環境需求
- Python 3.10 以上（測過 3.13）
- `pandas >= 2.0`、`numpy >= 1.24`；`matplotlib` 可選（只用來畫圖）
- 已測試：pandas 3.0.5 + numpy 2.5.3、pandas 2.2.3 + numpy 2.1.3，兩者逐筆結果完全相同

```bash
pip install -r requirements.txt
```

## 3. 重現步驟
```bash
cd h1_strategy
python3 verify.py                  # 1. 檢查資料檔、重跑回測、逐筆比對 → 應印出 PASS
python3 h1_six_divergence.py       # 2. 產生 output/trades.csv、output/stats.txt、output/equity.png
```
預期結果（`output/stats.txt`）：
```
交易 962 筆（多 760 / 空 202），不同訊號 493 個
勝率 52.9%；每筆 +0.449R；PF 2.14
研究期（< 2025）每筆 +0.547R；驗證期 +0.376R
組合總損益 +71.94R；最大回撤 平倉 4.52R / 含浮動 5.50R
逐年（組合 R）：2023: +13.8  2024: +23.5  2025: +26.3  2026: +8.3
每版本固定 0.01 手：總 $10,181；最大回撤 平倉 $1,267 / 含浮動 $2,476
```
全部約 3 秒。

## 4. 用自己的資料
```bash
python3 h1_six_divergence.py --data 你的_M15.csv --out output_new
```
- 格式：MT5「匯出柱狀圖」的 M15，Tab 分隔，欄位 `<DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>`，
  日期 `2023.01.03`、時間 `01:00:00`，SPREAD 單位為點（0.01 美元）。
- 時間用 broker 時間（本資料為紐約 + 7）。日線是用 broker 日期切的，換時區不同的券商結果會稍有差異。
- 程式會自己把 M15 合成 H1；開頭約 250 根 H1 與 70 個交易日是暖機（擺動點、SMA50 斜率），不會有交易。
- **最重要的下一步**：匯出 2022 年以前（最好包含 2011~2015 黃金空頭）的 M15，參數不改直接跑，確認策略在空頭市場也成立。

## 5. 實盤用：列出目前有效的掛單
```bash
python3 h1_six_divergence.py --data 最新的_M15.csv --signals
```
在最後一根 H1 收完後執行，會列出「訊號棒在最後 4 根內、還沒觸發、也還沒取消」的掛單，以及目前的盤面過濾是否成立。
（回測的盤面過濾是用「進場前一根」判斷；實盤掛單時用最新一根判斷，成交前請再確認一次。）
停損與停利請在成交後依 STRATEGY.md 的規則設定（停利 = 進場價 + 3 × 停損距離）。

## 6. 和研究版本的差異
研究程式 `research/part32_final.py` 的結果是 960 筆、+70.39R；本套件是 962 筆、+71.94R。差別只有一個：
- 同一個版本在同一根 K 棒有兩個訊號時（同一根 K 棒同時突破兩個擺動點，2023~2026 共 170 次），只能進一筆。
  研究程式交給 pandas 的預設排序決定（不穩定排序，結果取決於套件版本）；
  本套件固定為「**取最近的擺動點**」（停損放在最近的回檔低點），所以在任何環境結果都相同。
- 兩者有 21 筆單的停損不同、1 筆進場時間不同，整體結論不變。研究版的原始輸出放在 `reference/research_version/`。

## 7. 程式結構（`h1_six_divergence.py`）
| 函式 | 作用 |
|---|---|
| `load_m15`、`to_h1` | 讀 MT5 資料、合成 H1 |
| `indicators` | ATR14、MACD、RSI14、Stoch %K(14,3)、CCI20、OBV |
| `mirror` | 價格與指標上下翻轉，讓做空用同一套「向上突破」程式 |
| `daily_regime` | 日線狀態（SMA50 + 20 天斜率，前一天） |
| `detect` | 擺動點、突破、六種背離 → 訊號棒 |
| `backtest` | 盤面過濾、突破訊號棒進場、停損、3R / 32 根出場、每版本一次一筆 |
| `exit_long` | 出場模擬（同根同時碰到停損停利算停損、進場那根不給停利、跳空用開盤價） |
| `report`、`mtm_curve` | 統計、含浮動回撤、美元換算 |
| `pending_signals` | `--signals` 用的目前掛單 |
