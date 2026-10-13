"""第二十七部分：用 M1 速度 z 當「其他策略」的進場濾網（特別是空單）。

讀取本 repo 其他分支裡 2023~2026 多年度的交易明細（git show，不複製檔案），
在每筆進場前一根已收盤的 M1 上計算：
  z20  = z_v20（EWMA span 20 / ATR60，滾動 1380 根 z）——「最近幾分鐘」的速度
  z60  = 同法但 span 60 ——「最近半小時」的速度
對齊交易方向：zd = 方向 × z（正 = 價格正快速往單子方向走；做空時 zd > 0 = 正在急跌）。
看每個策略的多 / 空單，依 zd 分組的平均 R（樣本內 < 2025、樣本外 ≥ 2025），以及事先登記的濾網效果。
輸出：results_part27.txt
"""
import io
import subprocess
import numpy as np
import pandas as pd
from load import load

SPLIT = pd.Timestamp("2025-01-01")
SRC = {  # 名稱: (分支, 路徑, 進場欄, 方向欄, R 欄)
    "M15 前高前低突破（定案）": ("ccr-2a50af23-y02xev", "research/final_m15_trades.csv", "進場時間", "方向", "R"),
    "M15 前高前低突破（全部訊號，未濾）": ("ccr-2a50af23-y02xev", "research/part27_m15_trades_s3_t2.csv", "進場時間", "方向", "R"),
    "M15 突破 + 日線濾網": ("ccr-2a50af23-y02xev", "research/part28_m15_filtered_trades.csv", "進場時間", "方向", "R"),
    "IB 突破（part46）": ("ccr-2a50af23-y02xev", "research/part46_ib_trades.csv", "entry_time", "side", "R"),
    "IB 突破核心（part68，凍結）": ("claude/optimistic-meitner-rgzrz5", "research/part68_trades.csv", "t_in", "side", "R"),
    "多週期 VR（part49）": ("claude/optimistic-meitner-rgzrz5", "research/part49_trades.csv", "entry", "side", "R"),
    "ER 系統（part50）": ("ccr-754cdf47-jr7hab", "research/part50_trades_main.csv", "entry_time", "dir", "R"),
    "結構 HH/LL（part22b）": ("ccr-3c65414a-qutea9", "research/part22b_trades_k345_N480.csv", "entry", "dir", "R"),
}
out = []


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out.append(s)


m = load("data/XAUUSD_M1_2023_2026.csv")
c = m.close
pc = c.shift()
tr = pd.concat([m.high - m.low, (m.high - pc).abs(), (m.low - pc).abs()], axis=1).max(axis=1)
atr = tr.rolling(60).mean()
gap = m.index.to_series().diff() > pd.Timedelta(minutes=6)
dlt = c.diff().where(~gap)
Z = {}
for span in (20, 60):
    v = dlt.ewm(span=span, adjust=False).mean() / atr
    Z[span] = ((v - v.rolling(1380).mean()) / v.rolling(1380).std())


def side_of(s):
    s = s.astype(str).str.strip()
    return np.where(s.str.contains("空|short|sell", case=False) | (s == "-1") | (s == "-1.0"), -1,
                    np.where(s.str.contains("多|long|buy", case=False) | (s == "1") | (s == "1.0"), 1, 0))


def load_trades(branch, path, tcol, scol, rcol):
    raw = subprocess.run(["git", "show", f"origin/{branch}:{path}"], capture_output=True, check=True).stdout
    d = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig", low_memory=False)
    T = pd.DataFrame({"t": pd.to_datetime(d[tcol], errors="coerce"), "side": side_of(d[scol]), "R": pd.to_numeric(d[rcol], errors="coerce")})
    T = T.dropna()
    T = T[(T.side != 0) & (T.t >= m.index[2000])]
    # 進場前一根已收盤的 M1（進場分鐘往前一根），避免偷看
    key = T.t.dt.floor("min") - pd.Timedelta(minutes=1)
    for span in (20, 60):
        T[f"z{span}"] = Z[span].reindex(key, method="ffill").to_numpy() * T.side.to_numpy()
    return T.dropna()


BINS = [-np.inf, -2, -1, 1, 2, np.inf]
LAB = ["≤−2 逆向急", "−2~−1", "−1~+1 平", "+1~+2", "≥+2 順向急"]


def table(T, col):
    g = pd.cut(T[col], BINS, labels=LAB)
    r = T.groupby(g, observed=False).agg(n=("R", "size"), R=("R", "mean"))
    r["IS"] = T[T.t < SPLIT].groupby(g[T.t < SPLIT], observed=False).R.mean()
    r["OOS"] = T[T.t >= SPLIT].groupby(g[T.t >= SPLIT], observed=False).R.mean()
    return r


def fmt(T):
    i, o = T[T.t < SPLIT].R, T[T.t >= SPLIT].R
    return f"n={len(T):5d} 每筆 {T.R.mean():+.3f}R 總 {T.R.sum():+7.1f}R（IS {i.mean():+.3f} / OOS {o.mean():+.3f}）"


p("=" * 110)
p("第二十七部分：M1 速度 z 當其他策略的濾網（zd = 方向 × z；做空時 zd > 0 表示進場前正在急跌、zd < 0 表示正在急漲）")
p("=" * 110)
pool = []
for name, args in SRC.items():
    try:
        T = load_trades(*args)
    except Exception as e:                                                    # 分支不在 / 欄位不同
        p(f"[{name}] 讀取失敗：{e}")
        continue
    T["策略"] = name
    pool.append(T)
    p(f"\n##### {name}（{T.t.min():%Y-%m} ~ {T.t.max():%Y-%m}）")
    for sd, lab in ((-1, "空單"), (1, "多單")):
        S = T[T.side == sd]
        if len(S) < 50:
            continue
        p(f"--- {lab}：{fmt(S)}")
        for col, nm in (("z20", "z20（最近幾分鐘）"), ("z60", "z60（最近半小時）")):
            t = table(S, col)
            p(f"  {nm}： " + " | ".join(f"{k} n={int(r.n)} {r.R:+.2f}（{r.IS:+.2f}/{r.OOS:+.2f}）" for k, r in t.iterrows() if r.n > 0))

# ---------------------------------------------------------------- 全部策略合併
P = pd.concat(pool, ignore_index=True)
P = P[P.策略 != "M15 前高前低突破（全部訊號，未濾）"]                          # 未濾訊號量太大，另外看
p("\n" + "=" * 110)
p("合併（不含未濾訊號池）：每個策略等權（每筆 R 先減該策略、該方向的平均，看 z 帶來的『相對』差異）")
P["Rx"] = P.R - P.groupby(["策略", "side"]).R.transform("mean")
for sd, lab in ((-1, "空單"), (1, "多單")):
    S = P[P.side == sd]
    for col in ("z20", "z60"):
        g = pd.cut(S[col], BINS, labels=LAB)
        t = S.groupby(g, observed=False).agg(n=("Rx", "size"), 相對R=("Rx", "mean"))
        t["IS"] = S[S.t < SPLIT].groupby(g[S.t < SPLIT], observed=False).Rx.mean()
        t["OOS"] = S[S.t >= SPLIT].groupby(g[S.t >= SPLIT], observed=False).Rx.mean()
        p(f"--- {lab} {col}")
        p(t.round(3).to_string())

# ---------------------------------------------------------------- 事先登記的濾網
p("\n--- 濾網（事先登記，依第二十二~二十六部分的結論）：")
p("  F1 空單：進場前正在急跌（z20 對齊 ≥ +2）就不做空 —— 因為急跌後容易反彈")
p("  F2 空單：進場前正在急漲（z20 對齊 ≤ −2）就不做空 —— 因為急漲後容易延續")
p("  F3 多單：進場前正在急跌（z20 對齊 ≤ −2）照做（對照）")
for name in P.策略.unique():
    S = P[(P.策略 == name) & (P.side == -1)]
    if len(S) < 50:
        continue
    f1 = S[S.z20 < 2]
    f2 = S[S.z20 > -2]
    f12 = S[(S.z20 < 2) & (S.z20 > -2)]
    p(f"[{name}] 空單 原始 {fmt(S)}")
    p(f"{'':6s}F1 不追急跌 {fmt(f1)}")
    p(f"{'':6s}F2 不空急漲 {fmt(f2)}")
    p(f"{'':6s}F1+F2      {fmt(f12)}")
open("results_part27.txt", "w").write("\n".join(out) + "\n")
