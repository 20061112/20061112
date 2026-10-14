"""第二十五部分：實際波動（RV）預測，以及拿來過濾 / 調整部位。

RV：每個交易日（broker 01:00~23:59）M15 對數報酬平方和開根號（%）。另算隔夜跳空不計入。
預測「明天的 RV」，只用到今天收盤為止的資料：
  naive  = 今天 RV
  ma5 / ma22 = 過去 5 / 22 天 RV 平方平均開根號
  ewma   = RV² 的指數平均（λ = 0.94）
  HAR    = log RV(t+1) ~ log RV(1 天) + log RV(5 天) + log RV(22 天)，每月月初只用過去資料重估（滾動前推）
評估：log RV 的樣本外 R²、與實際的排名相關，逐年。
應用：
  1. 方向：預測 RV / 波動擴張比 跟隔天報酬有沒有關係
  2. LONBRK（2023~2026，第十部分的樣本外交易）、版本 A（2026/3~10）：依「當天開盤前」的預測 RV 分組看每筆 R
  3. 波動目標部位：每天的部位 ∝ 1 / 預測 RV，看日 Sharpe、回撤
"""
import numpy as np
import pandas as pd
from load import load
from part22_stalin_rank import tstat

pd.set_option("display.width", 220)


def daily_rv():
    d = load("data/XAUUSD_M15_2023_2026.csv")
    d["day"] = (d.index - pd.Timedelta(hours=1)).normalize()
    r = np.log(d.close).diff()
    r[d.day != d.day.shift()] = np.nan                       # 不算跨日跳空
    g = d.assign(r2=r ** 2).groupby("day")
    D = pd.DataFrame({"rv": np.sqrt(g.r2.sum()) * 100, "n": g.size(),
                      "open": g.open.first(), "close": g.close.last()})
    D = D[D.n >= 60]                                         # 去掉不完整的日子
    D["ret"] = np.log(D.close / D.close.shift()) * 100       # 收盤到收盤（含跳空），%
    return D


def forecasts(D):
    rv = D.rv
    F = pd.DataFrame(index=D.index)
    F["naive"] = rv
    F["ma5"] = np.sqrt((rv ** 2).rolling(5).mean())
    F["ma22"] = np.sqrt((rv ** 2).rolling(22).mean())
    F["ewma"] = np.sqrt((rv ** 2).ewm(alpha=0.06).mean())
    X = np.log(F[["naive", "ma5", "ma22"]])
    y = np.log(rv).shift(-1)                                 # 第 t 列 = 用 t 為止預測 t+1
    har = pd.Series(np.nan, index=D.index)
    months = D.index.to_period("M")
    y_month = pd.Series(months, index=D.index).shift(-1)    # 目標日所在的月份（避免用到本月第一天）
    for m in months.unique():
        past = (y_month < m).to_numpy() & X.notna().all(axis=1).to_numpy() & y.notna().to_numpy()
        if past.sum() < 120:
            continue
        A = np.c_[np.ones(past.sum()), X[past].to_numpy()]
        b = np.linalg.lstsq(A, y[past].to_numpy(), rcond=None)[0]
        cur = (months == m) & X.notna().all(axis=1).to_numpy()
        har[cur] = np.exp(np.c_[np.ones(cur.sum()), X[cur].to_numpy()] @ b)
    F["HAR"] = har
    F = F.shift(1)                                           # 第 t 列 = 第 t 天開盤前就知道的預測
    F["actual"] = rv
    F["expand"] = F["HAR"] / F["ma22"]                       # 波動擴張比：預測 / 近一個月
    return F


def eval_forecasts(F):
    G = F.dropna()
    la = np.log(G.actual)
    rows = []
    for c in ["naive", "ma5", "ma22", "ewma", "HAR"]:
        lf = np.log(G[c])
        r = {"model": c, "R2_log": 1 - ((la - lf) ** 2).sum() / ((la - la.mean()) ** 2).sum(),
             "rank_corr": lf.rank().corr(la.rank()), "MAE_log": (la - lf).abs().mean()}
        for y, s in G.groupby(G.index.year):
            if len(s) < 100:
                continue
            r[f"R2_{y}"] = 1 - ((np.log(s.actual) - np.log(s[c])) ** 2).sum() / ((np.log(s.actual) - np.log(s.actual).mean()) ** 2).sum()
        rows.append(r)
    return pd.DataFrame(rows), G


def by_regime(T, key, col, label, tcol="day"):
    """每筆依當天預測值分 3 組；門檻用「過去 250 個交易日」的 1/3、2/3 分位（滾動前推，
    因為黃金 RV 逐年上升，用全部歷史會讓 2026 全部落在「高」）"""
    hist = F[key].dropna()
    q1 = hist.rolling(250, min_periods=60).quantile(1 / 3).shift(1)
    q2 = hist.rolling(250, min_periods=60).quantile(2 / 3).shift(1)
    v, a, b = F[key].reindex(T[tcol]).values, q1.reindex(T[tcol]).values, q2.reindex(T[tcol]).values
    grp = np.where(np.isnan(a), "（暖機）", np.where(v < a, "低", np.where(v < b, "中", "高")))
    T = T.assign(grp=grp)
    out = T.groupby("grp")[col].agg(筆數="size", 平均R="mean", t=tstat, 勝率=lambda x: (x > 0).mean())
    yr = T.pivot_table(index="grp", columns=T[tcol].dt.year, values=col, aggfunc="mean")
    print(f"\n[{label}] 依 {key} 分組（每筆 R）")
    print(out.join(yr).round(3).to_string())


def vol_target(T, col, label, tcol="day"):
    """每天部位 = 預測中位數 / 當天預測 RV（上限 2 倍），比較固定部位"""
    daily = T.groupby(tcol)[col].sum()
    f = F["HAR"].reindex(daily.index)
    med = F["HAR"].rolling(250, min_periods=60).median().shift(1).reindex(daily.index)
    w = (med / f).clip(upper=2.0)
    ok = w.notna()
    rows = []
    for name, s in (("固定部位", daily[ok]), ("波動目標", daily[ok] * w[ok])):
        eq = s.cumsum()
        rows.append({"方式": name, "天數": len(s), "總R": s.sum(), "日均R": s.mean(), "日波動": s.std(),
                     "年化Sharpe": s.mean() / s.std() * np.sqrt(250), "最大回撤R": (eq - eq.cummax()).min(),
                     "總R/回撤": s.sum() / -(eq - eq.cummax()).min()})
    print(f"\n[{label}] 波動目標部位（以日為單位加總 R）")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


D = daily_rv()
F = forecasts(D)

if __name__ == "__main__":
    print(f"交易日 {len(D)}（{D.index[0].date()} ~ {D.index[-1].date()}）")
    print("\n每年日 RV（%）中位數：", D.rv.groupby(D.index.year).median().round(3).to_dict())
    print("RV 的自我相關（log）：lag1 %.2f、lag5 %.2f、lag22 %.2f" % tuple(np.log(D.rv).autocorr(k) for k in (1, 5, 22)))

    E, G = eval_forecasts(F)
    print("\n=== 1. 預測明天 RV（滾動前推，至少 120 天資料後開始，約 2023/7 起）===")
    print(E.round(3).to_string(index=False))

    print("\n=== 2. RV 能不能預測方向？（隔天收盤報酬，排名相關，逐年）===")
    H = F.join(D[["ret"]]).dropna()
    rows = []
    for c in ["HAR", "expand"]:
        rows.append({"feat": c, "IC_ret": H[c].rank().corr(H.ret.rank()), "IC_|ret|": H[c].rank().corr(H.ret.abs().rank()),
                     **{y: s[c].rank().corr(s.ret.rank()) for y, s in H.groupby(H.index.year)}})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # 3. LONBRK 樣本外交易
    L = pd.read_csv("oos_lonbrk_trades.csv", parse_dates=["day", "entry_time"])
    print("\n=== 3. LONBRK（2023~2026，參數凍結）===")
    for k in ["HAR", "expand"]:
        by_regime(L, k, "pnl_R", "LONBRK")
    vol_target(L, "pnl_R", "LONBRK")

    # 4. 版本 A（2026/3~10）
    A = pd.read_csv("final_A_trades.csv", parse_dates=["交易日"]).rename(columns={"交易日": "day", "損益_R": "R"})
    print("\n=== 4. 版本 A（2026/3~10，滾動前推門檻）===")
    for k in ["HAR", "expand"]:
        by_regime(A, k, "R", "版本 A")
    vol_target(A, "R", "版本 A")
    # 版本 A 的停損距離 vs 預測 RV：IB 寬度本身已經反映了當天波動多少？
    A["har"] = F["HAR"].reindex(A.day).values
    A["stop_pct"] = A["停損距離"] / A["進場價"] * 100
    print(f"\n版本 A 停損距離（%）與預測 RV 的排名相關：{A.stop_pct.rank().corr(A.har.rank()):.2f}")
