"""第二十二部分：MACD / RSI / Stoch / CCI / OBV(tick volume) 背離 + 波動狀態條件
+ 波動調整後的推進背離 + 速度衰減背離，並檢查背離出現後是否有可交易的進場訊號。

資料：一開始就使用 2023~2026 全部資料（第十部分的教訓）。
  研究期 2023~2024 用來看條件、驗證期 2025~2026/10 只看結果。

結構定義（以頂背離說明；底背離把價格與指標取負號後套用同一段程式）：
  P1      = 已確認的擺動高點（左右各 K 根內最高；在 P1+K 收盤才知道）
  回檔    = P1 之後到突破前的最低點，需 >= PULL_ATR × ATR(P1)
  第二推  = 之後某根 t 的高點 > P1 高點，且 t 為 P1 以來最高（每根新高都檢查）
  t - P1 <= MAXGAP；第二推從第一根突破起最多看 MAXPUSH 根
  事件在 K 棒 t 收盤時判定，只用到 t 以前（含 t）的資料。

背離條件（在 t 與 P1 比較）
  macd / rsi / stoch / cci / obv : 指標在 t 低於 P1（價格更高、指標更低）
  push  波動調整推進背離 : 第二推幅度 / 起點 ATR < 第一推幅度 / 起點 ATR
         第一推 = P1 之前（上一個擺動高點與 P1 之間，最多 40 根）最低點 → P1
         第二推 = P1 與 t 之間最低點 → t
  speed 速度衰減背離     : 第二推平均速度（幅度 ATR 數 / 根數）< 第一推
  每個條件取「該 P1 的第二推中第一根符合條件的 K 棒」為事件。
  base 為對照組：第一根突破 P1 的 K 棒（不加任何背離條件）。

波動狀態（在 t 計算）
  vol_lvl  = ATR14 / 過去 VOL_WIN 根 ATR14 中位數（低 <0.8，中，高 >1.25）
  vol_chg  = ATR14(t) / ATR14(P1)（第二推期間波動擴張或收縮）
  bbw_pct  = 布林帶寬在過去 500 根的百分位

進場（訊號棒 t 之後 WAIT 根內；若先超過停損價則取消）
  T0 收盤直接進     : t+1 開盤
  T1 跌破訊號棒低點 : 停損單，價格 = 訊號棒低點
  T2 反轉 K         : 陰線且收盤 < 前一根低點 → 下一根開盤
  T3 Stoch 死叉     : %K 由上往下穿 %D 且收黑 → 下一根開盤
停損 = 進場前的最高點 + 0.1 ATR（且離進場價至少 0.5 ATR）；停利 2R；最長 HOLD 根；同根同時碰停損停利算停損；
進場那根若也碰到停損價算停損；每筆扣當下點差。
"""
import numpy as np
import pandas as pd
from load import load
from backtest import resample

K = 3
PULL_ATR = 1.0
MAXGAP = 60
MAXPUSH = 10
WAIT = 4
BUF = 0.1
MIN_R = 0.5
TP = 2.0
HOLD = 32
VOL_WIN = 1000
SPLIT = pd.Timestamp("2025-01-01")

DIVS = ["macd", "rsi", "stoch", "cci", "obv", "push", "speed"]
TRIG = ["T0", "T1", "T2", "T3"]


# ---------------------------------------------------------------- 指標
def indicators(df):
    o, h, l, c, v = (df[k].to_numpy(float) for k in ("open", "high", "low", "close", "tickvol"))
    s = pd.Series(c)
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    atr = pd.Series(tr).rolling(14).mean().to_numpy()
    macd = (s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()).to_numpy()
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = (100 - 100 / (1 + up / dn)).to_numpy()
    hh, ll = pd.Series(h).rolling(14).max(), pd.Series(l).rolling(14).min()
    fast = 100 * (s - ll) / (hh - ll)
    k_ = fast.rolling(3).mean()
    d_ = k_.rolling(3).mean()
    tp = pd.Series((h + l + c) / 3)
    ma = tp.rolling(20).mean()
    md = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = ((tp - ma) / (0.015 * md)).to_numpy()
    obv = np.cumsum(np.sign(np.r_[0, np.diff(c)]) * v)
    mid, sd = s.rolling(20).mean(), s.rolling(20).std()
    bbw = (4 * sd / mid)
    bbw_pct = bbw.rolling(500).rank(pct=True).to_numpy()
    vol_lvl = atr / pd.Series(atr).rolling(VOL_WIN, min_periods=200).median().to_numpy()
    return dict(o=o, h=h, l=l, c=c, atr=atr, macd=macd, rsi=rsi, stoch=k_.to_numpy(), stochd=d_.to_numpy(),
                cci=cci, obv=obv, bbw_pct=bbw_pct, vol_lvl=vol_lvl, spr=df.spread.to_numpy(float) * 0.01)


def mirror(X):
    """底背離 → 取負號後變成頂背離。"""
    Y = dict(X)
    Y["o"], Y["c"] = -X["o"], -X["c"]
    Y["h"], Y["l"] = -X["l"], -X["h"]
    Y["macd"], Y["cci"], Y["obv"] = -X["macd"], -X["cci"], -X["obv"]
    Y["rsi"], Y["stoch"], Y["stochd"] = 100 - X["rsi"], 100 - X["stoch"], 100 - X["stochd"]
    return Y


# ---------------------------------------------------------------- 事件偵測（頂背離方向）
def detect(X):
    h, l, atr = X["h"], X["l"], X["atr"]
    n = len(h)
    win = 2 * K + 1
    rm = pd.Series(h).rolling(win, center=True).max().to_numpy()
    piv = np.where((h == rm) & np.isfinite(atr))[0]
    events = []
    prev = None
    for P1 in piv:
        P0 = prev
        prev = P1
        if P1 < 250:
            continue
        a1 = max(P1 - 40, P0 if P0 is not None else 0)
        s1 = a1 + int(np.argmin(l[a1:P1 + 1]))
        A1 = (h[P1] - l[s1]) / atr[s1]
        V1 = A1 / max(P1 - s1, 1)
        # 第一根突破
        tb = None
        for t in range(P1 + K + 1, min(P1 + MAXGAP + 1, n)):
            if h[t] > h[P1]:
                tb = t
                break
        if tb is None:
            continue
        pull = h[P1] - l[P1:tb].min()
        if pull < PULL_ATR * atr[P1]:
            continue
        found = {}
        top = h[P1]
        for t in range(tb, min(tb + MAXPUSH, P1 + MAXGAP + 1, n - 1)):
            if h[t] <= top:
                continue
            top = h[t]
            s2 = P1 + int(np.argmin(l[P1:t + 1]))
            A2 = (h[t] - l[s2]) / atr[s2]
            V2 = A2 / max(t - s2, 1)
            cond = dict(base=t == tb,
                        macd=X["macd"][t] < X["macd"][P1], rsi=X["rsi"][t] < X["rsi"][P1],
                        stoch=X["stoch"][t] < X["stoch"][P1], cci=X["cci"][t] < X["cci"][P1],
                        obv=X["obv"][t] < X["obv"][P1], push=A2 < A1, speed=V2 < V1)
            for name, ok in cond.items():
                if ok and name not in found:
                    found[name] = t
            if len(found) == len(cond):
                break
        for name, t in found.items():
            s2 = P1 + int(np.argmin(l[P1:t + 1]))
            A2 = (h[t] - l[s2]) / atr[s2]
            ndiv = sum(int(v) for v in (X["macd"][t] < X["macd"][P1], X["rsi"][t] < X["rsi"][P1],
                                        X["stoch"][t] < X["stoch"][P1], X["cci"][t] < X["cci"][P1],
                                        X["obv"][t] < X["obv"][P1]))
            events.append(dict(cond=name, t=t, P1=P1, gap=t - P1, A1=A1, A2=A2, push_ratio=A2 / A1,
                               V1=V1, V2=A2 / max(t - s2, 1), ndiv=ndiv,
                               vol_lvl=X["vol_lvl"][t], vol_chg=atr[t] / atr[P1], bbw_pct=X["bbw_pct"][t],
                               rsi=X["rsi"][t]))
    return events


# ---------------------------------------------------------------- 進場與出場（在轉換後空間一律做空）
def simulate(X, t, trig):
    o, h, l, c, atr, spr = X["o"], X["h"], X["l"], X["c"], X["atr"], X["spr"]
    n = len(c)
    A = atr[t]
    ext = h[t]
    ei = ep = None
    if trig == "T0":
        if t + 1 >= n:
            return None
        ei, ep = t + 1, o[t + 1]
    else:
        for j in range(t + 1, min(t + 1 + WAIT, n - 1)):
            stop_lvl = ext + BUF * A
            if trig == "T1":
                if l[j] < l[t]:
                    ei, ep = j, min(l[t], o[j])
                    ext = max(ext, h[j]) if h[j] >= stop_lvl else ext
                    break
                if h[j] >= stop_lvl:
                    return None
            else:
                if h[j] >= stop_lvl:
                    return None
                ext = max(ext, h[j])
                if trig == "T2":
                    ok = c[j] < o[j] and c[j] < l[j - 1]
                else:
                    ok = (X["stoch"][j] < X["stochd"][j] and X["stoch"][j - 1] >= X["stochd"][j - 1] and c[j] < o[j])
                if ok:
                    ei, ep = j + 1, o[j + 1]
                    break
        if ei is None:
            return None
    stop = max(ext + BUF * A, ep + MIN_R * A)      # 停損距離至少 MIN_R ATR，避免 R 太小被點差吃掉
    R = stop - ep
    if R <= 0:
        return None
    tgt = ep - TP * R
    xi, xp, why = None, None, "time"
    for j in range(ei, min(ei + HOLD + 1, n)):
        if h[j] >= stop:
            xi, xp, why = j, (stop if j == ei else max(stop, o[j])), "stop"
            break
        if j > ei and l[j] <= tgt:
            xi, xp, why = j, tgt, "tp"
            break
    if xi is None:
        xi = min(ei + HOLD, n - 1)
        xp = c[xi]
    pnl = (ep - xp) - spr[ei]
    return dict(ei=ei, xi=xi, R_atr=R / A, exit=why, pnl_R=pnl / R)


def fwd(X, t, hz):
    c, atr, spr, n = X["c"], X["atr"], X["spr"], len(X["c"])
    if t + hz >= n:
        return np.nan
    return (c[t] - c[t + hz] - spr[t]) / atr[t]


def build(df):
    X = indicators(df)
    idx = df.index
    rows, trades = [], []
    for side, Y in ((-1, X), (1, mirror(X))):          # side = 實際方向：-1 做空（頂背離）、+1 做多（底背離）
        for e in detect(Y):
            e.update(side=side, time=idx[e["t"]], f4=fwd(Y, e["t"], 4), f8=fwd(Y, e["t"], 8), f16=fwd(Y, e["t"], 16))
            for tg in TRIG:
                r = simulate(Y, e["t"], tg)
                if r is not None:
                    trades.append(dict(cond=e["cond"], trig=tg, t=e["t"], side=side, setup=idx[e["t"]],
                                       entry_time=idx[r["ei"]], exit_time=idx[r["xi"]], P1=e["P1"],
                                       R_atr=r["R_atr"], exit=r["exit"], pnl_R=r["pnl_R"],
                                       ndiv=e["ndiv"], vol_lvl=e["vol_lvl"], vol_chg=e["vol_chg"],
                                       bbw_pct=e["bbw_pct"], push_ratio=e["push_ratio"]))
            rows.append(e)
    return pd.DataFrame(rows), pd.DataFrame(trades)


def nonoverlap(T):
    T = T.sort_values("entry_time")
    keep, busy = [], pd.Timestamp.min
    for i, r in zip(T.index, T.itertuples()):
        if r.entry_time > busy:
            keep.append(i)
            busy = r.exit_time
    return T.loc[keep]


def get_frames():
    m15 = load("data/XAUUSD_M15_2023_2026.csv")
    m1 = load("data/XAUUSD_M1_2023_2026.csv")
    h1 = resample(m15, "1h")
    m5 = resample(m1, "5min")
    return {"M5": m5, "M15": m15, "H1": h1}
