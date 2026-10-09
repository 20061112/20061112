"""第三十九部分：OU 長週期偏離（其實是順勢）— 放寬出場、取消「最多一單」、與 ER 順勢策略（第 29 部分）比較 / 合併。

訊號：z = (p - μ*)/σ_eq（第 37 部分 zscore），|z| ≥ 3，方向往 μ*（= 長週期趨勢方向，見第 38 部分）。
設定
  S1 M15 ou L400            S2 M15 ou_ew L800
  S3 多週期 ou：M5 L1200 + M15 L400 + H1 L100
  S4 多週期 ou_ew：M5 L2400 + M15 L800 + H1 L200
和 ER 策略用同一套成本 / 規則：扣進場點差、隔夜 $0.7/盎司/晚（週三 → 週四 ×3）、週五最後一根 K 棒平倉、週五 20 點後不開新單。
R = 損益 / 初始停損距離（= 用固定 % 風險下單時的報酬；第 38 部分是固定 1.5 ATR，寬停損會被高估）。

部位模式
  one       同一設定同時最多 1 筆（第 37~38 部分的做法）
  unlimited 每次 |z| 穿越門檻都進場，不管手上有沒有單
  pyr8      只要 |z| ≥ 3，每 8 根就再進一筆（金字塔），同時最多 5 筆
  pyr8u     同上，不設上限
"""
import numpy as np
import pandas as pd
from load import load
from spring import resample, atr
from part37_wf import zscore, SPLIT

SWAP = 0.7
EXITS = {
    "基準 μ0 sl1.5 H40": dict(sl=1.5, tgt="mu0", hold=40),
    "時間 H80 sl1.5": dict(sl=1.5, hold=80),
    "時間 H160 sl3": dict(sl=3, hold=160),
    "時間 H320 sl5": dict(sl=5, hold=320),
    "移動4ATR(初始3) H320": dict(sl=3, hold=320, trail=4),
    "移動6ATR(初始4) H640": dict(sl=4, hold=640, trail=6),
    "|z|<2 出場 sl3 H320": dict(sl=3, hold=320, zexit=2.0),
    "|z|<1 出場 sl3 H320": dict(sl=3, hold=320, zexit=1.0),
    "跌破EMA(L/4) sl3 H320": dict(sl=3, hold=320, ma=True),
    "跌破EMA(L/4)+移動4ATR H640": dict(sl=3, hold=640, ma=True, trail=4),
}
MODES = {"one": dict(cross=True, cap=1), "unlimited": dict(cross=True, cap=None),
         "pyr8": dict(cross=False, cd=8, cap=5), "pyr8u": dict(cross=False, cd=8, cap=None)}


class Frame:
    def __init__(self, m1, tf):
        self.tf, self.mins = tf, {"5min": 5, "15min": 15, "1h": 60}[tf]
        df = resample(m1, tf)
        self.df, self.idx = df, df.index
        self.o, self.h, self.l, self.c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        self.A = atr(df).to_numpy()
        self.spr = df.spread.to_numpy(float) * 0.01
        nxt = np.r_[self.idx[1:].dayofweek, [0]]
        self.wk_last = (nxt < self.idx.dayofweek) | (np.r_[np.diff(self.idx.asi8), [0]] > 2 * 86400e9)
        self.wk_last[-1] = True
        self.day = self.idx.normalize().asi8
        self.dow = self.idx.dayofweek.to_numpy()
        self.hour = self.idx.hour.to_numpy()

    def nights(self, e, j):
        d = np.unique(self.day[e:j + 1])
        return int(sum(3 if pd.Timestamp(a).dayofweek == 2 else 1 for a in d[:-1]))

    def run(self, method, L, thr, ex, mode):
        z, mu = zscore(self.df, method, L)
        ma = pd.Series(self.c).ewm(span=max(L // 4, 5), adjust=False).mean().to_numpy()
        p, M = EXITS[ex], MODES[mode]
        za = np.abs(z)
        n = len(z)
        open_exit = []
        last_entry = {1: -10 ** 9, -1: -10 ** 9}
        rows = []
        for t in range(1, n - 1):
            if not za[t] >= thr or np.isnan(self.A[t]):
                continue
            if M["cross"] and not za[t - 1] < thr:
                continue
            d = -int(np.sign(z[t]))
            if not M["cross"] and t - last_entry[d] < M["cd"]:
                continue
            k = t + 1
            ts = self.idx[k]
            if self.dow[k] == 4 and self.hour[k] >= 20:
                continue
            open_exit = [x for x in open_exit if x >= k]
            if M["cap"] is not None and len(open_exit) >= M["cap"]:
                continue
            e, a = self.o[k], self.A[t]
            if p.get("tgt") == "mu0" and d * (mu[t] - e) <= 0:
                continue
            risk = p["sl"] * a
            stop, best = e - d * risk, e
            j_out, x, why = None, None, "time"
            for j in range(k, min(k + p["hold"], n)):
                if (self.l[j] <= stop) if d == 1 else (self.h[j] >= stop):
                    j_out, x, why = j, stop, "stop"
                    break
                cj = self.c[j]
                if p.get("tgt") == "mu0" and d * (cj - mu[t]) >= 0:
                    j_out, x, why = j, cj, "tp"
                    break
                if p.get("zexit") is not None and -d * z[j] < p["zexit"]:
                    j_out, x, why = j, cj, "z"
                    break
                if p.get("ma") and d * (cj - ma[j]) < 0:
                    j_out, x, why = j, cj, "ma"
                    break
                if self.wk_last[j]:
                    j_out, x, why = j, cj, "flat"
                    break
                best = max(best, self.h[j]) if d == 1 else min(best, self.l[j])
                if p.get("trail"):
                    stop = max(stop, best - p["trail"] * a) if d == 1 else min(stop, best + p["trail"] * a)
            if j_out is None:
                j_out = min(k + p["hold"], n) - 1
                x = self.c[j_out]
            open_exit.append(j_out)
            last_entry[d] = t
            nt = self.nights(k, j_out)
            usd = (x - e) * d - self.spr[k] - SWAP * nt
            rows.append(dict(src=f"{self.tf}|{method}|{L}", time=self.idx[t], entry=ts,
                             xtime=self.idx[j_out], side=d, why=why,
                             hours=(j_out - k + 1) * self.mins / 60, risk_usd=risk, usd=usd, R=usd / risk,
                             R15=usd / (1.5 * a)))
        return rows


SETS = {
    "S1 M15 ou L400": [("15min", "ou", 400)],
    "S2 M15 ou_ew L800": [("15min", "ou_ew", 800)],
    "S3 多週期 ou": [("5min", "ou", 1200), ("15min", "ou", 400), ("1h", "ou", 100)],
    "S4 多週期 ou_ew": [("5min", "ou_ew", 2400), ("15min", "ou_ew", 800), ("1h", "ou_ew", 200)],
}


def daily(T, days, col="R"):
    return pd.Series(T[col].to_numpy(), index=T.xtime.dt.date).groupby(level=0).sum().reindex(days, fill_value=0.0)


def metr(T, days, col="R"):
    if len(T) < 5:
        return dict(n=len(T))
    s = daily(T, days, col)
    first = [d for d in days if pd.Timestamp(d) < SPLIT]
    out = dict(n=len(T), per_day=len(T) / len(days), avg=T[col].mean(), win=(T[col] > 0).mean(),
               PF=T[col][T[col] > 0].sum() / -T[col][T[col] < 0].sum(), sumR=s.sum())
    for lab, dd in (("", days), ("前", first), ("後", [d for d in days if d not in set(first)])):
        x = s.reindex(dd)
        eq = x.cumsum().to_numpy()
        out[lab + "MDD"] = -(eq - np.maximum.accumulate(np.r_[0, eq])[1:]).min()
        out[lab + "Sharpe"] = x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else np.nan
    out["P/MDD"] = out["sumR"] / out["MDD"]
    return out


def fmt(m, lab):
    if m["n"] < 5:
        return f"  {lab:<44} n={m['n']}"
    return (f"  {lab:<44} n={m['n']:4d}（{m['per_day']:.2f}/天） avg={m['avg']:+.3f}R win={m['win']:.0%} PF={m['PF']:.2f} "
            f"總R={m['sumR']:+6.1f} MDD={m['MDD']:5.1f} P/MDD={m['P/MDD']:5.1f} Sharpe={m['Sharpe']:+.2f}"
            f"（前 {m['前Sharpe']:+.2f} / 後 {m['後Sharpe']:+.2f}）")


def main():
    m1 = load("data/XAUUSD_M1_2026.csv")
    days = sorted(set(m1.index.date))
    frames = {tf: Frame(m1, tf) for tf in ("5min", "15min", "1h")}
    lines = []
    P = lines.append

    def book(set_name, ex, mode):
        rows = []
        for tf, method, L in SETS[set_name]:
            rows += frames[tf].run(method, L, 3.0, ex, mode)
        return pd.DataFrame(rows).sort_values("entry").reset_index(drop=True)

    # ---------- 1. 出場 × 部位模式 ----------
    P("=" * 140)
    P("1. 放寬出場 × 部位模式（|z| ≥ 3；R = 損益 / 初始停損；含點差、隔夜、週五平倉）")
    grid = []
    books = {}
    for sn in SETS:
        P(f"\n--- {sn}")
        for ex in EXITS:
            for mode in MODES:
                T = book(sn, ex, mode)
                books[(sn, ex, mode)] = T
                m = metr(T, days)
                grid.append(dict(set=sn, exit=ex, mode=mode, **m))
                if mode in ("one", "unlimited", "pyr8"):
                    why = T.why.value_counts(normalize=True)
                    P(fmt(m, f"{ex} [{mode}]") + f"  持有中位 {T.hours.median():4.1f}h "
                      + " ".join(f"{k}{v:.0%}" for k, v in why.items()))
    G = pd.DataFrame(grid)
    G.to_csv("part39_grid.csv", index=False)
    P("\n  各出場在 4 組設定 × 4 種模式（16 組）裡的平均 Sharpe / 前半 / 後半 / 每天筆數")
    P(G.groupby("exit")[["Sharpe", "前Sharpe", "後Sharpe", "per_day", "avg", "P/MDD"]].mean().round(2)
      .sort_values("前Sharpe", ascending=False).to_string())
    P("\n  各部位模式（40 組平均）")
    P(G.groupby("mode")[["Sharpe", "前Sharpe", "後Sharpe", "per_day", "avg", "MDD", "P/MDD"]].mean().round(2).to_string())

    # ---------- 2. 只用前半段選版本 ----------
    pick = G[G.n >= 30].sort_values("前Sharpe", ascending=False).iloc[0]
    P("\n" + "=" * 140)
    P(f"2. 只用前半段 Sharpe 選出的版本：{pick["set"]} / {pick["exit"]} / {pick["mode"]}"
      f"（前半 {pick['前Sharpe']:+.2f} → 後半 {pick['後Sharpe']:+.2f}）")
    P("   前半段前 10 名與其後半段：")
    P(G[G.n >= 30].sort_values("前Sharpe", ascending=False).head(10)
      [["set", "exit", "mode", "n", "avg", "前Sharpe", "後Sharpe", "P/MDD"]].round(2).to_string(index=False))
    OU = books[(pick["set"], pick["exit"], pick["mode"])]
    OUb = books[("S3 多週期 ou", "基準 μ0 sl1.5 H40", "one")]

    # ---------- 3. 與 ER 比較 ----------
    ER = pd.read_csv("part29_final_trades.csv", parse_dates=["time", "xtime"])
    ER = ER.rename(columns={"R_1 單位": "R1"}).assign(R=lambda d: d.R1)
    P("\n" + "=" * 140)
    P("3. 與 ER 回檔順勢策略（第 29 部分，1 單位）比較 —— 都以每筆 1R 風險計")
    P(fmt(metr(ER, days), "ER 順勢（A+B+C）"))
    P(fmt(metr(OU, days), f"OU 前半選出版"))
    P(fmt(metr(OUb, days), "OU 多週期 基準出場 [one]"))
    for lab, O in (("前半選出版", OU), ("基準版", OUb)):
        de, do = daily(ER, days), daily(O, days)
        P(f"\n  [{lab}] 每日 R 相關係數 {de.corr(do):+.2f}；兩者同一天都有出場的天數 {int(((de != 0) & (do != 0)).sum())} / "
          f"ER {int((de != 0).sum())} 天、OU {int((do != 0).sum())} 天")
        same = opp = none = 0
        for r in O.itertuples():
            live = ER[(ER.time <= r.time) & (ER.xtime > r.time)]
            if live.empty:
                none += 1
            elif (live.side == r.side).any():
                same += 1
            else:
                opp += 1
        P(f"  OU 進場當下 ER 有同向單 {same}、只有反向單 {opp}、沒有單 {none}（共 {len(O)} 筆）")
        mo = pd.DataFrame({"ER": de, "OU": do})
        mo.index = pd.to_datetime(mo.index)
        P("  逐月 R：\n" + mo.groupby(mo.index.strftime("%m")).sum().round(1).T.to_string())
        for w, name in ((1.0, "1:1"), (0.5, "ER 1 : OU 0.5")):
            comb = pd.concat([ER[["xtime", "R"]], O[["xtime", "R"]].assign(R=O.R * w)])
            P(fmt(metr(comb, days), f"合併 {name}"))
    txt = "\n".join(lines)
    print(txt)
    open("results_part39.txt", "w").write(txt + "\n")
    OU.to_csv("part39_ou_trades.csv", index=False)


if __name__ == "__main__":
    main()
