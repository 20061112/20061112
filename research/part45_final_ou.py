"""第四十五部分：OU 長週期順勢集成 — 最終版本的詳細交易統計。

規則
  訊號：OU（最近 L 根等權 AR(1)）z = (p − μ*)/σ_eq，|z| 由 < 3 穿越到 ≥ 3 的那根收盤
  16 組：M5 / M10 / M15 / M20 × 回看 70 / 100 / 130 / 200 小時（L = 小時 × 60 / 分鐘），每組獨立、各自最多 1 筆
  進場：下一根開盤市價，方向 = 往 μ*（= 長週期趨勢方向）；週五 20 點（broker）後不開新單
  停損：進場價 ∓ 1.5 × ATR14（訊號 K 棒）；不移動
  出場：碰停損、持有滿 40 根收盤、或週五最後一根 K 棒收盤（不過週末）
  倉位：每組基本風險 = 1/16 單位；訊號當下 ATR14 / ATR200 < 1 → ×2
  成本：進場那根的點差、隔夜 $0.7/盎司/晚（週三 → 週四 ×3）
R：每筆以「該筆初始停損距離」為 1R；組合報酬單位 = 全倉 1R（16 組 × 1/16）。
"""
import numpy as np
import pandas as pd
from load import load
from part39_ou_vs_er import SWAP, SPLIT
from part43_twoside_vol_tf import TFrame, signals, HOURS, SL, HOLD

TFS = {"5min": 5, "10min": 10, "15min": 15, "20min": 20}
NCFG = 16


def trades(F, ts, ds, cfg):
    rows = []
    busy = -1
    n = len(F.c)
    for t, d in zip(ts, ds):
        k = t + 1
        if k <= busy:
            continue
        a, e = F.A[t], F.o[k]
        vr = a / F.A200[t]
        risk = SL * a
        stop = e - d * risk
        j_out, x, why = None, None, "時間"
        mfe = mae = 0.0
        for j in range(k, min(k + HOLD, n)):
            if (F.l[j] <= stop) if d == 1 else (F.h[j] >= stop):
                j_out, x, why = j, stop, "停損"
                mae = risk
                break
            mfe = max(mfe, (F.h[j] - e) if d == 1 else (e - F.l[j]))
            mae = max(mae, (e - F.l[j]) if d == 1 else (F.h[j] - e))
            if F.wk_last[j]:
                j_out, x, why = j, F.c[j], "週五平倉"
                break
        if j_out is None:
            j_out = min(k + HOLD, n) - 1
            x = F.c[j_out]
        busy = j_out
        nt = F.nights(k, j_out)
        gross = (x - e) * d
        rows.append(dict(cfg=cfg, tf=F.tf, signal=F.idx[t], entry=F.idx[k], exit=F.idx[j_out] + pd.Timedelta(minutes=F.mins),
                         side="多" if d == 1 else "空", entry_px=e, exit_px=x, stop_px=stop, atr=a, vr=vr,
                         w=2.0 if vr < 1 else 1.0, risk_usd=risk, gross_usd=gross, spread_usd=F.spr[k],
                         swap_usd=SWAP * nt, nights=nt, usd=gross - F.spr[k] - SWAP * nt, why=why,
                         hours=(j_out - k + 1) * F.mins / 60, mfe_R=mfe / risk, mae_R=min(mae / risk, 1.0)))
    return rows


def streak(v):
    best = cur = 0
    for x in v:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return best


def main():
    m1 = load("data/XAUUSD_M1_2026.csv")
    days = sorted(set(m1.index.date))
    nd = len(days)
    rows = []
    for tf, mins in TFS.items():
        F = TFrame(m1, tf, mins)
        for h in HOURS:
            rows += trades(F, *signals(F, int(h * 60 / mins)), f"{tf}|{h}h")
    T = pd.DataFrame(rows).sort_values("entry").reset_index(drop=True)
    T["R"] = T.usd / T.risk_usd                       # 單筆 R（1 倍）
    T["pR"] = T.R * T.w / NCFG                        # 對組合的貢獻（全倉 R）
    T["xday"] = (T.exit - pd.Timedelta(minutes=1)).dt.date
    T.to_csv("part45_final_trades.csv", index=False)

    out = []
    P = out.append
    d = pd.Series(T.pR.to_numpy(), index=T.xday).groupby(level=0).sum().reindex(days, fill_value=0.0)
    eq = d.cumsum()
    dd = eq - np.maximum.accumulate(np.r_[0, eq.to_numpy()])[1:]
    mdd = -dd.min()
    first = [x for x in days if pd.Timestamp(x) < SPLIT]
    sh = lambda s: s.mean() / s.std() * np.sqrt(252)
    win = T[T.R > 0]
    loss = T[T.R <= 0]

    P("=" * 100)
    P("OU 長週期順勢集成（16 組，波動比 < 1 ×2）— 2026/1/2 ~ 10/8，192 個交易日")
    P("=" * 100)
    P("\n【總覽】（組合報酬單位 = 全倉 1R；單筆 R = 該筆停損距離）")
    P(f"  交易筆數        {len(T)} 筆（每天 {len(T) / nd:.2f} 筆；×2 的 {int((T.w == 2).sum())} 筆 = {np.mean(T.w == 2):.0%}）")
    ev, lt, ls = 0, None, None
    for r in T.itertuples():
        if lt is None or r.side != ls or (r.entry - lt) > pd.Timedelta(hours=2):
            ev += 1
        lt, ls = r.entry, r.side
    P(f"  獨立事件        {ev} 次（每天 {ev / nd:.2f} 次；平均每次 {len(T) / ev:.1f} 組一起進場）")
    P(f"  有交易的日子    {T.xday.nunique()} / {nd} 天")
    P(f"  勝率            {len(win) / len(T):.1%}")
    P(f"  平均賺 / 平均賠  +{win.R.mean():.2f}R / {loss.R.mean():.2f}R（賺賠比 {win.R.mean() / -loss.R.mean():.2f}）")
    P(f"  每筆期望        {T.R.mean():+.3f}R（1 倍）；加權後 {(T.R * T.w).mean():+.3f}R")
    P(f"  PF              {T.pR[T.pR > 0].sum() / -T.pR[T.pR < 0].sum():.2f}")
    P(f"  總報酬          {d.sum():+.1f}R（全倉）")
    P(f"  最大回撤        {mdd:.1f}R（{eq.index[int(np.argmin(dd.to_numpy()))]}）；獲利/回撤 {d.sum() / mdd:.1f}")
    P(f"  Sharpe（日，年化） {sh(d):.2f}（前半 {sh(d.loc[first]):.2f} / 後半 {sh(d.drop(first)):.2f}）")
    P(f"  正報酬日        {np.mean(d[d != 0] > 0):.0%}（有交易的日子）；最好一天 {d.max():+.2f}R、最差一天 {d.min():+.2f}R")
    P(f"  最長連虧 / 連勝  {streak(T.R <= 0)} / {streak(T.R > 0)} 筆（16 組合併依時間）")
    tl = pd.concat([pd.Series(T.w.to_numpy() / NCFG, index=T.entry), pd.Series(-T.w.to_numpy() / NCFG, index=T.exit)]).sort_index().cumsum()
    P(f"  同時持倉風險    最大 {tl.max():.2f}R、有持倉時中位數 {tl[tl > 1e-9].median():.2f}R")

    P("\n【出場與持有】")
    for k, s in T.groupby("why"):
        P(f"  {k:<6} {len(s):4d} 筆（{len(s) / len(T):.0%}） 平均 {s.R.mean():+.2f}R  平均持有 {s.hours.mean():.1f} 小時")
    P(f"  持有時間 中位數 {T.hours.median():.1f} 小時（M5 {T[T.tf == '5min'].hours.median():.1f}h、M20 {T[T.tf == '20min'].hours.median():.1f}h）；"
      f"過夜 {np.mean(T.nights > 0):.0%}")
    P(f"  MFE（最大浮盈）中位數：贏單 {win.mfe_R.median():.2f}R、輸單 {loss.mfe_R.median():.2f}R；輸單中曾經浮盈 ≥ 1R 的 {np.mean(loss.mfe_R >= 1):.0%}")
    P(f"  MAE（最大浮虧）中位數：贏單 {win.mae_R.median():.2f}R")
    bins = [-np.inf, -0.999, 0, 1, 3, 5, np.inf]
    lab = ["−1R（停損）", "−1~0", "0~1", "1~3", "3~5", ">5R"]
    c = pd.cut(T.R, bins, labels=lab).value_counts(normalize=True).reindex(lab)
    P("  R 分佈：" + "｜".join(f"{k} {v:.0%}" for k, v in c.items()) + f"；最大單筆 {T.R.max():+.1f}R")

    P("\n【成本】（每筆平均，美元/盎司）")
    P(f"  停損距離 中位數 ${T.risk_usd.median():.1f}（範圍 {T.risk_usd.min():.1f} ~ {T.risk_usd.max():.1f}）")
    P(f"  點差 ${T.spread_usd.mean():.2f}（= {np.mean(T.spread_usd / T.risk_usd):.1%} R）；隔夜 ${T.swap_usd.mean():.2f}；"
      f"成本合計占毛利 {(T.spread_usd.sum() + T.swap_usd.sum()) / T.gross_usd[T.gross_usd > 0].sum():.1%}")
    gr = (T.gross_usd / T.risk_usd * T.w / NCFG).sum()
    P(f"  毛報酬 {gr:+.1f}R → 扣成本後 {T.pR.sum():+.1f}R")

    P("\n【獲利集中度】")
    top = d.sort_values(ascending=False)
    P(f"  最好 5 天 {top.head(5).sum():+.1f}R（占總和 {top.head(5).sum() / d.sum():.0%}）；最好 10 天 {top.head(10).sum() / d.sum():.0%}")
    P(f"  拿掉最好 5 天剩 {d.sum() - top.head(5).sum():+.1f}R；拿掉最好 10 天剩 {d.sum() - top.head(10).sum():+.1f}R")
    P("  最好 5 天：" + "、".join(f"{k} {v:+.2f}R" for k, v in top.head(5).items()))

    def grp(col, order=None):
        g = T.groupby(col)
        res = g.agg(筆數=("R", "size"), 勝率=("R", lambda x: np.mean(x > 0)), 每筆R=("R", "mean"), 貢獻=("pR", "sum"))
        if order is not None:
            res = res.reindex(order)
        res["占比"] = res.貢獻 / T.pR.sum()
        return res.round(3)

    P("\n【依時框】")
    P(grp("tf", list(TFS)).to_string())
    T["回看"] = T.cfg.str.split("|").str[1]
    P("\n【依回看】")
    P(grp("回看", [f"{h}h" for h in HOURS]).to_string())
    P("\n【依方向】")
    P(grp("side").to_string())
    T["波動"] = np.where(T.vr < 1, "vr<1（×2）", np.where(T.vr < 1.32, "1~1.32", "≥1.32"))
    P("\n【依波動比】")
    P(grp("波動", ["vr<1（×2）", "1~1.32", "≥1.32"]).to_string())
    T["時段"] = pd.cut(T.entry.dt.hour, [-1, 7, 14, 19, 24], labels=["亞洲 0-7", "歐洲 8-14", "美盤 15-19", "美盤後 20-23"])
    P("\n【依進場時段（broker 時間）】")
    P(grp("時段", ["亞洲 0-7", "歐洲 8-14", "美盤 15-19", "美盤後 20-23"]).to_string())
    T["星期"] = T.entry.dt.dayofweek.map(dict(enumerate("一二三四五六日")))
    P("\n【依星期】")
    P(grp("星期", list("一二三四五")).to_string())

    P("\n【逐月】（全倉 R）")
    T["月"] = T.exit.dt.strftime("%Y-%m")
    mo = T.groupby("月").agg(筆數=("R", "size"), 勝率=("R", lambda x: np.mean(x > 0)), R=("pR", "sum"))
    mo["月內最大回撤"] = [(-(lambda e: (e - np.maximum.accumulate(np.r_[0, e])[1:]).min())(s.cumsum().to_numpy()))
                     for _, s in d.groupby(pd.to_datetime(pd.Series(d.index)).dt.strftime("%Y-%m").to_numpy())]
    P(mo.round(2).to_string())
    wk = d.groupby(pd.to_datetime(pd.Series(d.index)).dt.to_period("W").to_numpy()).sum()
    P(f"\n  週：正 {np.mean(wk[wk != 0] > 0):.0%}（有交易的週）；最好週 {wk.max():+.2f}R、最差週 {wk.min():+.2f}R")

    P("\n【資金模擬】起始 10,000 美元，每單位風險 = 權益 × r%（每組 1/16 單位，×2 時 2/16），不限最小手數")
    for r in (1.0, 2.0, 3.0, 5.0):
        e, peak, mdd_p = 10000.0, 10000.0, 0.0
        for v in d.to_numpy():
            e *= 1 + v * r / 100
            peak = max(peak, e)
            mdd_p = min(mdd_p, e / peak - 1)
        one = 10000 * r / 100 / NCFG / T.risk_usd.median() / 100
        P(f"  r = {r:.0f}%：期末 {e:,.0f}（{e / 10000 - 1:+.0%}）最大回撤 {mdd_p:.1%}；"
          f"一組 1 倍的手數約 {one:.3f} 手（中位停損 ${T.risk_usd.median():.1f}/盎司）")
    P("  注意：r 是「全倉 1R」的風險；單筆只用 1/16（×2 時 1/8），所以 r = 2% 時單筆風險 0.125%~0.25%。")
    P("  10,000 美元時單筆手數會低於 0.01 手，實際操作需要更大資金或把幾組合併成一筆下單。")

    txt = "\n".join(out)
    print(txt)
    open("results_part45.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
