"""用期貨（Nasdaq-100 / MNQ）取代 TQQQ 達成槓桿。

結構：QQQ 一直持有（定投 80% 照買、20% 存現金），訊號開啟時加上期貨，讓總曝險 = 3 倍淨值；訊號關閉時期貨平倉。
QQQ 永遠不賣（除了恐慌補倉之外，跟原策略一樣），換倉只動期貨。
期貨每日損益（每 1 元名目）= QQQ 價格報酬 + 股息率 / 252 − 無風險利率 / 252（買期貨 + 放 T-bill = 持有股票的總報酬）。
股息率 0.7%/年（QQQ 價格資料不含股息；期貨價格的持有成本會把股息扣回來）；轉倉與交易成本：每季轉倉 0.02% 名目、每次調整 0.02%。
現金：每日計息（正：T-bill；負：T-bill + 1%，代表券商融資或賣 QQQ 的成本），期貨損益每天結算進現金。
重設頻率：每天（等同沒有管理費的 TQQQ）、每週、每月、不重設（進場時定口數，到出場前只在定投時加碼）。
另外用整數 MNQ 口數模擬不同帳戶大小（每週定投金額），看口數顆粒度的影響。
輸出 results_futures.txt、results_futures.csv。
"""
import numpy as np
import pandas as pd
import engine as E
from run_tests import q, OK0, w_rule

out = []
P = lambda s: (print(s, flush=True), out.append(s))
DIV = 0.007
ROLL = 0.0002
TRADE = 0.0002
NEG_SPREAD = 0.01
MAINT = 0.07           # 維持保證金約為期貨名目的 7%
NDX_PER_QQQ = 41.0     # NDX 指數約為 QQQ 價格的 41 倍（近似）


def run_fut(w, reset="W", start=None, end=None, weekly_dollars=None, lev=3.0, cover=True):
    idx = q.index
    i0 = 0 if start is None else idx.searchsorted(pd.Timestamp(start))
    i1 = len(idx) if end is None else idx.searchsorted(pd.Timestamp(end))
    px = q.close.values; rf = q.rf.values; ws = q.week_start.values
    z = E.atr_z(q).shift(1).values; bz = E.boll_z(q).shift(1).values
    mon = q.index.to_period("M"); ms = np.r_[True, mon[1:] != mon[:-1]]
    qtr = q.index.to_period("Q"); qs = np.r_[True, qtr[1:] != qtr[:-1]]
    nq = 0.0; cash = 0.0; fs = 0.0; inv = 0.0; flows = []
    vals = np.zeros(i1 - i0); invs = np.zeros(i1 - i0); cash_frac = np.zeros(i1 - i0); levs = np.full(i1 - i0, np.nan)
    sw = 0; first = True; short_days = 0; worst_day = 0.0; on_prev = False; sold = 0.0
    p_start = px[i0]
    for k, i in enumerate(range(i0, i1)):
        # 1. 現金計息、期貨結算
        cash *= 1 + (rf[i] if cash >= 0 else rf[i] + NEG_SPREAD / 252)
        nav_prev = nq * px[i - 1] + cash if i > i0 else 0.0
        if fs != 0 and i > i0:
            pnl = fs * px[i - 1] * ((px[i] / px[i - 1] - 1) + DIV / 252 - rf[i])
            cash += pnl
            if qs[i]:
                cash -= abs(fs) * px[i] * ROLL
        # 2. 定投
        if ws[i] or first:
            first = False
            amt = 2.0 if (np.isfinite(bz[i]) and bz[i] <= -0.6) else 1.0
            inv += amt; flows.append((idx[i], -amt))
            nq += 0.8 * amt / px[i]; cash += 0.2 * amt
            if np.isfinite(z[i]) and z[i] > 2 and cash > 0:
                nq += cash / px[i]; cash = 0.0
        nav = nq * px[i] + cash
        # 3. 期貨目標
        on = w[i] > 0
        if on:
            target = max((lev * nav - nq * px[i]) / px[i], 0.0)
            do = (not on_prev) or reset == "D" or (reset == "W" and ws[i]) or (reset == "M" and ms[i]) \
                or (reset == "N" and ws[i] and False)
            if reset == "N" and on_prev and ws[i]:
                target = fs + 0.8 * (2.0 if (np.isfinite(bz[i]) and bz[i] <= -0.6) else 1.0) * (lev - 1) / px[i]   # 定投那筆也配 2 倍期貨
                do = True
        else:
            target, do = 0.0, fs != 0
        if do:
            if weekly_dollars:
                contract = 2 * NDX_PER_QQQ * px[-1] / weekly_dollars * px[i] / p_start   # 今天的 MNQ 名目，從窗口起點開始定投
                target = np.round(target * px[i] / contract) * contract / px[i]
            if abs(target - fs) > 1e-12:
                cash -= abs(target - fs) * px[i] * TRADE
                if (fs == 0) != (target == 0):
                    sw += 1
                fs = target
        on_prev = on
        need = MAINT * abs(fs) * px[i] if fs != 0 else 0.0
        if cover and cash < need and nq > 0:
            sell = min((need - cash) / (1 - TRADE), nq * px[i])
            nq -= sell / px[i]; cash += sell * (1 - TRADE); sold += sell
        nav = nq * px[i] + cash
        if i > i0 and nav_prev > 0:
            worst_day = min(worst_day, nav / nav_prev - 1)
        if fs != 0:
            levs[k] = (nq + fs) * px[i] / nav if nav > 0 else np.nan
            if cash < MAINT * fs * px[i]:
                short_days += 1
        vals[k] = nav; invs[k] = inv; cash_frac[k] = cash / nav if nav > 0 else np.nan
    flows.append((idx[i1 - 1], vals[-1]))
    return dict(value=vals, invested=invs, flows=flows, switches=sw, years=(idx[i1 - 1] - idx[i0]).days / 365.25,
                sold_per_year=sold / max(np.mean(vals), 1e-9) / max((idx[i1 - 1] - idx[i0]).days / 365.25, 1e-9),
                cash_min=np.nanmin(cash_frac), short_frac=short_days / max(np.isfinite(levs).sum(), 1),
                lev_med=np.nanmedian(levs), lev_p5=np.nanpercentile(levs, 5), lev_p95=np.nanpercentile(levs, 95), worst_day=worst_day)


w0 = w_rule(OK0)
base = E.stats(E.run(q, w0))
broll = E.rolling_windows(q, w0)
rows = []


def show(label, runner, extra=True):
    res = runner(None, None); f = E.stats(res)
    s10 = E.stats(runner("2010-02-12", None))
    roll = pd.DataFrame([E.stats(runner(f"{y}-01-01", f"{y + 10}-01-01")) for y in range(2000, 2017)])
    d = roll.irr.values - broll.irr.values
    row = dict(label=label, irr99=f["irr"], irr10=s10["irr"], d10=np.median(d), win10=int((d > 0).sum()), worst=f["worst"],
               worst10=roll.worst.min())
    msg = (f"{label:36s} IRR {f['irr']:.1%}｜2010 起 {s10['irr']:.1%}｜10 年 IRR 差中位 {np.median(d):+.1%}（{(d > 0).sum()}/17）｜"
           f"最慘 {f['worst']:.0%}")
    if extra and "cash_min" in res:
        row |= {k: res[k] for k in ("cash_min", "short_frac", "lev_med", "lev_p5", "lev_p95", "worst_day", "sold_per_year")}
        msg += (f"｜持期貨時實際槓桿 中位 {res['lev_med']:.2f}（5% {res['lev_p5']:.2f} ~ 95% {res['lev_p95']:.2f}）"
                f"｜現金最低 {res['cash_min']:.0%} 淨值｜賣 QQQ 補保證金 {res['sold_per_year']:.0%} 平均淨值/年｜最差單日 {res['worst_day']:.1%}")
    rows.append(row); P(msg)


P("原策略用 TQQQ；期貨版：QQQ 不賣、訊號開啟時加期貨到 3 倍淨值")
show("原策略（TQQQ）", lambda a, b: E.run(q, w0, a, b), extra=False)
show("期貨 3 倍，每週重設，QQQ 永遠不賣（虧損變負債）", lambda a, b: run_fut(w0, "W", a, b, cover=False))
P("以下：現金低於維持保證金（期貨名目 7%）時賣 QQQ 補足")
for reset, nm in [("D", "每天重設"), ("W", "每週重設"), ("M", "每月重設"), ("N", "不重設（進場定口數）")]:
    show(f"期貨 3 倍，{nm}", lambda a, b, r=reset: run_fut(w0, r, a, b))
for lv in (2.5,):
    show(f"期貨 {lv} 倍，每週重設（原報告）", lambda a, b, l=lv: run_fut(w0, "W", a, b, lev=l))

P("\n合約顆粒度：整數口 MNQ，用今天的合約大小（約 6 萬美元名目）從零開始定投，合約名目隨窗口內 QQQ 價格縮放；每週重設")
for dollars in [200, 500, 1000, 3000]:
    show(f"  每週定投 {dollars} 美元", lambda a, b, dl=dollars: run_fut(w0, "W", a, b, weekly_dollars=dl))
pd.DataFrame(rows).to_csv("results_futures.csv", index=False, float_format="%.4f")
px_now = q.close.iloc[-1]
P(f"\n目前（{q.index[-1].date()}）QQQ {px_now:.0f} → NDX ≈ {px_now * NDX_PER_QQQ:,.0f} → 一口 MNQ 名目 ≈ {2 * NDX_PER_QQQ * px_now:,.0f} 美元；"
  f"要加 2 倍淨值的曝險，每 {NDX_PER_QQQ * px_now:,.0f} 美元淨值需要 1 口")
open("results_futures.txt", "w").write("\n".join(out) + "\n")
