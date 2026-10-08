"""第十部分：TPO / IB / z-score 雛型研究（XAUUSD M1，2026/1/2 ~ 10/8，broker 時間）。

1. 每日標準 TPO 剖面（30 分字母、tick=1、POC、70% VA）與 IB。
2. 三種 IB：亞洲（01:00，交易日開頭的標準 IB）、倫敦（10:00）、紐約（15:30 = 08:30 ET）。
3. z-score 化：
   a. IB z     ：log(IB 區間) 對過去 20 天同時段 IB 的 z-score。
   b. 剖面 z   ：把前一天的 TPO 分佈當成分佈，價格位置 z = (價 − μ_prev) / σ_prev。
                 （VA 寬度 ≈ 2.2σ，所以 VAH/VAL ≈ ±1.1σ）
   c. 字母樹 z ：每一列 TPO 數在當天剖面內的 z-score（> +1 = HVN 厚區，< −0.5 = LVN 薄區）。
   d. IB 單位  ：(價 − IB 中點) / (IB 區間 / 2)，即 IB 延伸倍數。
4. 檢驗：IB 延伸、首次突破 IB 的延續/失敗、回到前日 POC 的機率、80% 法則、薄區/厚區的後續速度。
所有特徵只用到 IB 結束前（或當下）的資料。結果寫到 results_part10.txt。
"""
import sys
import numpy as np
import pandas as pd
from tpo import load_m1, build_days, tpo_profile, poc_va, profile_stats, print_profile, TICK

OUT = open("results_part10.txt", "w")


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s); OUT.write(s + "\n")


def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return (x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 and x.std() > 0 else np.nan


def summ(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return f"n={len(x):3d} mean={x.mean():+.3f} t={tstat(x):+.2f}" if len(x) else "n=0"


d = load_m1()
D, PROF = build_days(d)
DAYS = list(D.index)
G = {k: g for k, g in d.groupby("day") if k in D.index}
SPLIT = pd.Timestamp("2026-06-01")

P("=" * 100)
P("資料：", d.index.min(), "~", d.index.max(), " 交易日數", len(D), "（4/3~4/14 缺資料）")
P("VA 寬度 / σ_TPO：中位數 %.2f（常態 70%% = 2.07）；POC 與 μ 的距離 / σ：中位數 %.2f" %
  (((D.vah - D.val) / D.sd).median(), ((D.poc - D.mu) / D.sd).abs().median()))
P("日波幅中位數 %.1f，ATR10 中位數 %.1f" % (D.range.median(), D.atr10.median()))

# ---------------- 範例：最後一個完整交易日的字母圖 ----------------
day = DAYS[-2]
pr, c, rows, cz = PROF[day]; r = D.loc[day]
P("\n範例 TPO 字母圖", day.date(), " POC %.1f VAH %.1f VAL %.1f IB %.1f~%.1f（| = IB 範圍）" % (r.poc, r.vah, r.val, r.ibl, r.ibh))
P(print_profile(pr, c, rows, r.poc, r.vah, r.val, r.ibh, r.ibl, step=2))

# ---------------- 每日 × 每個 IB 時段的事件表 ----------------
SESS = {"ASIA": 0, "LDN": 18, "NY": 29}   # IB 起始 period（30 分為單位，0 = 01:00）


def ib_rows(name, sp):
    rows = []
    for i, day in enumerate(DAYS):
        if i == 0:
            continue
        g = G[day]; prev = D.iloc[i - 1]; prev_day = DAYS[i - 1]
        ib = g[(g.period >= sp) & (g.period < sp + 2)]
        after = g[g.period >= sp + 2]
        before = g[g.period < sp]
        if len(ib) < 50 or len(after) < 60:
            continue
        ibh, ibl = ib.high.max(), ib.low.min(); rng = ibh - ibl
        if rng <= 0:
            continue
        mid = (ibh + ibl) / 2; px = ib.close.iloc[-1]; atr = D.atr10.iloc[i]
        # 前日剖面的字母樹 z（當下價格所在那一列）
        ppr, pc, _, pcz = PROF[prev_day]
        k = int(np.floor((px - ppr[0]) / TICK))
        tree_z = pcz[k] if 0 <= k < len(pcz) else np.nan       # 在前日區間外 = NaN
        # 當天到 IB 前的發展中剖面（倫敦/紐約才有）
        dev_z = dev_poc = np.nan
        if len(before) > 60:
            bp, bc, _ = tpo_profile(before)
            dmu, dsd, _, _ = profile_stats(bp, bc)
            dev_poc = poc_va(bp, bc)[0] + TICK / 2
            dev_z = (px - dmu) / dsd if dsd > 0 else np.nan
        # IB 位置相對前日價值區
        if ibl > prev.vah: loc = "above_VA"
        elif ibh < prev.val: loc = "below_VA"
        elif ibl >= prev.val and ibh <= prev.vah: loc = "inside_VA"
        else: loc = "overlap"
        # 結果：IB 結束後到當天收盤
        ah, al, close = after.high.max(), after.low.min(), after.close.iloc[-1]
        up_t = after.index[after.high > ibh]; dn_t = after.index[after.low < ibl]
        first = None
        if len(up_t) and (not len(dn_t) or up_t[0] < dn_t[0]): first = 1
        elif len(dn_t): first = -1
        # 首次突破後：突破價進場，收盤出場；IB 另一側停損
        brk_ret = brk_R = np.nan
        if first is not None:
            t0 = up_t[0] if first == 1 else dn_t[0]
            lvl = ibh if first == 1 else ibl; stop = ibl if first == 1 else ibh
            rest = after[after.index >= t0]
            hit = (rest.low <= stop) if first == 1 else (rest.high >= stop)
            exit_px = stop if hit.any() else close
            brk_ret = first * (exit_px - lvl) / atr
            brk_R = first * (exit_px - lvl) / rng
        # 前日 POC 是否在 IB 後被碰到（IB 內未碰到者才算）
        ppoc = prev.poc
        poc_in_ib = ibl <= ppoc <= ibh
        poc_touch = float(al <= ppoc <= ah) if not poc_in_ib else np.nan
        rows.append(dict(day=day, sess=name, ibh=ibh, ibl=ibl, rng=rng, atr=atr, ib_atr=rng / atr,
                         z_prev=(px - prev.mu) / prev.sd, z_ibmid_prev=(mid - prev.mu) / prev.sd,
                         tree_z=tree_z, dev_z=dev_z, vloc=loc, px=px,
                         ext_up=max(0, ah - ibh) / rng, ext_dn=max(0, ibl - al) / rng,
                         first=first, brk_ret=brk_ret, brk_R=brk_R,
                         ret_close=(close - px) / atr, close_pos=(close - mid) / (rng / 2),
                         ppoc_dir=np.sign(ppoc - px), poc_dist=(ppoc - px) / atr, poc_touch=poc_touch,
                         rest_range=(ah - al) / atr,
                         final_poc_pos=(D.poc.iloc[i] - mid) / (rng / 2)))
    E = pd.DataFrame(rows).set_index("day")
    lr = np.log(E.rng)
    E["ib_z"] = (lr - lr.rolling(20).mean().shift(1)) / lr.rolling(20).std().shift(1)
    E["half"] = np.where(E.index < SPLIT, "H1", "H2")
    return E


ALL = {s: ib_rows(s, sp) for s, sp in SESS.items()}


def bucket_table(E, col, bins, labels, outs):
    E = E.dropna(subset=[col])
    b = pd.cut(E[col], bins, labels=labels)
    P(f"  {col} 分組：")
    for lab in labels:
        x = E[b == lab]
        if not len(x):
            continue
        parts = []
        for o in outs:
            v = x[o].dropna()
            parts.append(f"{o}={v.mean():+.2f}" if len(v) else f"{o}=nan")
        P(f"    {lab:>14s} n={len(x):3d}  " + "  ".join(parts))


for s, E in ALL.items():
    P("\n" + "=" * 100)
    P(f"[{s}] IB = broker {1 + SESS[s] // 2:02d}:{30 * (SESS[s] % 2):02d} 起 60 分鐘；事件 {len(E)} 天")
    P("  IB 區間中位數 %.1f（%.2f ATR）；IB 後有延伸：任一邊 %.0f%%、兩邊都延伸 %.0f%%" %
      (E.rng.median(), E.ib_atr.median(), 100 * ((E.ext_up > 0) | (E.ext_dn > 0)).mean(),
       100 * ((E.ext_up > 0) & (E.ext_dn > 0)).mean()))
    E["ext_max"] = E[["ext_up", "ext_dn"]].max(axis=1)
    E["ext_sum"] = E.ext_up + E.ext_dn
    E["poc_in_ib"] = ((E.final_poc_pos.abs() <= 1)).astype(float)
    P("  當天最終 POC 落在 IB 內的比例 %.0f%%" % (100 * E.poc_in_ib.mean()))
    P("  延伸倍數（IB 單位）分位數 ext_max：", E.ext_max.quantile([.25, .5, .75, .9]).round(2).to_dict())

    P("\n  (a) IB z-score → 後續延伸（IB 單位）與剩餘波幅（ATR）")
    bucket_table(E, "ib_z", [-9, -1, -0.3, 0.3, 1, 9], ["z<-1", "-1~-0.3", "-0.3~0.3", "0.3~1", "z>1"],
                 ["ext_max", "ext_sum", "rest_range", "poc_in_ib"])
    cc = E[["ib_z", "ext_sum", "rest_range"]].dropna().corr(method="spearman")
    P("  Spearman: ib_z vs ext_sum %.2f, ib_z vs rest_range(ATR) %.2f" % (cc.loc["ib_z", "ext_sum"], cc.loc["ib_z", "rest_range"]))

    P("\n  (b) 首次突破 IB → 突破價進場、收盤出場、IB 另一側停損（brk_ret: ATR，brk_R: IB 區間）")
    P("    全部：", summ(E.brk_ret), "| R:", summ(E.brk_R))
    for h in ["H1", "H2"]:
        P(f"    {h}：", summ(E[E.half == h].brk_ret))
    bucket_table(E, "ib_z", [-9, -0.5, 0.5, 9], ["小IB z<-0.5", "中", "大IB z>0.5"], ["brk_ret", "brk_R"])
    # 突破方向與前日價值區的關係
    E["brk_toward_prev"] = np.where(E["first"].isna(), np.nan, np.where(E["first"] == np.sign(-E.z_prev), 1.0, 0.0))
    for tw, lab in [(1.0, "突破方向朝前日 μ（回價值）"), (0.0, "突破方向遠離前日 μ（離開價值）")]:
        x = E[E.brk_toward_prev == tw]
        P(f"    {lab:24s}", summ(x.brk_ret))

    P("\n  (c) IB 相對前日價值區（loc）→ 收盤位置（IB 單位）、IB 後報酬（ATR）、首次突破方向")
    for loc in ["above_VA", "overlap", "inside_VA", "below_VA"]:
        x = E[E.vloc == loc]
        if len(x) < 3:
            continue
        P(f"    {loc:10s} n={len(x):3d} close_pos={x.close_pos.mean():+.2f} ret_close={x.ret_close.mean():+.2f}"
          f" (t={tstat(x.ret_close):+.2f}) 首破向上={(x['first'] == 1).mean() * 100:.0f}%"
          f" 碰前日POC={x.poc_touch.mean() * 100:.0f}%")

    P("\n  (d) 剖面 z（IB 結束時價格相對前日 TPO 分佈）→ 回到前日 POC 機率、到收盤報酬（ATR，正 = 往前日 μ 方向）")
    E["ret_to_mu"] = -np.sign(E.z_prev) * E.ret_close
    E["az"] = E.z_prev.abs()
    bucket_table(E, "az", [0, 0.5, 1, 1.5, 2.5, 99], ["|z|<0.5", "0.5~1", "1~1.5", "1.5~2.5", ">2.5"],
                 ["poc_touch", "ret_to_mu"])
    for lab, x in [("|z|>1.5", E[E.az > 1.5]), ("|z|<1", E[E.az < 1])]:
        P(f"    ret_to_mu {lab}: {summ(x.ret_to_mu)} | H1 {summ(x[x.half == 'H1'].ret_to_mu)} | H2 {summ(x[x.half == 'H2'].ret_to_mu)}")

    if s != "ASIA":
        P("\n  (e) 發展中剖面 z（當天開盤到 IB 前的 TPO 分佈）→ 到收盤報酬（正 = 往發展中 μ）")
        E["ret_to_dev"] = -np.sign(E.dev_z) * E.ret_close
        E["adz"] = E.dev_z.abs()
        bucket_table(E, "adz", [0, 1, 2, 3, 99], ["|z|<1", "1~2", "2~3", ">3"], ["ret_to_dev", "brk_ret"])

    P("\n  (f) 字母樹 z（價格所在列在前日剖面的 TPO 數 z）→ IB 後剩餘波幅（ATR）、首破延續")
    E["tz_b"] = pd.cut(E.tree_z, [-9, -0.5, 0.5, 9], labels=["LVN薄", "中", "HVN厚"])
    for lab in ["LVN薄", "中", "HVN厚"]:
        x = E[E.tz_b == lab]
        P(f"    {lab:6s} n={len(x):3d} rest_range={x.rest_range.mean():.2f}  brk_ret {summ(x.brk_ret)}")
    x = E[E.tree_z.isna()]
    P(f"    前日區間外 n={len(x):3d} rest_range={x.rest_range.mean():.2f}  brk_ret {summ(x.brk_ret)}")

# ---------------- 80% 法則（交易日開盤在前日 VA 外，之後連續兩個 30 分期收在 VA 內） ----------------
P("\n" + "=" * 100)
P("80% 法則：開盤在前日 VA 外，之後連續兩個 30 分期收盤在 VA 內 → 觸及 VA 另一側的機率")
res = []
for i, day in enumerate(DAYS[1:], 1):
    prev = D.iloc[i - 1]; g = G[day]
    o = g.open.iloc[0]
    if prev.val <= o <= prev.vah:
        continue
    side = 1 if o > prev.vah else -1            # 1 = 開在上方，目標 VAL
    pc = g.groupby("period").close.last()
    inside = (pc >= prev.val) & (pc <= prev.vah)
    trig = None
    for k in range(1, len(pc)):
        if inside.iloc[k] and inside.iloc[k - 1]:
            trig = pc.index[k]; break
    if trig is None:
        res.append(dict(day=day, trig=0)); continue
    rest = g[g.period > trig]
    entry = pc.loc[trig]
    tgt = prev.val if side == 1 else prev.vah
    stop = prev.vah if side == 1 else prev.val     # 重新回到 VA 外 = 失敗
    hit_t = rest.index[(rest.low <= tgt) if side == 1 else (rest.high >= tgt)]
    hit_s = rest.index[(rest.high >= stop + 0.0) if side == 1 else (rest.low <= stop)]
    win = len(hit_t) and (not len(hit_s) or hit_t[0] <= hit_s[0])
    res.append(dict(day=day, trig=1, touch=int(len(hit_t) > 0), win=int(bool(win)),
                    ret_close=-side * (g.close.iloc[-1] - entry) / D.atr10.iloc[i], period=trig))
R = pd.DataFrame(res)
P(f"  開盤在前日 VA 外 {len(R)} 天，觸發 {int(R.trig.sum())} 天；觸發後當天碰到另一側 {R.touch.mean() * 100:.0f}%；"
  f"先碰另一側才回 VA 外邊界 {R.win.mean() * 100:.0f}%")
P("  觸發進場 → 收盤報酬（ATR，正 = 往另一側）", summ(R.ret_close.dropna()))

# ---------------- 圖 ----------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    for j, (s, E) in enumerate(ALL.items()):
        x = E.dropna(subset=["ib_z"])
        ax[0, j].scatter(x.ib_z, x.ext_sum, s=12, alpha=.6, color="#3b6ea8")
        ax[0, j].set_title(f"{s}: IB z vs extension (IB units)")
        ax[0, j].set_xlabel("IB range z (20d)"); ax[0, j].set_ylabel("ext_up + ext_dn")
        ax[0, j].grid(alpha=.3)
        b = pd.cut(E.z_prev.abs(), [0, .5, 1, 1.5, 2.5, 99])
        m = E.groupby(b, observed=True).poc_touch.mean()
        ax[1, j].bar(range(len(m)), m.values * 100, color="#c9824a")
        ax[1, j].set_xticks(range(len(m))); ax[1, j].set_xticklabels(["<.5", ".5-1", "1-1.5", "1.5-2.5", ">2.5"][:len(m)])
        ax[1, j].set_title(f"{s}: P(touch prev POC) by |profile z|"); ax[1, j].set_ylabel("%"); ax[1, j].grid(alpha=.3, axis="y")
    plt.tight_layout(); plt.savefig("part10_tpo_ib.png", dpi=110)
    P("\n圖：part10_tpo_ib.png")
except ImportError:
    pass

for s, E in ALL.items():
    E.to_csv(f"part10_events_{s}.csv")
D.to_csv("part10_daily_profile.csv")
OUT.close()
