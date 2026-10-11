"""第二十九部分：IB 價值區（VAH / VAL）上的底倉與網格。日內、23:50 全平。

IB 價值區：IB 60 根 M1 做時間-價格剖面（價格格子 = 0.01 × ATR10，每根 M1 在它碰到的每個格子各記 1），
          從 POC 用 CBOT 法擴到 70% → VAH / VAL（與 tpo.py 同一個 poc_va）。
IB 起點：01:00 亞洲、10:00 倫敦、16:30 紐約。格距 g = 0.05 / 0.10 × ATR10。

想法一（S1：VA 邊緣底倉 + 順向網格，一天一個方向）
  IB 結束時價格已在 VAH 上方 → 立刻做多；在 VAL 下方 → 立刻做空；在 VA 內 → 掛 VAH 買進停止單 / VAL 賣出停止單，先到先做。
  停損：VA 另一側（多單 = VAL）或 IB 另一側（多單 = IB 低），碰到全平、今天停止。
  元件（以進場價 E 為格線起點，k = 目前在順方向第幾條線，M = 進場後到過的最高線）：
    BASE 底倉 1 單位
    LT   順向追高點的逢回買：從「進場後最高點」每回檔一格買 1 單位、反彈一格就賣（持倉 = M − k）
    L    固定錨點逢回買（同第二十八部分，持倉 = max(0, −k)）
    PU   順勢加碼（持倉 = max(0, k)）

想法二（S2：VAH 上方做多網格、VAL 下方做空網格，一天可以來回切換）
  多方區：格線從 VAH 起算（VAH + j·g）。價格往上碰到 VAH 進入多方區；跌到 VAH − g 離開，多方區的網格與底倉全平。
  空方區：從 VAL 起算，鏡像。IB 結束時價格已在區內就立刻啟動。
  元件：
    ZB  區內底倉：在多方區持有 1 多、在空方區持有 1 空（= VA 突破、跌回就出、可重進）
    ZLT 區內順向逢回買：從進入多方區後的最高線每回檔一格買 1 單位、反彈一格賣（持倉 = M − k，k ≥ 0）
    ZPU 區內順勢加碼：持倉 = k（VAH 上方第 k 條線持有 k 單位）

成交：M1 內部路徑 陽線 O→L→H→C、陰線 O→H→L→C，格線價 / 停止單價成交（跳空以開盤價），買進付點差。
基準：同模型跑在每根 M1 隨機上下鏡像的 K 棒上（無方向性）。樣本內 2023~2024、樣本外 2025~2026。
單位：每日損益 ÷ ATR10（ATR × 盎司 / 天）。
"""
import numpy as np, pandas as pd
from multiprocessing import Pool
from part28_ib_grid import DAYS, mirrored, FLAT, CUT, t
from tpo import poc_va

SEEDS, IB_MIN = 10, 60
STARTS = {"01:00 亞洲": 60, "10:00 倫敦": 600, "16:30 紐約": 990}
MS = (0.05, 0.10)


def ib_va(ohlc, atr):
    tick = 0.01 * atr
    lo0 = np.floor(ohlc[:, 2].min() / tick)
    a = (np.floor(ohlc[:, 2] / tick) - lo0).astype(int); b = (np.floor(ohlc[:, 1] / tick) - lo0).astype(int)
    cnt = np.zeros(b.max() + 1, int)
    for i, j in zip(a, b): cnt[i:j + 1] += 1
    prices = (lo0 + np.arange(len(cnt))) * tick
    _, vah, val = poc_va(prices, cnt)
    return vah + tick, val


def path(ohlc, j):
    o, h, l, c = ohlc[j]
    return (o, l, h, c) if c >= o else (o, h, l, c)


class Book:
    def __init__(self, names): self.pos = dict.fromkeys(names, 0); self.cash = dict.fromkeys(names, 0.0)

    def to(self, tg, px, s):
        for nm, v in tg.items():
            q = v - self.pos[nm]
            if q: self.cash[nm] += -q * px - (s * q if q > 0 else 0); self.pos[nm] += q

    def flat(self, px, s): self.to(dict.fromkeys(self.pos, 0), px, s)


def s1_day(mins, ohlc, sp, atr, s0, g, stop_kind):
    ib = (mins >= s0) & (mins < s0 + IB_MIN)
    if ib.sum() < IB_MIN * 0.8: return None
    vah, val = ib_va(ohlc[ib], atr); ibh, ibl = ohlc[ib, 1].max(), ohlc[ib, 2].min()
    idx = np.where((mins >= s0 + IB_MIN) & (mins < FLAT))[0]
    if len(idx) < 2: return None
    side = None; c0 = ohlc[idx[0] - 1, 3]
    if c0 > vah or c0 < val:                       # IB 結束時已在 VA 外 → 下一根開盤市價進場
        side, j0, p0, E = (1 if c0 > vah else -1), idx[0], 0, ohlc[idx[0], 0]
    else:                                           # 停止單：沿 M1 路徑，先碰到 VAH 或 VAL 的那一邊
        for j in idx:
            for q, x in enumerate(path(ohlc, j)):
                if x >= vah or x <= val:
                    side = 1 if x >= vah else -1
                    E = (max(vah, x) if side == 1 else min(val, x)) if q == 0 else (vah if side == 1 else val)
                    j0, p0 = j, q; break
            if side: break
        if side is None: return None
    stop = (val if side == 1 else vah) if stop_kind == "VA" else (ibl if side == 1 else ibh)
    B = Book(("BASE", "LT", "L", "PU")); kd = M = 0      # kd = 順方向第幾條線（E + side·kd·g）

    def tg(): return dict(BASE=side, LT=side * (M - kd), L=side * max(0, -kd), PU=side * max(0, kd))

    B.to(tg(), E, sp[j0]); last = j0
    for j in idx[idx >= j0]:
        last = j; pts = path(ohlc, j)[(p0 + 1 if j == j0 else 0):]
        for q, x in enumerate(pts):
            y = side * (x - E) / g                       # 順方向距離（格）
            if side * (x - stop) <= 0:
                gap = q == 0 and j != j0
                B.flat(x if gap else stop, sp[j]); return dict(side=side, va=(vah - val) / atr, R=abs(E - stop) / atr, **B.cash)
            while y >= kd + 1 or y <= kd - 1:
                kd += 1 if y >= kd + 1 else -1
                M = max(M, kd)
                B.to(tg(), E + side * kd * g, sp[j])
    B.flat(ohlc[last, 3], sp[last])
    return dict(side=side, va=(vah - val) / atr, R=abs(E - stop) / atr, **B.cash)


def s2_day(mins, ohlc, sp, atr, s0, g):
    ib = (mins >= s0) & (mins < s0 + IB_MIN)
    if ib.sum() < IB_MIN * 0.8: return None
    vah, val = ib_va(ohlc[ib], atr)
    idx = np.where((mins >= s0 + IB_MIN) & (mins < FLAT))[0]
    if len(idx) < 2: return None
    B = Book(("ZB", "ZLT", "ZPU"))
    # 每一區一組格線：多方區 u = (x − VAH)/g、空方區 u = (VAL − x)/g；k = 最後碰到的線
    # 碰到線 0 → 啟動（M = 0）；碰到線 −1 → 離開、該區全平
    Z = {1: dict(ref=vah), -1: dict(ref=val)}

    def target():
        tg = dict(ZB=0, ZLT=0, ZPU=0)
        for sd, z in Z.items():
            if z["on"]: tg["ZB"] += sd; tg["ZLT"] += sd * (z["M"] - z["k"]); tg["ZPU"] += sd * z["k"]
        return tg

    j0 = idx[0]; x0 = ohlc[j0, 0]
    for sd, z in Z.items():
        z["k"] = int(np.floor(sd * (x0 - z["ref"]) / g)); z["on"] = z["k"] >= 0; z["M"] = max(z["k"], 0)
    B.to(target(), x0, sp[j0])
    for j in idx:
        for x in path(ohlc, j)[(1 if j == j0 else 0):]:
            for sd, z in Z.items():
                u = sd * (x - z["ref"]) / g
                while u >= z["k"] + 1 or u <= z["k"] - 1:
                    z["k"] += 1 if u >= z["k"] + 1 else -1
                    if z["k"] == 0 and not z["on"]: z["on"], z["M"] = True, 0
                    elif z["k"] == -1 and z["on"]: z["on"] = False
                    if z["on"]: z["M"] = max(z["M"], z["k"])
                    if z["k"] >= -1: B.to(target(), z["ref"] + sd * z["k"] * g, sp[j])
    B.flat(ohlc[idx[-1], 3], sp[idx[-1]])
    return dict(va=(vah - val) / atr, **B.cash)


def job(args):
    kind, s0, m, stop_kind, seed = args
    rng = np.random.default_rng(seed) if seed is not None else None
    rows = []
    for k, a, mins, ohlc, sp in DAYS:
        oh = mirrored(ohlc, rng) if rng is not None else ohlc
        r = s1_day(mins, oh, sp, a, s0, m * a, stop_kind) if kind == "S1" else s2_day(mins, oh, sp, a, s0, m * a)
        if r: rows.append(dict(day=k, atr=a, **r))
    return args, pd.DataFrame(rows)


COMBOS = {"S1": {"BASE（VA 邊緣底倉）": ["BASE"], "LT（追高點逢回買）": ["LT"], "BASE + LT": ["BASE", "LT"],
                 "L（固定錨點逢回買）": ["L"], "BASE + L": ["BASE", "L"], "PU（順勢加碼）": ["PU"], "BASE + PU": ["BASE", "PU"]},
          "S2": {"ZB（區內底倉）": ["ZB"], "ZLT（區內逢回買網格）": ["ZLT"], "ZB + ZLT": ["ZB", "ZLT"],
                 "ZPU（區內順勢加碼）": ["ZPU"], "ZB + ZPU": ["ZB", "ZPU"]}}

if __name__ == "__main__":
    cfgs = [("S1", s0, m, sk) for s0 in STARTS.values() for m in MS for sk in ("VA", "IB")] + \
           [("S2", s0, m, "-") for s0 in STARTS.values() for m in MS]
    jobs = [c + (sd,) for c in cfgs for sd in [None] + list(range(SEEDS))]
    with Pool(4) as pool:
        res = dict(pool.map(job, jobs, chunksize=1))
    out = open("results_part29.txt", "w")
    def P(s=""): print(s); out.write(s + "\n")
    P("第二十九部分：IB 價值區（VAH / VAL）的底倉與網格（日內、23:50 全平）")
    P("單位：每日損益 ÷ ATR10；括號 = t；基準 = 鏡像 K 棒（無方向性）同模型 10 次平均；樣本內 2023~24 / 樣本外 2025~26。")
    keep = []
    name = {v: k for k, v in STARTS.items()}
    for c in cfgs:
        kind, s0, m, sk = c
        real = res[c + (None,)]; base = pd.concat([res[c + (sd,)] for sd in range(SEEDS)])
        IS = (real.day.dt.year < CUT).values
        hdr = "想法一 S1：VA 邊緣底倉 + 順向網格" if kind == "S1" else "想法二 S2：VAH 上做多網格 / VAL 下做空網格"
        extra = f"｜停損 {'VA 另一側' if sk == 'VA' else 'IB 另一側'}｜R 中位 {real.R.median():.2f} ATR" if kind == "S1" else ""
        P(f"\n{hdr}｜IB {name[s0]}｜格距 {m} ATR{extra}｜{len(real)} 天，VA 寬中位 {real.va.median():.2f} ATR")
        P(f"  {'組合':22s} | {'樣本內':>13s} {'樣本外':>13s} | {'最差日':>7s} | {'基準':>6s} | 逐年")
        for nm, cols in COMBOS[kind].items():
            a = real[cols].sum(axis=1) / real.atr; b = base[cols].sum(axis=1) / base.atr
            yr = " ".join(f"{y % 100}:{a[(real.day.dt.year == y).values].mean():+.3f}" for y in range(2023, 2027))
            P(f"  {nm:22s} | {a[IS].mean():+.3f}({t(a[IS]):+5.2f}) {a[~IS].mean():+.3f}({t(a[~IS]):+5.2f}) | "
              f"{a.min():+7.2f} | {b.mean():+6.3f} | {yr}")
            keep.append(dict(kind=kind, ib=name[s0], m=m, stop=sk, combo=nm, IS=a[IS].mean(), t_IS=t(a[IS]),
                             OOS=a[~IS].mean(), t_OOS=t(a[~IS]), worst=a.min(), base=b.mean(),
                             pos_years=sum(a[(real.day.dt.year == y).values].mean() > 0 for y in range(2023, 2027))))
    pd.DataFrame(keep).to_csv("part29_summary.csv", index=False, float_format="%.4f")
    out.close()
