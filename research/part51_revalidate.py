"""第五十一部分：第 49 部分裡「看起來有用、但可能是調參數調出來的」候選，全部用 2023 ~ 2025 重新驗證。

參數完全照當時（只用 2026 決定的）設定，不改。全部是 M15（M5 / M1 候選需要 M1 資料，這份 M15 檔無法驗證）。
出場統一：下一根開盤進、1.5 ATR14 停損、L 根收盤出、扣點差、不重疊（49e 的 backtest）。
超額 = 每筆 R − 同年同 L「全部順勢」。逆勢規則另列同年「全部逆勢」基準。
候選（括號內是第 49 部分在 2026 的結果）
  A 49   SLIP_FOLLOW：f 滾動 2000 根百分位 ≤ 20% → 順勢，L48（+0.31R）
  B 49   STICK_FADE ：f 百分位 ≥ 80% → 逆勢，L24（+0.12R，後半 +0.21R）
  C 49   STICK_FADE ：同上，L48（f 高 → 偏反轉 Q5−Q1 −1.8 ATR）
  D 49b  z(ln f) ≤ −2（SMA W500）→ 順勢，L48（+0.52R）
  E 49c  滑 z(ln μ) ≤ −1 且 加速 a_rel > 0（SMA W500）→ 順勢，L48（+0.44R）
  F 49c  z(cosθ) ≤ −2（SMA W500）→ 順勢，L48（+0.61R）——即很陡，第 50 部分已驗
  G 49d  轉折角 越走越平（< −0.3）→ 順勢，L48（超額 +0.07）
  H 49d  位置 回檔區（pos < 0.6）→ 順勢，L48（超額 +0.12）
  I 49d  很陡（SMA W500 z(cosθ) ≤ −1）+ 大週期同向 → 順勢，L48（超額 +0.15）
  J 49d  大週期同向 + pos < 0.6 → 順勢，L48（超額 +0.04）
"""
import numpy as np
import pandas as pd
from load import load
from part49_friction import features, roll_pct, tst
from part49e_mua_detail import backtest
from part49g_zwin_m5 import z_sma

PATH = "data/XAUUSD_M15_2023_2026.csv"
PERS = [2023, 2024, 2025, "23-25", 2026]


def split(T):
    y = T.entry_time.dt.year
    return {2023: T[y == 2023], 2024: T[y == 2024], 2025: T[y == 2025], "23-25": T[y <= 2025], 2026: T[y == 2026]}


def run(df, F, sig, L, flip=False):
    G = F.copy()
    if flip:
        G["dir"] = -G["dir"]
    return backtest(df, G, sig, L)


def main():
    df = load(PATH)
    c = df.close
    Fs = {L: features(df, L) for L in (24, 48)}
    base = {}
    for L, F in Fs.items():
        valid = F.a.notna() & F.atr.notna() & (F.dir.fillna(0) != 0)
        base[(L, False)] = {k: v.R.mean() for k, v in split(run(df, F, valid, L)).items()}
        base[(L, True)] = {k: v.R.mean() for k, v in split(run(df, F, valid, L, True)).items()}

    F48, F24 = Fs[48], Fs[24]
    v48 = F48.a.notna() & F48.atr.notna() & (F48.dir.fillna(0) != 0)
    v24 = F24.a.notna() & F24.atr.notna() & (F24.dir.fillna(0) != 0)
    p48, p24 = roll_pct(F48.f), roll_pct(F24.f)
    L, h = 48, 24
    th1 = np.arctan((c.shift(h) - c.shift(L)) / (F48.atr * np.sqrt(h)))
    th2 = np.arctan((c - c.shift(h)) / (F48.atr * np.sqrt(h)))
    turn = (th2 - th1) * F48.dir
    hi, lo = df.high.rolling(L).max(), df.low.rolling(L).min()
    pos = ((c - lo) / (hi - lo)).where(F48.dir > 0, (hi - c) / (hi - lo))
    big = np.sign(c - c.shift(4 * L)) * F48.dir
    zc = z_sma(F48.cos, 500)
    cands = [
        ("A SLIP_FOLLOW f≤20%  L48 順", 48, v48 & (p48 <= 0.2), False),
        ("B STICK_FADE  f≥80%  L24 逆", 24, v24 & (p24 >= 0.8), True),
        ("C STICK_FADE  f≥80%  L48 逆", 48, v48 & (p48 >= 0.8), True),
        ("D z(ln f)≤−2        L48 順", 48, v48 & (z_sma(np.log(F48.f.clip(lower=1e-6)), 500) <= -2), False),
        ("E 滑 z(lnμ)≤−1 且加速 L48 順", 48, v48 & (z_sma(np.log(F48.mu), 500) <= -1) & (F48.a_rel > 0), False),
        ("F 很陡 z(cosθ)≤−2   L48 順", 48, v48 & (zc <= -2), False),
        ("G 轉折 越走越平     L48 順", 48, v48 & (turn < -0.3), False),
        ("H 位置 回檔區       L48 順", 48, v48 & (pos < 0.6), False),
        ("I 很陡≤−1+大週期同向 L48 順", 48, v48 & (zc <= -1) & (big > 0), False),
        ("J 大週期同向+回檔區  L48 順", 48, v48 & (big > 0) & (pos < 0.6), False),
    ]
    out = ["第五十一部分：第 49 部分候選的樣本外重新驗證（2023 ~ 2025，參數不改）", "",
           "全部順勢基準 L48: " + "  ".join(f"{k} {v:+.3f}" for k, v in base[(48, False)].items()),
           "全部順勢基準 L24: " + "  ".join(f"{k} {v:+.3f}" for k, v in base[(24, False)].items()),
           "全部逆勢基準 L48: " + "  ".join(f"{k} {v:+.3f}" for k, v in base[(48, True)].items()),
           "全部逆勢基準 L24: " + "  ".join(f"{k} {v:+.3f}" for k, v in base[(24, True)].items()), ""]
    rows = []
    for name, LL, sig, flip in cands:
        T = run(df, Fs[LL], sig, LL, flip)
        out.append(f"[{name}]")
        for k, s in split(T).items():
            if len(s) < 10:
                out.append(f"  {k}: n={len(s)}")
                continue
            m, t, _ = tst(s.R.to_numpy())
            p = s.pnl
            pf = p[p > 0].sum() / -p[p < 0].sum()
            ex = m - base[(LL, flip)][k]
            out.append(f"  {str(k):6s} n={len(s):4d} 勝{(p > 0).mean():.0%} {m:+.3f}R(t={t:+.1f}) 超額{ex:+.3f} PF{pf:.2f}")
            rows.append(dict(cand=name, per=str(k), n=len(s), R=m, t=t, ex=ex, pf=pf))
        out.append("")
    S = pd.DataFrame(rows)
    piv = S.pivot(index="cand", columns="per", values="ex")[["2023", "2024", "2025", "23-25", "2026"]]
    pf = S.pivot(index="cand", columns="per", values="pf")["23-25"]
    tt = S.pivot(index="cand", columns="per", values="t")["23-25"]
    piv["23-25 PF"] = pf
    piv["23-25 t"] = tt
    piv["三年超額皆正"] = (piv[["2023", "2024", "2025"]] > 0).all(axis=1)
    out.append("===== 超額總表（每筆 R − 同年同方向全部進場）=====")
    out.append(piv.round(3).to_string())
    txt = "\n".join(out)
    print(txt)
    with open("results_part51.txt", "w") as fh:
        fh.write(txt + "\n")
    S.to_csv("part51_summary.csv", index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
