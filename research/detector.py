"""晨星/夜星（廣義：趨勢段末端的 2~4 根反轉組合）偵測器。

訊號在「確認 K 棒」收盤時產生，不使用任何未來資料。
多頭（晨星）邏輯，空頭鏡像：
  1. 極值 K 棒 e（位於確認棒前 1~max_wait 根）的低點是過去 swing_len 根的最低點，
     且 e 之後到確認棒之間沒有再破低。
  2. 前段幅度：e 之前 run_len 根內的最高點到 e 低點的距離 >= run_atr × ATR60。
  3. 前段效率：ER(er_len) 於 e 時 >= er_min（走勢夠「直」）。
  4. 確認：確認棒收紅 (close > open)，且收盤距極值 >= confirm_atr × ATR60。
"""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from features import add_features


@dataclass
class Params:
    swing_len: int = 20
    run_len: int = 12
    run_atr: float = 3.0
    er_len: int = 8
    er_min: float = 0.0
    max_wait: int = 3
    confirm_atr: float = 1.0
    need_color: bool = True


def detect(df, p: Params = Params()):
    f = df if "atr60" in df else add_features(df)
    o, h, l, c = (f[k].to_numpy() for k in ("open", "high", "low", "close"))
    A = f.atr60.to_numpy()
    er = f[f"er{p.er_len}"].to_numpy()
    n = len(f)
    out = []
    used = set()  # 每個極值只發一次訊號
    for t in range(max(p.swing_len, 61) + p.max_wait, n):
        for side in (1, -1):
            for k in range(1, p.max_wait + 1):
                e = t - k
                if (e, side) in used:
                    continue
                if side == 1:
                    ext = l[e]
                    if ext > l[e - p.swing_len:e].min() or l[e + 1:t + 1].min() < ext:
                        continue
                    run = h[e - p.run_len:e + 1].max() - ext
                    conf = c[t] - ext
                    color = c[t] > o[t]
                else:
                    ext = h[e]
                    if ext < h[e - p.swing_len:e].max() or h[e + 1:t + 1].max() > ext:
                        continue
                    run = ext - l[e - p.run_len:e + 1].min()
                    conf = ext - c[t]
                    color = c[t] < o[t]
                if run < p.run_atr * A[e] or er[e] < p.er_min:
                    continue
                if conf < p.confirm_atr * A[e] or (p.need_color and not color):
                    continue
                used.add((e, side))
                out.append(dict(time=f.index[t], ext_time=f.index[e], side=side, ext=ext, entry=c[t],
                                atr=A[e], run_atr=run / A[e], er=er[e], conf_atr=conf / A[e]))
                break
    return pd.DataFrame(out)
