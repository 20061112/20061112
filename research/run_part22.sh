#!/bin/sh
# 第二十二部分：均值定義 × 回復力。輸出 results_part22.txt 與 part22_*.csv / *.png
set -e
cd "$(dirname "$0")"
for m in ema20 ema50 kama rvwap20 linreg20; do   # 隨機對照（慢，平行跑；每個約 20 分鐘）
  [ -f part22_null_$m.txt ] || python3 part22_robust.py null $m &
done
wait
rm -f results_part22.txt
python3 part22_means.py      # A 均值定義比較
python3 part22_restore.py    # B 事件 IC（均值 × 量測）
python3 part22_beta.py       # E 狀態相依 β（核心框架）
python3 part22_robust.py     # F 穩健性（隨機對照、跨週期、前瞻長度）
python3 part22_pairs.py      # G 兩兩配對契合度
python3 part22_rfi.py        # H 各均值各自挑選的 RFI
python3 part22_final.py      # I 通用 RFI-U 與雙向交易
python3 part22_crossA.py     # J 對版本 A 的過濾
