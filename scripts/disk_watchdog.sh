#!/usr/bin/env bash
# 磁盘 watchdog：每 5 分钟查 /home 剩余空间。低于阈值 → 写 DISK_STOP（训练循环自检后优雅退出）+ 杀长跑训练，防写满崩坏。
# 启动：nohup bash scripts/disk_watchdog.sh & ；停止：kill 或 touch scripts/WATCHDOG_STOP
WARN_GB=30
CRIT_GB=15
LOG=/home/wm-vla-rl/scripts/disk_watchdog.log
STOP=/home/wm-vla-rl/scripts/DISK_STOP
SELF_STOP=/home/wm-vla-rl/scripts/WATCHDOG_STOP
rm -f "$STOP" "$SELF_STOP"
echo "[watchdog start $(date -u +%H:%M:%SZ)] WARN<${WARN_GB}G CRIT<${CRIT_GB}G" >> "$LOG"
while true; do
  [ -f "$SELF_STOP" ] && { echo "[watchdog stop $(date -u +%H:%M:%SZ)]" >> "$LOG"; break; }
  free=$(df -BG --output=avail /home | tail -1 | tr -dc '0-9')
  ts=$(date -u +%H:%M:%SZ)
  if [ "${free:-999}" -lt "$CRIT_GB" ]; then
    echo "$ts CRITICAL free=${free}G < ${CRIT_GB}G -> DISK_STOP + kill 微调" >> "$LOG"
    touch "$STOP"
    pkill -f "exp_p1.*finetune\|finetune_kda" 2>/dev/null
  elif [ "${free:-999}" -lt "$WARN_GB" ]; then
    echo "$ts WARN free=${free}G" >> "$LOG"
  else
    echo "$ts ok free=${free}G" >> "$LOG"
  fi
  sleep 300
done
