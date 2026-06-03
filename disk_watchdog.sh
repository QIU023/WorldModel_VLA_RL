while true; do
  free=$(df --output=avail -BG /home | tail -1 | tr -dc '0-9')
  echo "$(date -u +%H:%M:%SZ) disk_free=${free}GB" >> /home/wm-vla-rl/disk_watchdog.log
  if [ "$free" -lt 15 ]; then
    echo "$(date -u +%H:%M:%SZ) !!! DISK<15GB !!!" >> /home/wm-vla-rl/disk_watchdog.log
    touch /home/wm-vla-rl/DISK_LOW
  fi
  sleep 180
done
