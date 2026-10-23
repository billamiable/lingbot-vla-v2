#!/usr/bin/env bash
# Emulate a cloud block disk (e.g. an Alibaba Cloud ESSD) on local NVMe, to see whether the teacher
# cache still keeps up when it lives on network-attached storage.
#
#   bash essd_sim.sh setup <cache dir>   image file -> loop -> dm-delay -> ext4 at $MNT, copy the cache
#   bash essd_sim.sh delay <ms>          extra read latency (dm-delay, whole ms; 5.15 rounds 1 ms up to ~8 ms)
#   bash essd_sim.sh teardown            unmount and delete everything setup made
#
# Bandwidth and IOPS caps are not set here: they come from the cgroup the reader runs in
# (systemd-run -p IOReadBandwidthMax / IOReadIOPSMax on /dev/mapper/$DM, see run_essd.sh).
# The device read-ahead is set to $RA_KB (128 KB, the virtio-blk default of a cloud VM).
set -euo pipefail
IMG=${IMG:-/raid/essd_sim/disk.img} MNT=${MNT:-/mnt/essd_sim} DM=${DM:-essd_sim} RA_KB=${RA_KB:-128}

set_table() {  # set_table <read delay ms>
  local dev sz
  dev=$(dmsetup table "$DM" | awk '{print $4}'); sz=$(dmsetup table "$DM" | awk '{print $2}')
  dmsetup suspend "$DM"
  dmsetup reload "$DM" --table "0 $sz delay $dev 0 $1 $dev 0 0"
  dmsetup resume "$DM"
  echo "$RA_KB" > "/sys/block/$(basename "$(readlink -f /dev/mapper/$DM)")/queue/read_ahead_kb"
}

case ${1:?usage: essd_sim.sh setup <cache dir> | delay <ms> | teardown} in
  setup)
    src=${2:?cache dir}
    mkdir -p "$(dirname "$IMG")" "$MNT"
    truncate -s "$(( $(du -sb "$src" | cut -f1) * 11 / 10 / 1024**3 + 16 ))G" "$IMG"
    loop=$(losetup --direct-io=on --show -f "$IMG")
    dmsetup create "$DM" --table "0 $(blockdev --getsz "$loop") delay $loop 0 0"
    mkfs.ext4 -q /dev/mapper/$DM
    mount /dev/mapper/$DM "$MNT"
    cp -a "$src" "$MNT/"
    sync
    set_table 0
    echo "cache at $MNT/$(basename "$src")" ;;
  delay) set_table "${2:?ms}" ;;
  teardown)
    umount "$MNT" || true
    loop=$(dmsetup deps -o devname "$DM" 2>/dev/null | grep -o 'loop[0-9]*' || true)
    dmsetup remove "$DM" || true
    [ -n "$loop" ] && losetup -d "/dev/$loop"
    rm -f "$IMG" ;;
esac
