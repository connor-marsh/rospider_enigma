#!/bin/zsh
export XDG_RUNTIME_DIR="/run/user/$(id -u)"
export PULSE_SERVER="unix:$XDG_RUNTIME_DIR/pulse/native"

USB_SINK=$(
  pactl list sinks | awk '
    BEGIN {
      sink_name = ""
      description = ""
      sysfs_path = ""
      found = 0            # 已经找到符合 level<=11 的结果
      fallback = ""        # 用来保存第一个 level<=12 的候选
    }
    /^Sink #[0-9]+/ {
      # 每次遇到新 Sink 前，先处理上一条记录
      if (sink_name != "") {
        # 先看是否满足严格条件
        if (!found && tolower(description) ~ /usb.*audio device/) {
          n = split(sysfs_path, parts, "/")
          level = n - 1
          if (level <= 11 && sysfs_path != "") {
            print sink_name
            found = 1
            exit
          }
        }
        # 再看是否满足回退条件（level<=12），且只保存第一个
        if (fallback=="" && tolower(description) ~ /usb.*audio device/) {
          n = split(sysfs_path, parts, "/")
          level = n - 1
          if (level <= 12 && sysfs_path != "") {
            fallback = sink_name
          }
        }
      }
      # 重置到新 Sink 的状态
      sink_name = ""
      description = ""
      sysfs_path = ""
    }
    /Name: alsa_output/ {
      sink_name = $2
    }
    /sysfs.path =/ {
      line = $0
      sub(/.*sysfs.path = "/, "", line)
      sub(/".*/, "", line)
      sysfs_path = line
    }
    {
      # 累积描述，方便后面匹配“usb audio device”
      description = description " " tolower($0)
    }
    END {
      # 最后再处理最后一个 Sink
      if (!found && sink_name != "" && tolower(description) ~ /usb.*audio device/) {
        n = split(sysfs_path, parts, "/")
        level = n - 1
        if (level <= 11 && sysfs_path != "") {
          print sink_name
          found = 1
        } else if (fallback=="") {
          # 如果 strict 也不满足，但还未设置 fallback
          if (level <= 12 && sysfs_path != "") {
            fallback = sink_name
          }
        }
      }
      # 如果 strict 没找到，就输出 fallback（可能为空）
      if (!found && fallback != "") {
        print fallback
      }
    }
  '
)

if [ -n "$USB_SINK" ]; then
    echo "set output: $USB_SINK"
    pactl set-default-sink "$USB_SINK"
    pactl set-sink-volume "$USB_SINK" 80%
    echo "set volume: 80%"
else
    echo "cannot find output device"
fi

USB_SOURCE=$(pactl list short sources | grep -v "\.monitor" | awk '
  BEGIN {
    xfm_dp = ""
    usb_audio = ""
  }
  {
    # 优先匹配XFM-DP设备
    if ($2 ~ /XFM-DP/ && xfm_dp == "") {
      xfm_dp = $2
    }
    # 其次匹配GeneralPlus USB设备
    else if ($2 ~ /usb.*audio device/i && usb_audio == "") {
      usb_audio = $2
    }
  }
  END {
    if (xfm_dp != "") {
      print xfm_dp
    } else if (usb_audio != "") {
      print usb_audio
    }
  }
')

if [ -n "$USB_SOURCE" ]; then
    echo "set input: $USB_SOURCE"
    pactl set-default-source "$USB_SOURCE"
else
    echo "cannot find input device"
fi

