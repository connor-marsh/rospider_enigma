#!/bin/bash
sleep 5

do_expand_fs() {
    echo "Expanding $PARTITION..."
    PARTITION="/dev/nvme0n1p1"
    PARTUUID=$(sudo blkid -s PARTUUID -o value $PARTITION)
    PARTITION_NUM=${PARTITION##*e0n*p}
    DISK=${PARTITION%%p*}
    PART_START=$(sudo parted ${DISK} -ms unit s p | grep "^${PARTITION_NUM}:" | cut -f 2 -d: | sed 's/[^0-9]//g')
    [ "$PART_START" ] || return 1
    echo -e "d\n$PARTITION_NUM\nn\n$PARTITION_NUM\n$PART_START\n\nN\n\nw\n\n"|fdisk $DISK 2>&1
    echo -e "x\nc\n$PARTITION_NUM\n$PARTUUID\nm\nw\ny\nq\n" | sudo gdisk $DISK
    sleep 0.5
    echo -e "c\n1\nAPP\nw\ny\nq\n" | sudo gdisk $DISK
    sleep 0.5
    resize2fs $PARTITION
}

if [[ $EUID -ne 0 ]]; then
    echo "This script must be run as root!" 1>&2
    sudo $0 $@
fi

do_expand_fs

if [[ !($@ =~ "no_reboot") ]];then
    output=$(lsusb -t)

    if echo "$output" | grep -q "Port 1: Dev 2, If 0, Class=Hub, Driver=hub/4p, 10000M"; then
        echo "Device found: Port 1: Dev 2, If 0, Class=Hub, Driver=hub/4p, 10000M"
    else
        echo "Device not found."
        #sudo cp /home/ubuntu/.dtb/sub/tegra234-p3767-camera-p3768-imx219-dual.dtbo /boot/
        #sudo cp /home/ubuntu/.dtb/sub/kernel_tegra234-p3768-0000+p3767-0003-nv-super.dtb /boot/dtb/
        #sudo cp /home/ubuntu/.dtb/sub/extlinux.conf /boot/extlinux/
        #sudo sed -i 's#exit 0#echo device > /sys/class/usb_role/usb2-0-role-switch/role\nexit 0#g' /opt/nvidia/l4t-usb-device-mode/nv-l4t-usb-device-mode-start.sh
        #sudo cp /home/ubuntu/.dtb/sub/*.rules /etc/udev/rules.d/
    fi
    sleep 1
    sudo systemctl disable expand_rootfs.service
    sudo reboot -d -f
fi
