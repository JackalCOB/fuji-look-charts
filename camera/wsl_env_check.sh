#!/bin/sh
# Report what the WSL side has for the camera converter: kernel, Python, libusb, pyusb, and the
# camera as Linux sees it. Read-only.
echo "kernel: $(uname -r)  user: $(whoami)"
python3 --version
ls /usr/lib/x86_64-linux-gnu/ 2>/dev/null | grep -i '^libusb' || echo "libusb: not found"
python3 -c 'import usb; print("pyusb", usb.__version__)' 2>/dev/null || echo "pyusb: not installed"
python3 -c 'import venv, ensurepip; print("venv and ensurepip: present")' 2>/dev/null || echo "venv or ensurepip: missing"
command -v lsusb >/dev/null && lsusb || echo "lsusb: not installed"
for d in /sys/bus/usb/devices/*; do
    [ -f "$d/idVendor" ] || continue
    if [ "$(cat "$d/idVendor")" = "04cb" ]; then
        echo "Fujifilm device at $d: $(cat "$d/idVendor"):$(cat "$d/idProduct") $(cat "$d/product" 2>/dev/null)"
        ls -l "/dev/bus/usb/$(printf '%03d' "$(cat "$d/busnum")")/$(printf '%03d' "$(cat "$d/devnum")")"
    fi
done
