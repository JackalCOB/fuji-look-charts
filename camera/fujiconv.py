"""Drive a Fujifilm camera's USB raw conversion mode from a script.

The camera must be in its USB RAW CONV./BACKUP RESTORE connection mode. The tool speaks PTP
(ISO 15740) over USB bulk transfers, using the camera's conversion workflow as FilmKit's
documentation describes it: upload a raw with the vendor operations 0x900C and 0x900D, read the
conversion profile (device property 0xD185), write a modified profile, trigger the conversion
(property 0xD183), poll for the result object, download it, and delete it. The code is written from
those protocol facts and the PTP standard. No other tool's code is copied.

Runs in WSL2 with the camera attached through usbipd-win, as root, because the USB device node is
root-only there. pyusb comes from the folder named by PYTHONPATH.

    fujiconv.py info
    fujiconv.py profile RAF OUT.bin             upload a raw and save its conversion profile, no conversion
    fujiconv.py convert --out DIR [--as-shot] [--sim NAME=CODE ...] [--set INDEX=VALUE ...] RAF ...
    fujiconv.py batch --manifest CSV --source DIR --out DIR [--repeats N] [--looks NAME ...]
        [--colour-space srgb|adobe]

`convert` uploads each raw once and converts it once per look requested. `--sim` names a look and
the profile value to write at the film-simulation index. `--set` writes further profile fields by
index. Every output is logged with its source checksum, settings, and timing to DIR/convert_log.csv.

`batch` is the production run. It reads the raws a manifest lists, refuses any whose checksum has
changed, and converts each into every look in LOOKS, N times each, writing
DIR/<look>/<stem>__<look>__r<n>.jpg. The colour looks use the fixed settings (FIXED), and
`as_shot` uses the camera's own profile for the raw unchanged. A file that already exists is
skipped, so an interrupted run resumes where it stopped. Each raw's base profile is kept in
DIR/_profiles/. `--colour-space adobe` writes the colour looks in Adobe RGB (profile field 22, 1 sRGB
and 2 Adobe RGB, confirmed by the outputs' interoperability tags); the default leaves the field as the
raw's own profile has it, which for the public raw is sRGB.
"""
import argparse
import csv
import hashlib
import os
import struct
import sys
import time

import usb.core
import usb.util

FUJI_VENDOR = 0x04CB
PTP_CLASS = 6

CMD, DATA, RESP = 1, 2, 3
OP_GET_DEVICE_INFO = 0x1001
OP_OPEN_SESSION = 0x1002
OP_CLOSE_SESSION = 0x1003
OP_GET_OBJECT_HANDLES = 0x1007
OP_GET_OBJECT = 0x1009
OP_DELETE_OBJECT = 0x100B
OP_GET_PROP = 0x1015
OP_SET_PROP = 0x1016
OP_FUJI_SEND_OBJECT_INFO = 0x900C
OP_FUJI_SEND_OBJECT = 0x900D
RESP_OK = 0x2001
RESP_SESSION_ALREADY_OPEN = 0x201E
PROP_CONVERSION_PROFILE = 0xD185
PROP_START_CONVERSION = 0xD183
OBJECT_FORMAT_RAF = 0xF802

CHUNK = 1 << 20

# Profile field holding the film simulation, and the codes of the X-T4's nine colour looks. Each code
# was confirmed on 13 October 2026 by the film simulation the camera wrote into the output.
SIM_INDEX = 8
COLOUR_SPACE_INDEX = 22
COLOUR_SPACES = {"srgb": 1, "adobe": 2}
LOOKS = {"as_shot": None, "provia": 1, "velvia": 2, "astia": 3, "pro_neg_hi": 4, "pro_neg_std": 5,
         "classic_chrome": 11, "eterna": 16, "classic_neg": 17, "eterna_bleach_bypass": 18}

# The fixed conversion settings, by profile field: DR100, D-range priority off, grain, colour
# chrome, smooth skin and colour chrome FX blue off, white balance as shot with no shift, highlight,
# shadow, colour, sharpness and clarity at 0, noise reduction at 0. The field meanings follow FilmKit's
# X100VI map; the README lists which were confirmed against the output's metadata.
FIXED = {6: 100, 7: 0, 9: 1, 10: 1, 11: 1, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0,
         20: 0x2000, 25: 1, 27: 0}


class PTPError(RuntimeError):
    pass


def ptp_string(s):
    if not s:
        return b"\x00"
    chars = s + "\x00"
    return bytes([len(chars)]) + chars.encode("utf-16-le")


def read_ptp_string(buf, pos):
    n = buf[pos]
    pos += 1
    if n == 0:
        return "", pos
    raw = bytes(buf[pos:pos + 2 * n])
    return raw.decode("utf-16-le").rstrip("\x00"), pos + 2 * n


class Camera:
    def __init__(self, timeout_ms=10_000):
        self.dev = None
        for d in usb.core.find(find_all=True, idVendor=FUJI_VENDOR):
            self.dev = d
            break
        if self.dev is None:
            raise PTPError("no Fujifilm USB device found. Is the camera attached to WSL with usbipd?")
        self.dev.set_configuration()
        cfg = self.dev.get_active_configuration()
        intf = next((i for i in cfg if i.bInterfaceClass == PTP_CLASS), None)
        if intf is None:
            raise PTPError("the camera exposes no PTP interface. Check its USB connection mode")
        self.intf = intf.bInterfaceNumber
        if self.dev.is_kernel_driver_active(self.intf):
            self.dev.detach_kernel_driver(self.intf)
        usb.util.claim_interface(self.dev, self.intf)
        ep = lambda direction: usb.util.find_descriptor(
            intf, custom_match=lambda e: usb.util.endpoint_type(e.bmAttributes) == usb.util.ENDPOINT_TYPE_BULK
            and usb.util.endpoint_direction(e.bEndpointAddress) == direction)
        self.ep_out, self.ep_in = ep(usb.util.ENDPOINT_OUT), ep(usb.util.ENDPOINT_IN)
        if self.ep_out is None or self.ep_in is None:
            raise PTPError("bulk endpoints not found on the PTP interface")
        self.max_packet = self.ep_out.wMaxPacketSize
        self.timeout = timeout_ms
        self.tx = 0

    # -- transport --

    def _write(self, payload, timeout=None):
        t = timeout or self.timeout
        view = memoryview(payload)
        for start in range(0, len(view), CHUNK):
            self.ep_out.write(view[start:start + CHUNK], timeout=t)
        if len(payload) % self.max_packet == 0:
            self.ep_out.write(b"", timeout=t)

    def _send(self, ctype, code, tx, params=(), data=b"", timeout=None):
        body = b"".join(struct.pack("<I", p) for p in params) + data
        self._write(struct.pack("<IHHI", 12 + len(body), ctype, code, tx) + body, timeout)

    def _recv(self, timeout=None):
        t = timeout or self.timeout
        buf = bytearray(self.ep_in.read(CHUNK, timeout=t))
        if len(buf) < 12:
            raise PTPError(f"short PTP container ({len(buf)} bytes)")
        total, ctype, code, tx = struct.unpack_from("<IHHI", buf)
        while len(buf) < total:
            buf += self.ep_in.read(CHUNK, timeout=t)
        return ctype, code, tx, bytes(buf[12:total])

    def transact(self, code, params=(), data_out=None, timeout=None):
        """One PTP transaction. Returns (response code, response params, data-in payload)."""
        self.tx += 1
        self._send(CMD, code, self.tx, params, timeout=timeout)
        if data_out is not None:
            self._send(DATA, code, self.tx, data=data_out, timeout=timeout)
        data_in = b""
        ctype, rcode, _, payload = self._recv(timeout)
        if ctype == DATA:
            data_in = payload
            ctype, rcode, _, payload = self._recv(timeout)
        if ctype != RESP:
            raise PTPError(f"expected a response container, got type {ctype}")
        rparams = [struct.unpack_from("<I", payload, i)[0] for i in range(0, len(payload) - len(payload) % 4, 4)]
        return rcode, rparams, data_in

    def check(self, what, code, params=(), data_out=None, timeout=None):
        rcode, rparams, data_in = self.transact(code, params, data_out, timeout)
        if rcode != RESP_OK:
            raise PTPError(f"{what} failed with response 0x{rcode:04X}")
        return data_in

    # -- session --

    def open(self):
        rcode, _, _ = self.transact(OP_OPEN_SESSION, [1])
        if rcode == RESP_SESSION_ALREADY_OPEN:
            self.transact(OP_CLOSE_SESSION)
            rcode, _, _ = self.transact(OP_OPEN_SESSION, [1])
        if rcode != RESP_OK:
            raise PTPError(f"OpenSession failed with response 0x{rcode:04X}")

    def close(self):
        try:
            self.transact(OP_CLOSE_SESSION)
        finally:
            usb.util.release_interface(self.dev, self.intf)
            usb.util.dispose_resources(self.dev)

    def device_info(self):
        d = self.check("GetDeviceInfo", OP_GET_DEVICE_INFO)
        pos = 2 + 4 + 2
        _, pos = read_ptp_string(d, pos)
        pos += 2
        arrays = []
        for _ in range(5):
            (n,) = struct.unpack_from("<I", d, pos)
            pos += 4
            arrays.append(list(struct.unpack_from(f"<{n}H", d, pos)))
            pos += 2 * n
        ops, _, props, _, _ = arrays
        manufacturer, pos = read_ptp_string(d, pos)
        model, pos = read_ptp_string(d, pos)
        version, pos = read_ptp_string(d, pos)
        return {"manufacturer": manufacturer, "model": model, "version": version, "operations": ops,
                "properties": props}

    # -- conversion --

    def upload_raf(self, data):
        info = (struct.pack("<IHHI", 0, OBJECT_FORMAT_RAF, 0, len(data))
                + struct.pack("<HIIIIIIIHII", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
                + ptp_string("FUP_FILE.dat") + b"\x00\x00\x00")
        self.check("SendObjectInfo", OP_FUJI_SEND_OBJECT_INFO, [0, 0, 0], info)
        self.check("SendObject", OP_FUJI_SEND_OBJECT, [], data, timeout=120_000)

    def get_profile(self):
        p = self.check("GetDevicePropValue(0xD185)", OP_GET_PROP, [PROP_CONVERSION_PROFILE])
        if not p:
            raise PTPError("the conversion profile is empty. Was a raw uploaded?")
        return p

    def set_profile(self, profile):
        self.check("SetDevicePropValue(0xD185)", OP_SET_PROP, [PROP_CONVERSION_PROFILE], profile)

    def convert_once(self, timeout_s=60):
        self.check("StartRawConversion(0xD183)", OP_SET_PROP, [PROP_START_CONVERSION], struct.pack("<H", 0))
        start = time.time()
        while time.time() - start < timeout_s:
            d = self.check("GetObjectHandles", OP_GET_OBJECT_HANDLES, [0xFFFFFFFF, 0, 0])
            if len(d) >= 8 and struct.unpack_from("<I", d)[0] > 0:
                handle = struct.unpack_from("<I", d, 4)[0]
                jpeg = self.check("GetObject", OP_GET_OBJECT, [handle], timeout=120_000)
                self.transact(OP_DELETE_OBJECT, [handle])
                return jpeg
            time.sleep(0.25)
        raise PTPError(f"no converted image after {timeout_s} seconds")


def profile_params(profile):
    """The profile's trailing int32 fields: a count at offset 0, the fields at the end."""
    (n,) = struct.unpack_from("<H", profile, 0)
    off = len(profile) - 4 * n
    if n == 0 or off < 2:
        raise PTPError(f"unexpected profile layout: {len(profile)} bytes, {n} fields")
    return off, list(struct.unpack_from(f"<{n}i", profile, off))


def patched(profile, changes):
    off, fields = profile_params(profile)
    out = bytearray(profile)
    for index, value in changes.items():
        if not 0 <= index < len(fields):
            raise PTPError(f"profile field {index} is outside the {len(fields)} fields")
        struct.pack_into("<i", out, off + 4 * index, value)
    return bytes(out)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def cmd_info(cam, _):
    info = cam.device_info()
    print(f"{info['manufacturer']} {info['model']}, firmware {info['version']}")
    print(f"operations {len(info['operations'])}, properties {len(info['properties'])}")
    for name, code, group in [("SendObjectInfo 0x900C", OP_FUJI_SEND_OBJECT_INFO, "operations"),
                              ("SendObject 0x900D", OP_FUJI_SEND_OBJECT, "operations"),
                              ("profile 0xD185", PROP_CONVERSION_PROFILE, "properties"),
                              ("start 0xD183", PROP_START_CONVERSION, "properties")]:
        print(f"  {name}: {'supported' if code in info[group] else 'NOT advertised'}")
    vendor_ops = [f"0x{o:04X}" for o in info["operations"] if o >= 0x9000]
    vendor_props = [f"0x{p:04X}" for p in info["properties"] if 0xD100 <= p <= 0xD2FF]
    print(f"vendor operations: {' '.join(vendor_ops)}")
    print(f"vendor properties 0xD100-0xD2FF: {' '.join(vendor_props)}")


def cmd_profile(cam, args):
    data = open(args.raf, "rb").read()
    t0 = time.time()
    cam.upload_raf(data)
    t1 = time.time()
    profile = cam.get_profile()
    open(args.out, "wb").write(profile)
    off, fields = profile_params(profile)
    print(f"uploaded {len(data):,} bytes in {t1 - t0:.1f} s. Profile {len(profile)} bytes, "
          f"{len(fields)} int32 fields from offset {off}")
    for i, v in enumerate(fields):
        print(f"  [{i:2d}] {v}")


def cmd_convert(cam, args):
    os.makedirs(args.out, exist_ok=True)
    sims = dict(s.split("=", 1) for s in (args.sim or []))
    sims = {k: int(v, 0) for k, v in sims.items()} or {"as_shot": None}
    fixed = {int(k, 0): int(v, 0) for k, v in (s.split("=", 1) for s in (args.set or []))}
    sim_index = int(args.sim_index, 0)
    log_path = os.path.join(args.out, "convert_log.csv")
    new_log = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as logf:
        log = csv.writer(logf)
        if new_log:
            log.writerow(["source", "source_sha256", "look", "changes", "output", "output_sha256",
                          "bytes", "upload_s", "convert_s", "base_profile_sha256"])
        for raf in args.raf:
            data = open(raf, "rb").read()
            t0 = time.time()
            cam.upload_raf(data)
            upload_s = time.time() - t0
            base = cam.get_profile()
            stem = os.path.splitext(os.path.basename(raf))[0]
            for look, code in sims.items():
                changes = {} if args.as_shot else dict(fixed)
                if code is not None and not args.as_shot:
                    changes[sim_index] = code
                cam.set_profile(patched(base, changes) if changes else base)
                t1 = time.time()
                jpeg = cam.convert_once()
                convert_s = time.time() - t1
                out = os.path.join(args.out, f"{stem}__{look}.jpg")
                open(out, "wb").write(jpeg)
                log.writerow([raf, sha256(data), look, ";".join(f"{k}={v}" for k, v in sorted(changes.items())),
                              out, sha256(jpeg), len(jpeg), f"{upload_s:.2f}", f"{convert_s:.2f}", sha256(base)])
                logf.flush()
                print(f"{stem} {look}: {len(jpeg):,} bytes, upload {upload_s:.1f} s, convert {convert_s:.1f} s")
                upload_s = 0.0


def cmd_batch(cam, args):
    rows = [r for r in csv.DictReader(open(args.manifest, encoding="utf-8")) if r.get("role", "raw") == "raw"]
    looks = {k: LOOKS[k] for k in (args.looks or LOOKS)}
    extra = {COLOUR_SPACE_INDEX: COLOUR_SPACES[args.colour_space]} if args.colour_space else {}
    os.makedirs(os.path.join(args.out, "_profiles"), exist_ok=True)
    log_path = os.path.join(args.out, "convert_log.csv")
    new_log = not os.path.exists(log_path)
    done = skipped = 0
    started = time.time()
    with open(log_path, "a", newline="") as logf:
        log = csv.writer(logf)
        if new_log:
            log.writerow(["time", "source", "source_sha256", "look", "repeat", "changes", "output",
                          "output_sha256", "bytes", "upload_s", "convert_s", "base_profile_sha256"])
        for n, row in enumerate(rows, 1):
            stem = os.path.splitext(row["file"])[0]
            wanted = [(look, code, r) for look, code in looks.items() for r in range(1, args.repeats + 1)
                      if not os.path.exists(os.path.join(args.out, look, f"{stem}__{look}__r{r}.jpg"))]
            if not wanted:
                skipped += len(looks) * args.repeats
                continue
            data = open(os.path.join(args.source, row["file"]), "rb").read()
            if sha256(data) != row["sha256"]:
                raise PTPError(f"{row['file']}: checksum differs from the manifest, refusing to convert")
            t0 = time.time()
            cam.upload_raf(data)
            upload_s = time.time() - t0
            base = cam.get_profile()
            open(os.path.join(args.out, "_profiles", f"{stem}.bin"), "wb").write(base)
            for look, code, r in wanted:
                changes = {} if code is None else {**FIXED, **extra, SIM_INDEX: code}
                cam.set_profile(patched(base, changes) if changes else base)
                t1 = time.time()
                jpeg = cam.convert_once()
                convert_s = time.time() - t1
                folder = os.path.join(args.out, look)
                os.makedirs(folder, exist_ok=True)
                out = os.path.join(folder, f"{stem}__{look}__r{r}.jpg")
                with open(out + ".part", "wb") as f:
                    f.write(jpeg)
                os.replace(out + ".part", out)
                log.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), row["file"], row["sha256"], look, r,
                              ";".join(f"{k}={v}" for k, v in sorted(changes.items())), out, sha256(jpeg),
                              len(jpeg), f"{upload_s:.2f}", f"{convert_s:.2f}", sha256(base)])
                logf.flush()
                upload_s = 0.0
                done += 1
            elapsed = time.time() - started
            print(f"[{n}/{len(rows)}] {stem}: {len(wanted)} conversions, {done} done this run, "
                  f"{elapsed / 60:.1f} min elapsed", flush=True)
    print(f"finished: {done} converted, {skipped} already present")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info")
    p = sub.add_parser("profile")
    p.add_argument("raf")
    p.add_argument("out")
    c = sub.add_parser("convert")
    c.add_argument("--out", required=True)
    c.add_argument("--as-shot", action="store_true", help="convert with the camera's own profile, unchanged")
    c.add_argument("--sim", action="append", help="NAME=CODE, the value written at --sim-index")
    c.add_argument("--sim-index", default="8", help="profile field holding the film simulation")
    c.add_argument("--set", action="append", help="INDEX=VALUE, a further profile field to write")
    c.add_argument("raf", nargs="+")
    b = sub.add_parser("batch")
    b.add_argument("--manifest", required=True, help="CSV with file, sha256, and optionally role columns")
    b.add_argument("--source", required=True, help="folder the manifest's files are in")
    b.add_argument("--out", required=True)
    b.add_argument("--repeats", type=int, default=5)
    b.add_argument("--looks", nargs="+", choices=list(LOOKS), help="default: every look in LOOKS")
    b.add_argument("--colour-space", choices=list(COLOUR_SPACES), help="colour looks only; default: the raw's own")
    args = ap.parse_args()
    cam = Camera()
    cam.open()
    try:
        {"info": cmd_info, "profile": cmd_profile, "convert": cmd_convert, "batch": cmd_batch}[args.cmd](cam, args)
    finally:
        cam.close()


if __name__ == "__main__":
    try:
        main()
    except (PTPError, usb.core.USBError) as e:
        sys.exit(f"fujiconv: {e}")
