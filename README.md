# fuji-look-charts

Tools for measuring a Fujifilm camera's film simulations from synthetic charts that the camera itself converts. They
were written for LightCraft's measured X-T4 looks ([storytold/lightcraft#2445](https://github.com/storytold/lightcraft/pull/2445)).
A chart is a copy of a real raw with its sensor data replaced by patches of known colour. The camera converts each
chart in each film simulation over USB, the tools read the patches back, and a lookup is fitted per look in the form
LightCraft loads.

Every figure these tools produce is a measurement of the camera's own output. None of it is Fujifilm's data or
processing, and the looks they make are not made or endorsed by Fujifilm. Film simulation names are used only to say
which look was measured.

## What you need

- A camera with a USB raw conversion mode, the one Fujifilm X RAW STUDIO uses, set to USB RAW CONV./BACKUP RESTORE
- An uncompressed raw from that body to build the charts from, ideally CC0. The X-T4 looks used raw.pixls.us file
  3913
- Python 3.11 or later with numpy, SciPy and Pillow, and pyusb with libusb for the camera
- LightCraft's `rawdump` example, which decodes raws for these tools. Build it in a clone of
  [storytold/lightcraft](https://github.com/storytold/lightcraft) with
  `cargo build --release -p lightcraft-raw --example rawdump`, and set `LIGHTCRAFT_RAWDUMP` to its path
- ExifTool, for `analysis/check_conversion_tags.py` only
- On Windows, WSL2 with the camera attached through usbipd-win. `camera/wsl_env_check.sh` reports what the WSL side
  has

## The procedure

1. **Pilots.** `charts/layouts_pilot.py` writes an edge pilot and a geometry grid. Build each into a chart with
   `charts/make_chart.py`, check it with `charts/verify_chart.py` (LightCraft must decode exactly what was stored),
   and convert it with `camera/fujiconv.py convert`. `analysis/edge_pilot.py` measures how far the camera carries
   colour across an edge, which sets the patch size. `analysis/geometry.py` fits where each sensor position lands in
   the camera's output, because the camera corrects its lens's distortion when it converts
2. **Tone and corner brightening.** `charts/layouts_tone.py` writes tone ladders and flat fields. Once they are
   converted, `analysis/tone_ladder.py` measures each look's tone response, and `analysis/radial_gain.py` measures the
   camera's corner brightening as a polynomial in r²
3. **Colour charts.** `charts/layouts_colour.py` writes the training and validation charts from separate seeds.
   `make_chart.py --radial COEF.npy --geometry GEOMETRY.json` builds them with the corner brightening compensated at
   the output radius each sample lands on
4. **Conversion.** `fujiconv.py batch` converts every chart in every look, five times each, because conversions are
   not deterministic (repeats differ by 0.56 of an 8-bit level on average). `analysis/aggregate_repeats.py` takes a
   trimmed mean of the repeats, and `analysis/check_conversion_tags.py` checks that each output's metadata names the
   look asked for
5. **Reading.** `analysis/read_colour_charts.py` reads every patch, with positions mapped through the geometry fit,
   into one table per look
6. **Consistency.** `analysis/consistency.py` checks that a colour renders the same wherever it sits and whatever
   surrounds it, before anything is fitted
7. **Fitting.** `analysis/fit_lut.py` fits a lookup per look from white-balanced camera RGB to the camera's Adobe RGB
   output, on a 17 × 29 × 29 grid in CIELAB through the camera's forward matrix (`charts/colour_space.py`).
   `analysis/fit_profile.py` fits LightCraft's existing profile form (a matrix, a hue/saturation table, and tone and
   chroma curves) for comparison
8. **Export.** `analysis/export_looks.py` writes the lookups as LightCraft's measured-look asset

`analysis/sensor_proxy.py` makes a small camera-RGB image of any raw, framed like the camera's conversion of it through
the raw's own lens tables. It is how fitted looks were checked against the camera's conversions of real photographs,
which are not part of this repository.

Each script's docstring gives its arguments and what it writes. `python analysis/colour_metrics.py` checks the ΔE2000
code against the 34 test pairs Sharma, Wu and Dalal published.

## Conversion settings

The conversion profile is a list of numbered fields. The meanings follow FilmKit's documentation, and the ones marked
confirmed were checked against the camera's output. The fixed settings in `fujiconv.py` (`FIXED`) put everything other
than the film simulation at its neutral value.

| Field | Fixed value | Meaning | Confirmed |
|---:|---:|---|---|
| 3 | as the raw has it | Aspect ratio: 1, 4 and 7 give 3:2, 2, 5 and 8 give 16:9, 3 and 6 give 1:1 | Yes, by output dimensions |
| 6 | 100 | Dynamic range, as a percentage | No |
| 8 | the look | Film simulation | Yes, by output metadata, for the X-T4's nine colour looks |
| 9, 10, 11, 25 | 1 | Grain, Color Chrome, smooth skin and Color Chrome FX Blue, 1 meaning off | No |
| 12 to 19 | 0 | White balance (0 as shot), its shifts and temperature, highlight, shadow, colour and sharpness | No |
| 20 | 8192 | Noise reduction, 0x2000 meaning 0 | No |
| 22 | 1 or 2 | Colour space: 1 sRGB, 2 Adobe RGB | Yes, by the outputs' interoperability tags |
| 27 | 0 | Clarity | No |

## Consistency thresholds

Figures are 8-bit sRGB output levels from the trimmed mean of five conversions, read from the central 35% of each
patch's side. A difference is the largest of the three channel differences.

| Test | What it compares | Passes when |
|---|---|---|
| Anchors across charts | Each anchor cell in every chart, against that cell's median over the charts | 95th percentile at most 1.0 level |
| Anchors across positions | Each anchor colour's eight cells, against that colour's median, per chart | 95th percentile at most 1.5 levels |
| Training duplicates | The two copies of each grid colour, in different charts, cells and neighbours | Median at most 0.6 level, 95th percentile at most 2.0 |
| Validation repeats | Each repeated colour's eight cells, against that colour's median | 95th percentile at most 1.5 levels |
| Reordered against training | Each reordered colour against the mean of its training copies | Median at most 0.6 level, 95th percentile at most 2.0 |

Anchors across charts compare the same cell, so the limit is the tightest, and a failure there would mean the camera
adapts its rendering to a chart's content. Positions and duplicates carry two cells' compensation errors, so their
limits allow about twice the corner compensation's own spread.

## Adapting to another body

The constants that are specific to the X-T4:

- `ACTIVE`, the raw's active area, in `charts/layouts_colour.py`, `charts/layouts_pilot.py`, `charts/layouts_tone.py`
  and `analysis/sensor_proxy.py`
- `WHITE`, the white level, in `charts/make_chart.py` and `analysis/sensor_proxy.py`
- The camera's D65 forward matrix in `charts/colour_space.py`
- The film simulation codes in `camera/fujiconv.py` (`LOOKS`)
- The model, group and look names in `analysis/export_looks.py`

The geometry and corner brightening are measured per body and lens, so steps 1 and 2 must be repeated, not reused.

## Provenance

- `camera/fujiconv.py` is written from the protocol facts FilmKit documents (MIT licence), the operation and property
  codes, their order, the upload header and the profile's field positions, and from the PTP standard (ISO 15740). No
  FilmKit or rawji code is copied
- Raws are decoded only by LightCraft's own decoder. No other raw decoder's source was consulted
- `analysis/testdata/ciede2000_sharma2005.txt` holds the test pairs from G. Sharma, W. Wu and E. N. Dalal, "The
  CIEDE2000 color-difference formula: implementation notes, supplementary test data, and mathematical observations",
  Color Research and Application 30 (2005)
- The tools were written with Claude Code

## Licence

MIT, as in `LICENSE`, to match FilmKit's licence. The conversion driver works the camera over USB in a mode its maker
provides for its own software, and comes with no warranty.
