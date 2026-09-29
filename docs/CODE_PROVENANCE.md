# Code provenance

## Current release

The current release contains the IPTC inference overlay, its PUT integration
point, aggregate evaluation records, and source hashes.

| Component | Location | Provenance |
|---|---|---|
| IPTC inference overlay | `iptc/` | This repository |
| PUT integration | `third_party/PUT/` | [liuqk3/PUT](https://github.com/liuqk3/PUT) |
| Matching reference | `third_party/licenses/ToMe-LICENSE.txt` | [facebookresearch/ToMe](https://github.com/facebookresearch/ToMe) |

The extracted source files used by the overlay are recorded in
`iptc/source_hashes.json`. The verification script checks those hashes and the
local module lifecycle without loading a checkpoint.

## Historical directories

Older SBVC launchers, evaluation tools, and Latent Codes integrations remain
in the repository for provenance and compatibility. They are not part of the
current IPTC result path. Their upstream notices and license files are kept in
their original locations.

## Public assets

`assets/iptc_main_figure.png` is the current IPTC framework preview shown in
the README. Other legacy figures under `assets/` are retained as historical
visualization assets and should not be interpreted as additional IPTC
benchmark records.

