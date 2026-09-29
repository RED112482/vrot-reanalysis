# Vrot Reanalysis — White Lake pilot

This repository analyzes only the first tornado in the supplied master CSV:
White Lake, March 9, 2011, 13:20–13:23 UTC. It retrieves KMOB Level-II
velocity, evaluates each available lowest tilt during the reported tornado
interval, and generates a per-scan measurement CSV and zoomed six-panel PNG.

Each black-background image shows lowest-tilt velocity, reflectivity, spectrum
width, and correlation coefficient (when archived). The bottom row shows the
strongest qualifying upper-tilt velocity couplet and matching reflectivity.
Velocity uses the supplied palette; all fields use filled native radar gates.
The yellow line shows the surveyed track through that scan time.

The 2 nmi circle is centered on the interpolated tornado position. Each lobe
must contain at least three connected gates with similar-sign neighbors. The
cyan and orange dots mark weighted centers of the strongest quartile within
each coherent lobe. The median velocities of those cores give
`Vrot = (Vout - Vin) / 2`. The pair must be 0.15–1.5 nmi apart, mostly across
the radar beam at similar range, and have a midpoint within 1.25 nmi of the
estimated tornado position. The CSV includes the centroid coordinates, gate
counts, diameter, upper-tilt measurement, and unfiltered raw Vrot for review.
These are exploratory thresholds, not validated tornado detections.

## Run on GitHub

1. Open the **Actions** tab and choose **White Lake Vrot pilot**.
2. Click **Run workflow**, leave `main` selected, then click the green
   **Run workflow** button.
3. Open the new run. Once the **Analyze tornado and render velocity scans** job
   finishes, download **White-Lake-velocity-results** under **Artifacts**.
4. Unzip it to see `measurements.csv` and the scan PNGs.

The first run installs the radar environment and downloads historical volumes.
Later runs can reuse cached environment and radar files. The action does not
publish its results as a website or commit radar output to this repository.

## Run locally

```bash
conda env create -f environment.yml
conda activate white-lake-vrot
python six_panel_vrot.py
```

Use `--radar-files PATH` to read previously downloaded Level-II files instead
of the public archive. Output goes to `White_Lake_2011-03-09/`.

**Research caveat:** Vrot values are preliminary candidates. Review velocity
folding, the assumed linear tornado motion between surveyed endpoints, radar
geometry, and the chosen pair. A scan without a coherent pair retains a blank
Vrot and `NO_PAIR` status. The upper tilt is a rotation candidate, not proof of
a mesocyclone. The 2011 case may lack CC because dual-pol data may not exist
for that volume.
