# Vrot Reanalysis — White Lake pilot

This repository analyzes only the first tornado in the supplied master CSV:
White Lake, March 9, 2011, 13:20–13:23 UTC. It retrieves KMOB Level-II
velocity, evaluates each available lowest tilt during the reported tornado
interval, and generates a per-scan measurement CSV and zoomed PNG panels.

The panels show filled native radar gates on black using the supplied velocity
palette, a 2 nmi analysis circle, selected inbound/outbound gates, and the
surveyed track through each scan time.

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
python white_lake_vrot.py
```

Use `--radar-files PATH` to read previously downloaded Level-II files instead
of the public archive. Output goes to `White_Lake_2011-03-09/`.

**Research caveat:** Vrot values are preliminary candidates. Review velocity
folding, the assumed linear tornado motion between surveyed endpoints, radar
geometry, and the chosen gate pair. A scan without a coherent pair retains a
blank Vrot and `NO_PAIR` status.
