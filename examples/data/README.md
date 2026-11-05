# South Napa example waveform

`napa_2014_NC.C032.01.HNE.npz` contains the response-corrected acceleration
record used for the South Napa panels in Figure 2. The notebook computes
all transforms and peak measurements afresh; no IMFs or analysis tables
are bundled.

## Source and attribution

- Network: [USGS Northern California Seismic Network (NC)](https://doi.org/10.7914/SN/NC).
- Data access: [Northern California Earthquake Data Center (NCEDC)](https://doi.org/10.7932/NCEDC),
  UC Berkeley Seismological Laboratory. See the archive's
  [citation guidance](https://www.ncedc.org/aws-public-dataset.html).
- Event: 2014 South Napa earthquake.
- Channel: `NC.C032.01.HNE` (network.station.location.channel).
- Sampling rate: 200 Hz; no decimation.
- Recorded interval: 2014-08-24 10:19:49.450000 to 10:24:44.000000 UTC,
  inclusive; 58,911 samples.
- Time origin used in the paper: 2014-08-24 10:20:44 UTC. Sample times
  relative to this origin start at -54.55 s.
- Original request: origin minus 60 s to origin plus 240 s, using the
  NCEDC FDSN service. The triggered record starts after the request start.

Please cite the network and archive when reusing the waveform. This is a
processed derivative of their observational data; the repository's software
license is documented separately in `LICENSE`.

## Contents

Load with `numpy.load(..., allow_pickle=False)`:

| Key | Meaning |
| --- | --- |
| `acceleration` | Full processed record, float64, m/s² |
| `fs` | Sampling rate in Hz |
| `trace_id` | `NC.C032.01.HNE` |
| `starttime_utc` | First sample's UTC time |
| `event_origin_utc` | Paper's reference origin in UTC |
| `start_relative_to_origin_s` | First sample's time relative to that origin |

The complete processed record is retained because filtering and the
full-record CWT peak reading depend on samples beyond the plotted window.
The displayed waveform runs from -5 to +40 s. The CEEMDAN input is the
separate 40 s window centered on PGA (20 s each side, upper endpoint
excluded), at the original sampling rate.

## Processing

Processing was applied to the full raw record before selecting display
or decomposition windows:

1. Merge raw segments with ObsPy `merge(method=1, fill_value="interpolate")`.
2. Remove the mean, then a linear trend.
3. Apply a Hann taper with `max_percentage=0.05` (each end).
4. Remove the complete instrument response using StationXML and
   `output="ACC"`, `pre_filt=(0.025, 0.05, 80.0, 100.0)` Hz,
   `water_level=None`. Preserve the ObsPy response-removal defaults
   `zero_mean=True`, `taper=True`, `taper_fraction=0.05`: these perform an
   additional mean removal and cosine taper inside response removal.

No further filtering or resampling is applied to this acceleration
example. The optional notebook section implements this chain explicitly
and compares its result with the bundled samples without overwriting them.
It downloads from the [EarthScope FDSN service](https://service.earthscope.org),
then trims to the original full processing interval before detrending.
As checked on 2026-11-05, EarthScope supplies raw counts identical to the
source snapshot; the NCEDC endpoint starts 18.44 s later and cannot supply
the full original coverage. This is why the demo uses a fixed bundled
waveform. Later archive or instrument-metadata revisions can change a
fresh download.

The optional EarthScope download and response-removal recipe reproduced
every bundled acceleration sample exactly on 2026-11-05.

## Provenance checks

The exported NumPy samples were checked for exact equality with the paper's
processed miniSEED samples. Reprocessing the retained raw counts with the
retained StationXML also reproduced those samples exactly in the study
environment: Python 3.11.13, ObsPy 1.5.1, NumPy 1.26.4, SciPy 1.17.0.

SHA-256 checksums of the bundled file and original source snapshots:

```text
e930c625021a6b2eaa950454ab0e8509a12488f00f5636fcff319aabef07e561  napa_2014_NC.C032.01.HNE.npz
382836f153e1536cce376d3bf3d2dab7bbe59303c239254b4e1f1b94e1e53d60  original raw NC.C032.01.HNE.mseed
9acbc8afda501fd478be2ab9742d165fe72c7ca268be60db9e4efadd885d85da  original response NC.C032.xml
```
