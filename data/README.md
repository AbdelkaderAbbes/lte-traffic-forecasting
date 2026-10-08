# Data

The data are **not** committed (licence allows redistribution, but the file is ~50 MB zipped and should be fetched from the source).

- Record: <https://doi.org/10.5281/zenodo.17815388> - *Performance Management Counters from Live 5G, 4G and 2G Radio Access Network*
- Licence: CC BY 4.0. Paper: Lehoczky, Turcsany, Krajcovicova, Zatroch, Kajan, Galinski, *Scientific Data* (2026), <https://doi.org/10.1038/s41597-026-07723-0>
- File used here: `Dataset_02.zip` (LTE 1800 MHz), md5 `7d9c1c019e5d18bf318b4e4ad9393d91` (from the Zenodo record page)

```bash
python scripts/download_data.py      # download + checksum + extract into data/
# or download manually and place the CSV at data/Dataset_02_LTE_1800.csv
```

Expected columns (19): `Base station, Sector, Timestamp, Radio unit energy consumption, Baseband energy consumption,
4G max active users DL/UL, 4G data volume DL/UL, 4G max RRC users, 4G RB utilization, 4G CQI rank 1-4,
4G RRC users, 4G active users UL/DL, 4G MIMO rank DL`.

Unit of `4G data volume DL` and of the energy columns: **take them from the data descriptor** - the code is unit-agnostic
and reports volume with relative metrics (WAPE) and energy as percentages.

Please cite the dataset paper if you use this repository.
