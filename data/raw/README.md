# Raw evidence

Per source `SOURCES.json` pins every original input (URL, version, SHA-256, licence, raw policy). Original files are
committed under `files/` only when redistribution is permitted and they are small (Paderborn LEA GPL-3.0, Zenodo
CC-BY-4.0); MAS and Princeton inputs are retrieved on demand by `aipe-magnetics-db refresh --allow-download`.
Never modify these files: importers verify them by SHA-256.
