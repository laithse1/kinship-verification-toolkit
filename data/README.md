# Data Layout

This repository includes the runtime data needed by the maintained toolkit under:

- `data/kinface`
  - `KinFaceW-I`
  - `KinFaceW-II`
  - `traindata`
  - `testdata`
- `data/kinver`
  - `data-KinFaceW-I`
  - `data-KinFaceW-II`
- `data/family/data`
  - FIW pair metadata CSV files used by the native `family-deep` pipeline
- `data/FIDs`
  - local FIW FIDs image bundle
  - optional FIW feature exports and supporting FIW CSV metadata

Notes:

- The maintained `family-deep` pipeline now supports the repo-local `data/FIDs/FIDs` layout directly.
- The large FIW assets under `data/FIDs` are intentionally git-ignored so the GitHub repository stays lightweight.
- Some FIW metadata rows may reference face crops that are not present in a given local export. The loader resolves alternate crops for the same PID when possible and skips unresolved pairs automatically.
