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
- `data/mydataset`
  - local private in-house kinship collection with age-variant families, named-family subsets, and identical-twin material

Notes:

- The maintained `family-deep` pipeline now supports the repo-local `data/FIDs/FIDs` layout directly.
- The large FIW assets under `data/FIDs` are intentionally git-ignored so the GitHub repository stays lightweight.
- The private `data/mydataset` collection is also intentionally git-ignored.
- The toolkit now includes a native adapter that can summarize `data/mydataset` and export image-level and pair-level manifests for downstream experiments.
- Some FIW metadata rows may reference face crops that are not present in a given local export. The loader resolves alternate crops for the same PID when possible and skips unresolved pairs automatically.
