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

Current limitation:

- FIW face image folders are not bundled here because they were not present in the original workspace.
- As a result, `family-deep --dataset-name fiw ...` has the metadata it needs, but still requires the actual FIW image files if you want to run that path end-to-end.
