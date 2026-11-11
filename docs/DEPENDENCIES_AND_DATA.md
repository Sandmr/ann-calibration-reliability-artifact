# Dependencies and data provenance

## Environment

- Python 3.11 virtual environment: `.venv`
- Locked requirements: `requirements.txt`
- Faiss CPU 1.8.0 is the validated version on this Windows runtime.
- Faiss 1.16.0 was tested but terminated the process during HNSW construction; it is not used for reported results.
- No deep-learning package or model weight was installed.

## Dataset

- SIFT1M HDF5: `https://ann-benchmarks.com/sift-128-euclidean.hdf5`
- Downloaded file: `data/sift-128-euclidean.hdf5`
- SHA-256: `dd6f0a6ed6b7ebb8934680f861a33ed01ff33991eaee4fd60914d854a0ca5984`
- GloVe25 HDF5: `https://ann-benchmarks.com/glove-25-angular.hdf5`
- Downloaded file: `data/glove-25-angular.hdf5`
- SHA-256: `51004cb0ae962159f0db507a51fec2b395de14b166f55976c89f16bd2f8b6391`
