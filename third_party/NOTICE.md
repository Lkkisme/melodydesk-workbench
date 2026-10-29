# Included research implementations

- ROSVOT: https://github.com/RickyL-2000/ROSVOT, retrieved 2026-09-26 from main. Original MIT license and README retained under rosvot/. Included modules/ and utils/. Local change: RMVPE checkpoint loading uses weights_only=True. CPU inference adapter is in singing_model.py. Official README checkpoint download: Google Drive ID 1JNtNT37KiLq9uFQqHk7JFs-3trxd3bRh; archive SHA256 b6055e81315b93415c9bd7fc48e10a28a3da1bea960cab7385483bd7443ba852.
- PANNs: https://github.com/qiuqiangkong/audioset_tagging_cnn, retrieved 2026-09-26. Original MIT license in panns/LICENSE.MIT. cnn6.py contains the inference architecture extracted from upstream models.py; training mixup omitted. AudioSet label table preserved. CNN6 checkpoint from https://zenodo.org/records/3987831, MD5 e25e26b84585b14c7754c91e48efc9be.

Checkpoint downloads are stored separately under work/model-cache. Repository code licenses do not establish all training-data or checkpoint rights; their commercial distribution terms have not been independently audited here.
