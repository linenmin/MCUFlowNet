# Modifications to the EdgeFlowNet baseline

This directory bundles a **modified copy** of EdgeFlowNet
([pearwpi/EdgeFlowNet](https://github.com/pearwpi/EdgeFlowNet), MIT, vendored at upstream
commit `15b30c5`). Per the MIT License, the original copyright notice and license text are
preserved (`LICENSE`), and this file records what we changed.

## Files we modified, relative to upstream `15b30c5`

| File | Nature of change |
|------|------------------|
| `.gitignore` | adapted ignore rules for our workflow |
| `code/dataset_paths/FC2_dirnames.txt` | dataset path list adapted to our layout |
| `code/dataset_paths/MPI_Sintel_Clean-Final-Mixed_train_list.txt` | dataset path list |
| `code/dataset_paths/MPI_Sintel_Final_train_list.txt` | dataset path list |
| `code/dataset_paths/MPI_Sintel_train_clean.txt` | dataset path list |
| `code/misc/BatchCreationTF.py` | data pipeline adjustments for our experiments |
| `code/misc/DataHandling.py` | data handling adjustments |
| `code/misc/Losses.py` | loss adjustments used for our baseline runs |
| `code/misc/TensorBoardDisplay.py` | logging adjustments |
| `code/network/MultiScaleResNet.py` | baseline network tweaks for our comparison protocol |
| `code/train.py` | training-loop adjustments |
| `wrappers/run_test.py` | evaluation entry-point adjustments |
| `wrappers/run_train.py` | training entry-point adjustments |

Everything else under `EdgeFlowNet/` that is **not** in the table above is either (a) verbatim
from upstream, or (b) our own additions — most notably:

- `sramTest/` — our hardware-profiling toolkit (Vela/SRAM benchmarks) and the thesis
  figure-generation scripts (`make_thesis_ch*.py`), entirely our own work.

To see the exact line-level differences, diff this directory against upstream `15b30c5`:

```bash
git clone https://github.com/pearwpi/EdgeFlowNet.git /tmp/edgeflownet-upstream
cd /tmp/edgeflownet-upstream && git checkout 15b30c5
diff -r /tmp/edgeflownet-upstream <this-repo>/EdgeFlowNet
```
