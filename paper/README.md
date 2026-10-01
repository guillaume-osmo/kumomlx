# Paper

**[report.pdf](report.pdf)** — *Ipc is the Hosoya index, computed wrongly: a bounded
reformulation of molecular information content, and what it measures in terpene space.*

Source is `report.md`; render with:

```bash
pandoc report.md -o report.pdf --pdf-engine=xelatex
```

## Regenerating everything

| step | command | note |
|---|---|---|
| COCONUT sampling + dIpc | `python coconut.py 6000` | needs the Zenodo dump, see below |
| COCONUT statistics (Finding 3) | `python coconut_stats.py` | the published numbers |
| Figure 1 (Ipc defect) | `python fig_ipc.py` | |
| Figure 2 (Z_max) | `python fig_zmax.py` | |
| Figure 3 (terpenes) | `python fig_terp.py` | |

`Z_max` itself comes from `../rdkit-fix/zmax/` — see that directory's README for the `geng`
pipeline, the local search (`search.c`) and the transfer-matrix construction (`chain2.py`).

## COCONUT data

Finding 3 uses COCONUT 2022.01.01 with NPClassifier labels, from
[10.5281/zenodo.10629838](https://doi.org/10.5281/zenodo.10629838) (11 MB zipped):

```bash
curl -L -o coconut_pred.zip \
  https://zenodo.org/api/records/10629838/files/coconut_predictions.zip/content
unzip coconut_pred.zip -d /tmp
python coconut.py 6000        # ~50 s: samples, computes dIpc, size-matches controls
python coconut_stats.py
```

`terpenes.py`, `analysis.py` and `stats.py` are the exploratory scripts that preceded the
COCONUT analysis. They are kept because they run, but the reported results come from
`coconut.py` and `coconut_stats.py`.
