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
| terpene set, validated by the isoprene rule | `python terpenes.py` | prints anything rejected |
| dIpc for terpenes and the general set | `python analysis.py` | writes `data.json` |
| statistics for Finding 3 | `python stats.py` | Levene, Fligner, Mann–Whitney |
| Figure 1 (Ipc defect) | `python fig_ipc.py` | |
| Figure 2 (Z_max) | `python fig_zmax.py` | |
| Figure 3 (terpenes) | `python fig_terp.py` | |

`Z_max` itself comes from `../rdkit-fix/zmax/` — see that directory's README for the `geng`
pipeline, the local search (`search.c`) and the transfer-matrix construction (`chain2.py`).

## Every number in the paper traces to one of these

Nothing in the report was typed from memory. Where a figure disagreed with the text during
drafting, the text was corrected — the block-and-link model was described as an upper bound until
the search beat it at n=23 and n=24, and that correction is recorded in the Caveats.
