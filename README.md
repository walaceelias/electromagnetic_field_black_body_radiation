# Blackbody Radiation in Spherical Cavities: Thermodynamics of an Electromagnetic Field

This repository contains the Python code used to generate the figures for the article:
**"Blackbody Radiation in Spherical CavitieS: Thermodynamics of an Electromagnetic Field"**.

## Prerequisites

Python 3 and: `numpy`, `scipy`, `matplotlib`, `pandas`.

```bash
pip install -r requirements.txt
python generates_figures_em.py
```

Figures are written to `Figures_EM/`. A cached mode table (`df_max550.pkl`) is created
on first run and reused on subsequent runs to avoid recomputing the TE/TM root search
(which scales as O(x_cut^2)).
