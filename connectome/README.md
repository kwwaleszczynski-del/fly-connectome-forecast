# Reservoir subgraphs derived from the FlyWire connectome

The files in this folder are **derived from**:

> FlyWire Consortium. *FlyWire Whole-brain Connectome Connectivity Data*, version 783. Zenodo, 2024.
> doi:[10.5281/zenodo.10676866](https://doi.org/10.5281/zenodo.10676866). Licence:
> [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

Please also cite the papers describing the data: Dorkenwald et al. 2024, *Neuronal wiring diagram of an adult
brain*, Nature 634:124-138; Schlegel et al. 2024, *Whole-brain annotation and multi-connectome cell typing of
Drosophila*, Nature 634:139-152; Eckstein, Bates et al. 2024, *Neurotransmitter classification from electron
microscopy images at synaptic sites in Drosophila melanogaster*, Cell 187:2574-2594 (transmitter predictions);
Buhmann et al. 2021, Nature Methods 18:771-774 (synapse detection); Heinrich et al. 2018, MICCAI (synaptic clefts).

**Changes made** (by `python -m fly_forecast connectome`, code in `src/fly_forecast/connectome.py`): only the
table `proofread_connections_783.feather` was used. Rows were summed over neuropils per neuron pair; pairs with
fewer than 5 synapses and autapses were removed; each presynaptic neuron was assigned the transmitter with the
highest synapse-weighted mean probability over its outgoing connections; neurons predicted as dopaminergic,
serotonergic or octopaminergic were removed as presynaptic partners; signs were set to +1 for acetylcholine and -1
for GABA and glutamate. The subgraphs keep the N neurons with the most synapses (`topN`), or the 1,000 neurons with
the most synapses inside the right mushroom body neuropils MB_CA_R, MB_PED_R, MB_VL_R, MB_ML_R (`mb_right`, only
synapses inside those neuropils). This is a modelling simplification, not a curated biological dataset.

Files:

| file | content |
|---|---|
| `<name>_nodes.csv` | `node` (row/column index), FlyWire `root_id` (v783), predicted transmitter `nt`, `sign`, `syn_total` (synapses in the thresholded graph), `top_neuropil` |
| `<name>_edges.csv.gz` | `pre`, `post` (node indices), `weight` = sign x synapse count |
| `graph_stats.json` | size of the thresholded whole-brain graph |

The derived files are distributed under the same licence, CC BY 4.0.
