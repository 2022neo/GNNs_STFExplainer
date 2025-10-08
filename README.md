# STFExplainer: Stratified GNN Explanations through Sufficient Expansion
<div id="top" align="center">
<p align="center">
<img src="figures/STFExplainer.png" width="1000px" >
</p>
</div>

This is the official implementation of [STFExplainer](https://underline.io/lecture/93149-stratified-gnn-explanations-through-sufficient-expansion), accepted by AAAI'24.

> 📝 **Paper**: https://ojs.aaai.org/index.php/AAAI/article/view/29180<br/>
> ✒️ **Authors**: Yuwen Ji, Lei Shi, Zhimeng Liu, Ge Wang

## Installation
Set up the Python environment according to requirements.txt. Python 3.8 or above is required.

## Dataset
We provide the synthetic dataset, namely BAHierarchicalShapes2p, along with experimental code from data processing to quantitative analysis. 
Other datasets can be obtained from their official homepages.

## Experiments
### running quantitative experiment for baseline explainers
```
sh run_baselines.sh
```
Edit `run_baselines.sh` to reproduce our example baselines (gnnexplainer and pgexplainer).
The results will be included in file `baseline_results.txt`.
More baselines can be found in their official codebases.


### running quantitative experiment for our STFExplainer
```
sh run_stfexplainer.sh
```
The results will be included in file `/STFExplainer/crossv/`



# Reference
```
@inproceedings{ji2024stratified,
  title={Stratified GNN Explanations through Sufficient Expansion},
  author={Ji, Yuwen and Shi, Lei and Liu, Zhimeng and Wang, Ge},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={11},
  pages={12839--12847},
  year={2024}
}
```
