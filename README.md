confint
=======

This repository contains example code and data for computing confidence intervals for IR experiments.  The experiments
are described in the paper,

> Soboroff, Ian. "Computing confidence intervals for common IR measures." In the proceedings of the 2014 Workshop on Evaluation
> of Information Access [(EVIA 2014)](http://research.nii.ac.jp/ntcir/evia2014/), December, 2014, Tokyo, Japan.

All code and data used in the paper is in this repo.

The main script is trec_eval_ci.py, and it runs on trec_eval output files, so you will need evaluation output in 
trec_eval format for the measures handled in the script.  The variant scripts ntcir7_eval_ci.py and ntcir10_eval_ci.py
demonstrate how to easily mod the script to work on different measures and file formats.

