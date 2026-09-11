# GB-QSensing
This is the source code for the experiments conducted in the paper "Bayesian quantum sensing using graybox machine learning" available at https://arxiv.org/abs/2601.17465. If you like to use any of this code, please cite our paper.

The repository includes the raw experimental dataset *"raw_dataset.csv"* as well as the post-processed one *"Dataset_April_2025_sorted_90_10"* (pickle format). The latter can be generated from the former by calling the python function *"create_dataset"* that is in the *"sensing_functions.py"* module. The whitebox, graybox, and blackbox trained parameters are also available under the names: *"model_WB_exp_14_07_2025"*, *"model_GB_exp_11_08_2026"*, and *"model_BB_small_exp_12_08_2026"*. These can be loaded as shown in the example jupyter notebook.

In order to run the whole workflow, follow these steps:
1) Run *"python3 train_model.py GB"*. This will train the GB model. To train the BB small/larger models use either *"python3 train_model.py BB_small"* or *"python3 train_model.py BB_large"*
2) Run the notebook *"Sensing_Example_Full.ipynb"*, this will run the Bayesian procedure and generate all figures.
