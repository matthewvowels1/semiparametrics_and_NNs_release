# Semiparametrics and NNs (initial release)
Release code for experiments on influence functions with neural networks


Required Libraries and Packages:

python=3.7

pytorch=1.9.0

scikit-learn=0.24.2

scipy=1.6.2

statsmodels=0.12.2

numpy=1.20.3

pandas=1.3.0


### LF (v1), CFR, LR, SL, variants
```python
python3 main.py --run RUN1 --N 5000 --starting_iter 0 --num_tuning_trials 50 --num_runs 100 --data_rand 1 --super_learner_k 10 --run_SL 1 --run_treg 1 --run_LR 1 --run_NN 1 --run_NN_SL 1 --run_treg_SL 1 --run_NN_or_multinet 0 --data_masking 0 --layerwise_optim 0 --calibration 0  --dataset synth1
```

### LF (v1), MultiNet, variants
```python
python3 main.py --run RUN2 --N 5000 --starting_iter 0 --num_tuning_trials 50 --num_runs 100 --data_rand 1 --super_learner_k 10 --run_SL 1 --run_treg 1 --run_LR 1 --run_NN 1 --run_NN_SL 1 --run_treg_SL 1 --run_NN_or_multinet 1 --data_masking 0 --layerwise_optim 0 --calibration 0 --dataset synth1
```


### LF (v1), MultiNet + data masking, variants
```python
python3 main.py --run RUN3 --N 5000 --starting_iter 0 --num_tuning_trials 50 --num_runs 100 --data_rand 1 --super_learner_k 10 --run_SL 1 --run_treg 1 --run_LR 1 --run_NN 1 --run_NN_SL 1 --run_treg_SL 1 --run_NN_or_multinet 1 --data_masking 1 --layerwise_optim 0 --calibration 0  --dataset synth1
```


### LF (v1), MultiNet + data masking + layerwise training, variants
```python
python3 main.py --run RUN4 --N 5000 --starting_iter 0 --num_tuning_trials 50 --num_runs 100 --data_rand 1 --super_learner_k 10 --run_SL 1 --run_treg 1 --run_LR 1 --run_NN 1 --run_NN_SL 1 --run_treg_SL 1 --run_NN_or_multinet 1 --data_masking 1 --layerwise_optim 1 --calibration 0  --dataset synth1
```

Change the dataset with the ```--dataset``` flag, set to ```synth1``` (LF v1), ```synth2``` (LF v2), ```synth3``` (LF v3), or ```IHDP```. For IHDP, sample size flag is ignored.

If GPU support is available you can add the ```--gpu 1``` flag, although we have found that owing to the high I/O speed in this script it is not necessarily faster that CPU.