#! /bin/bash

EXPERIMENTS_ROOT=$HOME/DataspellProjects/bnn/experiments
FINAL_EXPERIMENTS="complexity_comparison sequential forgetting forgetting_beta_sweep hmc_baseline_for_forgetting crossec_loglik"

scp $EXPERIMENTS_ROOT/src/{data,experiment,factory,model}.py whistlingswan:teaching-home/experiments/src/

for experiment in $FINAL_EXPERIMENTS; do
    scp $EXPERIMENTS_ROOT/coherent_inference/$experiment.py whistlingswan:teaching-home/experiments/$experiment/
done
