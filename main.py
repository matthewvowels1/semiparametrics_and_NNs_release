
import argparse
from train_test import TrainTest


def main(args):
    TT = TrainTest(args)
    true_psi, df = TT.train_test()

    return true_psi, df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IF_experiments")
    parser.add_argument("--run", default='RUN2', type=str)  # the name of the run (will determine output filename prefix
    parser.add_argument("--N", default=5000, type=int)  # sample size (ignored if dataset=IHDP)
    parser.add_argument("--starting_iter", default=50, type=int)  # starting seed (useful if run fails)
    parser.add_argument("--num_tuning_trials", default=15, type=int)  # number of hyperparam trials
    parser.add_argument("--num_runs", default=49, type=int)  # number of simulations (for IHDP this needs to be < 100)
    parser.add_argument("--data_rand", default=1, type=int)  # whether to use different data draws for each simulation, or just the first
    parser.add_argument("--super_learner_k", default=10, type=int)  # number of folds in k-fold SL fit algo
    parser.add_argument("--run_SL", default=0, type=int)   # fit SL model on its own (not required for NN_SL combinations)
    parser.add_argument("--run_LR", default=0, type=int)  # fit logistic or linear regression
    parser.add_argument("--run_NN", default=1, type=int)  # set to 1 if EITHER CFR or multinet are required
    parser.add_argument("--run_treg", default=1, type=int)  # use targeted regularization (if training NN/multinet)
    parser.add_argument("--run_NN_SL", default=1, type=int)  #  NN/multinet w/ SL propensity model
    parser.add_argument("--run_treg_SL", default=1, type=int)  #  NN/multinet w/ SL propensity model and targeted regularisation
    parser.add_argument("--run_NN_or_multinet", default=1, type=int)   # choose whether all NN runs are with multinet (=1) or not (=0)
    parser.add_argument("--data_masking", default=0, type=int)  # N.B. data masking only works with multinet (splits data over layers)
    parser.add_argument("--layerwise_optim", default=0, type=int)  # N.B. layerwise optim only works with multinet (assigns a separate optim for each layer)
    parser.add_argument("--calibration", default=0, type=int)   # N.B. calibration does not work with multinet
    parser.add_argument("--SL_output", default='cls', type=str)  # or 'reg'. NB if dataset = IHDP then this will be forcably set to 'reg'
    parser.add_argument("--dataset", default='synth1', type=str)  # 'IHDP', or 'synth1', or 'synth2' or 'synth3'
    parser.add_argument("--gpu", default=0, type=int)  # whether to use GPU  (=1) or CPU (=0)

    args = parser.parse_args()

    path = "./IHDP/"
    true_psi, df = main(args)

