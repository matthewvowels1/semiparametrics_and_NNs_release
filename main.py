
import argparse
from train_test import TrainTest


def main(args):

    TT = TrainTest(args)
    true_psi, df = TT.train_test()

    print('Finished!')
    return true_psi, df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IF_experiments")
    parser.add_argument("--run", default='RUN2', type=str)  # run name (for filenames)
    parser.add_argument("--N", default=5000, type=int)  # dataset size (ignored if dataset==IHDP)
    parser.add_argument("--starting_iter", default=50, type=int)  # which seed to start at (for reproducibility purposes)
    parser.add_argument("--num_tuning_trials", default=2, type=int)  # number of hyperparameter searches for the neural networks
    parser.add_argument("--num_runs", default=49, type=int)  # number of simulations
    parser.add_argument("--data_rand", default=1, type=int)  # if set to 1, then each simulation will use a different draw from the dataset (default), otherwise, only uses the first simulation
    parser.add_argument("--super_learner_k", default=10, type=int)  # number of cross-validation folds for the superlearner training
    parser.add_argument("--run_SL", default=0, type=int)  # run superlearner
    parser.add_argument("--run_LR", default=0, type=int)  # run linear/logistic regression (depends on the dataset)
    parser.add_argument("--run_LR_SL", default=0, type=int)  # run logistic regression with SL as propensity model
    parser.add_argument("--run_NN", default=1, type=int)  # run neural network method (more base learner options below)
    parser.add_argument("--run_treg", default=1, type=int)  # run neural network + targeted regularization
    parser.add_argument("--run_NN_SL", default=1, type=int)  # run neural network + SL propensity score model combination
    parser.add_argument("--run_treg_SL", default=1, type=int)  # run neural network + targeted regularization with SL propensity score model
    parser.add_argument("--run_NN_or_multinet", default=1, type=int)   # choose whether all NN runs are with multinet (=1) or not (=0)
    parser.add_argument("--data_masking", default=0, type=int)  # N.B. data masking only works with multinet (splits data over layers)
    parser.add_argument("--layerwise_optim", default=0, type=int)  # N.B. layerwise optim only works with multinet (assigns a separate optim for each layer)
    parser.add_argument("--calibration", default=0, type=int)   # N.B. calibration does not work with multinet
    parser.add_argument("--SL_output", default='cls', type=str)  # or 'reg'. NB if dataset = IHDP then this will be forcably set to 'reg'
    parser.add_argument("--dataset", default='synth1', type=str)  # 'IHDP', or 'synth1', or 'synth2'
    parser.add_argument("--gpu", default=0, type=int)  # whether to use GPU
    args = parser.parse_args()

    path = "./IHDP/"
    true_psi, df = main(args)

