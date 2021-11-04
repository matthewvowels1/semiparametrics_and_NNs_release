

'''
***Closely based on code for TEDVAE by Weijia Zhang***
https://github.com/WeijiaZhang24/TEDVAE
See also the paper:
Zhang, W., Liu, L., and Li, J. (2020) Treatment effect estimation with disentangled
latent factors. https://arxiv.org/pdf/2001.10652.pdf
'''
import argparse
from train_test import TrainTest


def main(args):
    TT = TrainTest(args)
    true_psi, df = TT.train_test()

    return true_psi, df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IF_experiments")
    parser.add_argument("--run", default='RUN2', type=str)
    parser.add_argument("--N", default=5000, type=int)
    parser.add_argument("--starting_iter", default=50, type=int)
    parser.add_argument("--num_tuning_trials", default=2, type=int)
    parser.add_argument("--num_runs", default=49, type=int)
    parser.add_argument("--data_rand", default=1, type=int)
    parser.add_argument("--super_learner_k", default=10, type=int)
    parser.add_argument("--run_SL", default=0, type=int)
    parser.add_argument("--run_treg", default=1, type=int)
    parser.add_argument("--run_LR", default=0, type=int)
    parser.add_argument("--run_NN", default=1, type=int)
    parser.add_argument("--run_NN_SL", default=1, type=int)
    parser.add_argument("--run_treg_SL", default=1, type=int)
    parser.add_argument("--run_NN_or_multinet", default=1, type=int)   # choose whether all NN runs are with multinet (=1) or not (=0)
    parser.add_argument("--data_masking", default=0, type=int)  # N.B. data masking only works with multinet (splits data over layers)
    parser.add_argument("--layerwise_optim", default=0, type=int)  # N.B. layerwise optim only works with multinet (assigns a separate optim for each layer)
    parser.add_argument("--calibration", default=0, type=int)   # N.B. calibration does not work with multinet
    parser.add_argument("--SL_output", default='cls', type=str)
    parser.add_argument("--dataset", default='synth1', type=str)  # 'IHDP', or 'synth1', or 'synth2' or 'synth3'
    parser.add_argument("--gpu", default=0, type=int)
    args = parser.parse_args()

    path = "./IHDP/"
    true_psi, df = main(args)

