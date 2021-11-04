
import numpy as np
import pandas as pd
from QGNet import Tuner, QNet, GNet, Trainer, T_scaling
from QGMultiNet import MultiNetTuner, GMultiNet, QMultiNet, MultiNetTrainer
from data_gen import generate_data, sigm, inv_sigm, IHDP
import torch
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from super_learner import SuperLearner
from sklearn.svm import SVC
import traceback
from datetime import datetime


now = datetime.now()

current_time = now.strftime("%H:%M:%S")


def one_step(x_, y_, Q0, Q1, G10):
    D0 = ((1 - x_) * (y_ - Q0)) / (1 - G10) + Q0 - Q0.mean()
    D1 = (x_ * (y_ - Q1) / G10) + Q1 - Q1.mean()
    Q1_star = Q1 + D1
    Q0_star = Q0 + D0
    return Q1_star, Q0_star

def submodel(x_, y_, Q1, Q0, Q10, G10):

    H1 = x_ / (G10)
    H0 = (1 - x_) / (1 - G10)

    eps0, eps1 = sm.GLM(y_, np.concatenate([H0, H1], 1), offset=inv_sigm(Q10[:, 0]),
                        family=sm.families.Binomial()).fit().params

    Q0_star_solve = sigm(inv_sigm(Q0) + eps0 / (1 - G10))
    Q1_star_solve = sigm(inv_sigm(Q1) + eps1 / G10)
    return Q1_star_solve, Q0_star_solve


class TrainTest(object):
    def __init__(self, config):
        self.run = config.run
        self.N = config.N
        self.k = config.super_learner_k
        self.num_runs = config.num_runs  # number of simulations with 'optimal'  hyperparams for method eval
        self.data_rand = config.data_rand
        self.num_tuning_trials = config.num_tuning_trials  # number of trials for hyperparameter search
        self.test_iter = 100  # number of "weight updates" between model testing
        self.run_SL = config.run_SL
        self.run_treg = config.run_treg
        self.run_LR = config.run_LR
        self.run_NN = config.run_NN
        self.run_NN_SL = config.run_NN_SL
        self.run_treg_SL = config.run_treg_SL
        self.multinet = config.run_NN_or_multinet
        self.output = config.SL_output
        self.dataset = config.dataset
        self.starting_iter = config.starting_iter
        self.gpu = config.gpu
        self.device = 'cpu'
        if self.gpu == 1:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.fn = ''


        if self.multinet == 1:
            print('Using MultiNet')
            self.calibration = 0
            self.data_masking = config.data_masking
            self.layerwise_optim = config.layerwise_optim
        else:
            self.calibration = config.calibration
            self.data_masking = 0
            self.layerwise_optim = 0
        print('Running: ', self.run)


    def train_test(self):

        # First establish ground truth treatment effect:
        if self.dataset != 'IHDP':
            Z, x, y, Y1, Y0 = generate_data(N=5000000, seed=0, dataset=self.dataset)
        else:
            Z, x, y, Y1, Y0 = IHDP(seed=1)
        true_psi = (Y1 - Y0).mean()

        use_last_model = True  # whether to use the model with truly the best test loss (=False), or the one which the training loop 'breaked' on (=True)
        use_t = True  # whether to use treatment as a predictor in Q network
        test_loss_plot = False

        # set some empty lists
        sample_truth = []
        estimates_naive = []
        estimates_upd_treg = []
        estimates_upd_treg_submod = []
        estimates_upd_1s = []
        estimates_upd_submod = []

        estimates_naive_halfSL = []
        estimates_upd_treg_halfSL = []
        estimates_upd_treg_submod_halfSL = []
        estimates_upd_1s_halfSL = []
        estimates_upd_submod_halfSL = []


        estimates_naive_LR = []
        estimates_upd_1s_LR = []
        estimates_upd_submod_LR = []
        estimates_naive_SL = []
        estimates_upd_1s_SL = []
        estimates_upd_submod_SL = []
        eps_ = []
        betas_g = []
        betas_q = []

        # init seed
        seed = self.starting_iter  # this was set to 100 previously (no reason)
        i = self.starting_iter
        print('Starting from seed', i)
        while (i-self.starting_iter) < self.num_runs:
            seed += 1
            try:
                print('=====================RUN {} of {}==================='.format(i-self.starting_iter, self.num_runs - 1))
                if self.data_rand:
                    if self.dataset != 'IHDP':
                        z_, x_, y_, Y1, Y0 = generate_data(N=self.N, seed=seed, dataset=self.dataset)
                    else:
                        print(seed)
                        z_, x_, y_, Y1, Y0 = IHDP(seed=seed)
                else:
                    if self.dataset != 'IHDP':
                        z_, x_, y_, Y1, Y0 = generate_data(N=self.N, seed=1, dataset=self.dataset)
                    else:
                        z_, x_, y_, Y1, Y0 = IHDP(seed=1)
                # data generation:
                sample_truth.append((Y1 - Y0).mean())
                x = torch.tensor(x_).type(torch.float32).to(self.device)
                z = torch.tensor(z_).type(torch.float32).to(self.device)
                y = torch.tensor(y_).type(torch.float32).to(self.device)

                x_int1 = torch.ones_like(x).to(self.device)  # this is the 'intervention data'
                x_int0 = torch.zeros_like(x).to(self.device)
                x_int1_ = np.ones_like(x_)
                x_int0_ = np.zeros_like(x_)

                if self.run_NN or self.run_treg or self.run_NN_SL:

                    print('==============TUNING==============')
                    # run hyperparameter search
                    print('Tuning G')
                    if self.multinet:
                        gtuner = MultiNetTuner(x=x, y=y, z=z, net_type='G', test_iter=self.test_iter,
                               trials=self.num_tuning_trials, use_beta=False, use_t=use_t, data_masking=self.data_masking,
                               test_loss_plot=test_loss_plot, layerwise_optim=self.layerwise_optim, device=self.device)
                    else:
                        gtuner = Tuner(x=x, y=y, z=z, net_type='G', test_iter=self.test_iter, calibration=0, device=self.device,
                                   trials=self.num_tuning_trials, use_beta=False, use_t=use_t, test_loss_plot=test_loss_plot)

                    gtuning_history, best_g, x_pred, _ = gtuner.tune()

                    gtotal_losses = np.asarray(gtuning_history['best_model_test_loss'])
                    gbest_index = np.argmin(gtotal_losses)

                    gbest_params = {}
                    for key in gtuning_history.keys():
                        gbest_params[key] = gtuning_history[key][gbest_index]

                    print('Tuning Q')
                    if self.multinet:
                        qtuner = MultiNetTuner(x=x, y=y, z=z, x_pred=x_pred, net_type='Q', test_iter=self.test_iter, data_masking=self.data_masking, device=self.device,
                               trials=self.num_tuning_trials, use_beta=False, use_t=use_t, test_loss_plot=test_loss_plot, layerwise_optim=self.layerwise_optim)
                    else:
                        qtuner = Tuner(x=x, y=y, z=z, x_pred=x_pred, net_type='Q', test_iter=self.test_iter, calibration=0, device=self.device,
                                   trials=self.num_tuning_trials, use_beta=False, use_t=use_t, test_loss_plot=test_loss_plot)
                    qtuning_history, best_q, _, eps = qtuner.tune()

                    qtotal_losses = np.asarray(qtuning_history['best_model_test_loss'])
                    qbest_index = np.argmin(qtotal_losses)

                    qbest_params = {}
                    for key in qtuning_history.keys():
                        qbest_params[key] = qtuning_history[key][qbest_index]

                    print('Tuning Q + treg')
                    if self.multinet:
                        qtregtuner = MultiNetTuner(x=x, y=y, z=z, x_pred=x_pred, net_type='Q', test_iter=self.test_iter, data_masking=self.data_masking, device=self.device,
                                       trials=self.num_tuning_trials, use_beta=True, use_t=use_t, test_loss_plot=test_loss_plot, layerwise_optim=self.layerwise_optim)
                    else:
                        qtregtuner = Tuner(x=x, y=y, z=z, x_pred=x_pred, net_type='Q', test_iter=self.test_iter, calibration=0,
                               trials=self.num_tuning_trials, use_beta=True, use_t=use_t, test_loss_plot=test_loss_plot, device=self.device)
                    qtregtuning_history, best_qtreg, _, epstreg = qtregtuner.tune()

                    qtregtotal_losses = np.asarray(qtregtuning_history['best_model_test_loss'])
                    qtregbest_index = np.argmin(qtregtotal_losses)

                    qtregbest_params = {}
                    for key in qtregtuning_history.keys():
                        qtregbest_params[key] = qtregtuning_history[key][qtregbest_index]

                    print('Best Q params:', qbest_params)
                    print('Best Q with treg params:', qtregbest_params)
                    print('Best G params:', gbest_params)

                    output_type_Q = 'categorical'
                    output_size_Q = 1
                    output_type_G = 'categorical'
                    output_size_G = 1
                    input_size_Q = z.shape[-1] + 1  # we will concatenate the treatment var inside the qnet class
                    input_size_G = z.shape[-1]

                    qlayers = qbest_params['layers']
                    qdropout = qbest_params['dropout']
                    qlayer_size = qbest_params['layer_size']
                    qiters = qbest_params['iters']  # override the early stopping iter (will still use early stopping)
                    qlr = qbest_params['lr']
                    qbatch_size = qbest_params['batch_size']
                    qweight_reg = qbest_params['weight_reg']

                    glayers = gbest_params['layers']
                    gdropout = gbest_params['dropout']
                    glayer_size = gbest_params['layer_size']
                    giters = gbest_params['iters']  # override the early stopping iter (will still use early stopping)
                    glr = gbest_params['lr']
                    gbatch_size = gbest_params['batch_size']
                    gweight_reg = gbest_params['weight_reg']

                    qtreglayers = qtregbest_params['layers']
                    qtregdropout = qtregbest_params['dropout']
                    qtreglayer_size = qtregbest_params['layer_size']
                    qtregiters = qtregbest_params['iters']  # override the early stopping iter (will still use early stopping)
                    qtreglr = qtregbest_params['lr']
                    qtregbatch_size = qtregbest_params['batch_size']
                    qtregweight_reg = qtregbest_params['weight_reg']

                print('==============ESTIMATION==============')

                if self.run_NN or self.run_treg or self.run_NN_SL:
                    if self.multinet:
                        qnet = QMultiNet(input_size=input_size_Q, num_layers=qlayers, device=self.device,
                                    layers_size=qlayer_size, output_size=output_size_Q,
                                    output_type=output_type_Q, dropout=qdropout, use_t=use_t, layerwise_optim=self.layerwise_optim).to(self.device)

                        gnet = GMultiNet(input_size=input_size_G, num_layers=glayers, device=self.device,
                                    layers_size=glayer_size, output_size=output_size_G,
                                    output_type=output_type_G, dropout=gdropout, layerwise_optim=self.layerwise_optim).to(self.device)
                        print('Training G....')
                        # def G trainer
                        gtrainer = MultiNetTrainer(net=gnet, net_type='G', beta=0.0, iterations=giters, device=self.device,
                                           outcome_type=output_type_G, data_masking=self.data_masking,
                                           batch_size=gbatch_size, test_iter=1000, lr=glr,
                                           weight_reg=gweight_reg, test_loss_plot=test_loss_plot, split=False, layerwise_optim=self.layerwise_optim)
                        train_loss_g_, val_loss_g_, stop_it_g, best_model_g, best_model_test_loss_g, eps, last_modelg, gbetas_bm, gbetas_lm = gtrainer.train(
                            x, y, z)
                    else:
                        qnet = QNet(input_size=input_size_Q, num_layers=qlayers, device=self.device,
                                    layers_size=qlayer_size, output_size=output_size_Q,
                                    output_type=output_type_Q, dropout=qdropout, use_t=use_t).to(self.device)

                        gnet = GNet(input_size=input_size_G, num_layers=glayers, device=self.device,
                                    layers_size=glayer_size, output_size=output_size_G,
                                    output_type=output_type_G, dropout=gdropout).to(self.device)
                        print('Training G....')
                        # def G trainer
                        gtrainer = Trainer(net=gnet, net_type='G', beta=0.0, iterations=giters, outcome_type=output_type_G,
                                           batch_size=gbatch_size, test_iter=1000, lr=glr, weight_reg=gweight_reg, device=self.device,
                                           test_loss_plot=test_loss_plot, calibration=self.calibration, split=False)
                    # train G
                        train_loss_g_, val_loss_g_, stop_it_g, best_model_g, best_model_test_loss_g, eps, last_modelg, gtemp_bm, gtemp_lm = gtrainer.train(
                            x, y, z)

                    if use_last_model:
                        best_model_g = last_modelg
                        if self.multinet:
                            gbetas_bm = gbetas_lm
                            betas_g.append(gbetas_bm)
                        else:
                            gtemp_bm = gtemp_lm
                    # Get x_preds from G
                    _, x_pred, x_logits = gtrainer.test(best_model_g, x, y, z)

                    if self.multinet:
                        x_pred = x_pred.detach().cpu().numpy()
                        x_pred = np.dot(x_pred, gbetas_bm)
                        x_pred = torch.tensor(x_pred, dtype=torch.float32).to(self.device)
                    elif self.calibration:
                        print('Calibrating probs')
                        x_pred = torch.sigmoid(T_scaling(x_logits, gtemp_bm))

                    # def Q trainer (no treg)
                    if self.multinet:
                        qtrainer = MultiNetTrainer(net=qnet, net_type='Q', beta=0.0, iterations=qiters,
                                               outcome_type=output_type_Q, device=self.device,
                                               batch_size=qbatch_size, test_iter=1000, lr=qlr,
                                               weight_reg=qweight_reg, data_masking=self.data_masking,
                                               test_loss_plot=test_loss_plot, split=False, layerwise_optim=self.layerwise_optim)

                        print('Training Q....')
                        # train Q  (no treg)
                        train_loss_q_, val_loss_q_, stop_it_q, best_model_q, best_model_test_loss_q, eps, last_modelq, qbetas_bm, qbetas_lm = qtrainer.train(
                            x, y, z, x_pred=x_pred)

                    else:
                        qtrainer = Trainer(net=qnet, net_type='Q', beta=0.0, iterations=qiters, outcome_type=output_type_Q,
                                           batch_size=qbatch_size, test_iter=1000, lr=qlr, weight_reg=qweight_reg, device=self.device,
                                           test_loss_plot=test_loss_plot, calibration=self.calibration, split=False)

                        print('Training Q....')
                        # train Q  (no treg)
                        train_loss_q_, val_loss_q_, stop_it_q, best_model_q, best_model_test_loss_q, eps, last_modelq, qtemp_bm, qtemp_lm = qtrainer.train(
                            x, y, z, x_pred=x_pred)

                    if use_last_model:
                        best_model_q = last_modelq
                        if self.multinet:
                            qbetas_bm = qbetas_lm
                            betas_q.append(qbetas_bm)
                        else:
                            qtemp_bm = qtemp_lm

                    # generate counterfactual preds (no treg)
                    _, Q10, Q10_logits = qtrainer.test(best_model_q, x, y, z, x_pred)
                    _, Q1, Q1_logits = qtrainer.test(best_model_q, x_int1, y, z, x_pred)
                    _, Q0, Q0_logits = qtrainer.test(best_model_q, x_int0, y, z, x_pred)
                    _, G10, G10_logits = gtrainer.test(best_model_g, x, y, z)

                    if self.calibration:
                        print('Calibrating probs')
                        Q10 = torch.sigmoid(T_scaling(Q10_logits, qtemp_bm))
                        Q1 = torch.sigmoid(T_scaling(Q1_logits, qtemp_bm))
                        Q0 = torch.sigmoid(T_scaling(Q0_logits, qtemp_bm))
                        G10 = torch.sigmoid(T_scaling(G10_logits, gtemp_bm))

                    Q10 = Q10.detach().cpu().numpy()
                    Q1 = Q1.detach().cpu().numpy()
                    Q0 = Q0.detach().cpu().numpy()
                    G10 = G10.detach().cpu().numpy()

                    if self.multinet:
                        Q10 = np.dot(Q10, qbetas_bm)
                        Q1 = np.dot(Q1, qbetas_bm)
                        Q0 = np.dot(Q0, qbetas_bm)
                        G10 = np.dot(G10, gbetas_bm)

                    G10 = np.clip(G10, a_min=0.025, a_max=0.9975)

                    biased_psi = (Q1 - Q0).mean()
                    # record initial estimate
                    estimates_naive.append(biased_psi)

                    # record one step estimate
                    Q1_star, Q0_star = one_step(x_, y_, Q0, Q1, G10)
                    estimates_upd_1s.append((Q1_star - Q0_star).mean())

                    #  submodel approach
                    Q1_star_solve, Q0_star_solve = submodel(x_, y_, Q1, Q0, Q10, G10)

                    # record estimate
                    estimates_upd_submod.append((Q1_star_solve - Q0_star_solve).mean())

                if self.run_treg:
                    # -------------------------------------TREG ------------------------------------
                    print('Redefining and training Q with t-reg....')
                    # redefine Q network
                    qnet = None
                    qtrainer = None
                    if self.multinet:
                        qnet = QMultiNet(input_size=input_size_Q, num_layers=qtreglayers, device=self.device,
                                    layers_size=qtreglayer_size, output_size=output_size_Q,
                                    output_type=output_type_Q, dropout=qtregdropout, use_t=use_t, layerwise_optim=self.layerwise_optim).to(self.device)
                        # redefine Q trainer (treg enabled)
                        qtrainer = MultiNetTrainer(net=qnet, net_type='Q', beta=1.0, iterations=qtregiters,
                                           outcome_type=output_type_Q, device=self.device,
                                           batch_size=qtregbatch_size, test_iter=1000, lr=qtreglr,
                                           weight_reg=qtregweight_reg, data_masking=self.data_masking,
                                           test_loss_plot=test_loss_plot,  split=False, layerwise_optim=self.layerwise_optim)

                        # retrain Q using same x_preds which were generated ABOVE
                        train_loss_q_, val_loss_q_, stop_it_q, best_model_q, best_model_test_loss_q, eps, last_modelq, qbetas_bm, qbetas_lm = qtrainer.train(
                            x, y, z, x_pred=x_pred)
                    else:
                        qnet = QNet(input_size=input_size_Q, num_layers=qtreglayers, device=self.device,
                                    layers_size=qtreglayer_size, output_size=output_size_Q,
                                    output_type=output_type_Q, dropout=qtregdropout, use_t=use_t).to(self.device)
                        # redefine Q trainer (treg enabled)
                        qtrainer = Trainer(net=qnet, net_type='Q', beta=1.0, iterations=qtregiters, outcome_type=output_type_Q, device=self.device,
                                           batch_size=qtregbatch_size, test_iter=1000, lr=qtreglr, weight_reg=qtregweight_reg,
                                           test_loss_plot=test_loss_plot, calibration=self.calibration, split=False)

                        # retrain Q using same x_preds which were generated ABOVE
                        train_loss_q_, val_loss_q_, stop_it_q, best_model_q, best_model_test_loss_q, eps, last_modelq, qtemp_bm, qtemp_lm = qtrainer.train(
                            x, y, z, x_pred=x_pred)

                    if use_last_model:
                        best_model_q = last_modelq
                        if self.multinet:
                            qbetas_bm = qbetas_lm
                        else:
                            qtemp_bm = qtemp_lm
                    eps_.append(eps)
                    # generate counterfactual preds (treg enabled)
                    _, Q10, G10_logits = qtrainer.test(best_model_q, x, y, z, x_pred)
                    _, Q1, Q1_logits = qtrainer.test(best_model_q, x_int1, y, z, x_pred)
                    _, Q0, Q0_logits = qtrainer.test(best_model_q, x_int0, y, z, x_pred)

                    if self.calibration:
                        print('Calibrating probs')
                        Q10 = torch.sigmoid(T_scaling(Q10_logits, qtemp_bm))
                        Q1 = torch.sigmoid(T_scaling(Q1_logits, qtemp_bm))
                        Q0 = torch.sigmoid(T_scaling(Q0_logits, qtemp_bm))

                    Q10 = Q10.detach().cpu().numpy()
                    Q1 = Q1.detach().cpu().numpy()
                    Q0 = Q0.detach().cpu().numpy()

                    if self.multinet:
                        Q10 = np.dot(Q10, qbetas_bm)
                        Q1 = np.dot(Q1, qbetas_bm)
                        Q0 = np.dot(Q0, qbetas_bm)

                    upd_psi_treg = (Q1 - Q0).mean()
                    # record estimate
                    estimates_upd_treg.append(upd_psi_treg)

                if self.run_treg_SL:
                    # ------------------------PREVIOUS TREG MODEL WITH SUBMODEL UPDATE COMBINED------------------------------------------------------------
                    print('Combining TREG with submodel update')
                    #  submodel approach
                    Q1_star_solve, Q0_star_solve = submodel(x_, y_, Q1, Q0, Q10, G10)
                    # record estimate
                    estimates_upd_treg_submod.append((Q1_star_solve - Q0_star_solve).mean())

                if self.run_NN_SL:
                    # ------------------------------- NAIVE NN WITH SL FOR TREATMENT MODEL ----------------------------


                    print('Repeating but using a super learner as the propensity score model...')
                    if self.multinet:
                        qnet = QMultiNet(input_size=input_size_Q, num_layers=qlayers,
                            layers_size=qlayer_size, output_size=output_size_Q, device=self.device,
                            output_type=output_type_Q, dropout=qdropout, use_t=use_t, layerwise_optim=self.layerwise_optim).to(self.device)
                    else:
                        qnet = QNet(input_size=input_size_Q, num_layers=qlayers, device=self.device,
                                    layers_size=qlayer_size, output_size=output_size_Q,
                                    output_type=output_type_Q, dropout=qdropout, use_t=use_t).to(self.device)

                    print('Training G....')
                    Gest_dict = {'LR': LogisticRegression(), 'SVC': SVC(probability=True),
                                 'RF': RandomForestClassifier(), 'KNN': KNeighborsClassifier(),
                                 'AB': AdaBoostClassifier(), 'poly': 'poly'}

                    GSL = SuperLearner(output=self.output, est_dict=Gest_dict, k=self.k)
                    GSL.train_combiner(z_, x_[:, 0])
                    GSL.train_superlearner(z_, x_[:, 0])
                    x_pred = np.clip(GSL.estimation(z_, x_[:, 0]), a_min=0.025, a_max=0.975)

                    x_pred = torch.tensor(x_pred).type(torch.float32).to(self.device)

                    # def Q trainer (no treg)
                    if self.multinet:
                        qtrainer = MultiNetTrainer(net=qnet, net_type='Q', beta=0.0, iterations=qiters, outcome_type=output_type_Q,
                                   batch_size=qbatch_size, test_iter=1000, lr=qlr, weight_reg=qweight_reg, device=self.device,
                                   test_loss_plot=test_loss_plot, split=False, data_masking=self.data_masking, layerwise_optim=self.layerwise_optim)

                        print('Training Q....')
                        # train Q  (no treg)
                        train_loss_q_, val_loss_q_, stop_it_q, best_model_q, best_model_test_loss_q, eps, last_modelq, qbetas_bm, qbetas_lm = qtrainer.train(
                            x, y, z, x_pred=x_pred)
                    else:
                        qtrainer = Trainer(net=qnet, net_type='Q', beta=0.0, iterations=qiters, outcome_type=output_type_Q,
                                           batch_size=qbatch_size, test_iter=1000, lr=qlr, weight_reg=qweight_reg, device=self.device,
                                           test_loss_plot=test_loss_plot, calibration=self.calibration, split=False)

                        print('Training Q....')
                        # train Q  (no treg)
                        train_loss_q_, val_loss_q_, stop_it_q, best_model_q, best_model_test_loss_q, eps, last_modelq, qtemp_bm, qtemp_lm = qtrainer.train(
                            x, y, z, x_pred=x_pred)

                    if use_last_model:
                        best_model_q = last_modelq
                        if self.multinet:
                            qbetas_bm = qbetas_lm
                        else:
                            qtemp_bm = qtemp_lm
                    # generate counterfactual preds (no treg)
                    _, Q10, Q10_logits = qtrainer.test(best_model_q, x, y, z, x_pred)
                    _, Q1, Q1_logits = qtrainer.test(best_model_q, x_int1, y, z, x_pred)
                    _, Q0, Q0_logits = qtrainer.test(best_model_q, x_int0, y, z, x_pred)

                    if self.calibration:
                        print('Calibrating probs')
                        Q10 = torch.sigmoid(T_scaling(Q10_logits, qtemp_bm))
                        Q1 = torch.sigmoid(T_scaling(Q1_logits, qtemp_bm))
                        Q0 = torch.sigmoid(T_scaling(Q0_logits, qtemp_bm))

                    Q10 = Q10.detach().cpu().numpy()
                    Q1 = Q1.detach().cpu().numpy()
                    Q0 = Q0.detach().cpu().numpy()

                    if self.multinet:
                        Q10 = np.dot(Q10, qbetas_bm)
                        Q1 = np.dot(Q1, qbetas_bm)
                        Q0 = np.dot(Q0, qbetas_bm)

                    G10 = np.clip(GSL.estimation(z_, x_[:, 0]), a_min=0.025, a_max=0.975)


                    biased_psi = (Q1 - Q0).mean()
                    # record initial estimate
                    estimates_naive_halfSL.append(biased_psi)

                    # one step approach
                    Q1_star, Q0_star = one_step(x_, y_, Q0, Q1, G10)
                    # record estimate
                    estimates_upd_1s_halfSL.append((Q1_star - Q0_star).mean())

                    #  submodel approach
                    Q1_star_solve, Q0_star_solve = submodel(x_, y_, Q1, Q0, Q10, G10)
                    # record estimate
                    estimates_upd_submod_halfSL.append((Q1_star_solve - Q0_star_solve).mean())

                # -------------------------------------TREG + half SL ------------------------------------
                if self.run_treg_SL:
                    print('Redefining and training Q with t-reg....')
                    # redefine Q network
                    qnet = None
                    if self.multinet:
                        qnet = QMultiNet(input_size=input_size_Q, num_layers=qtreglayers, device=self.device,
                                    layers_size=qtreglayer_size, output_size=output_size_Q,
                                    output_type=output_type_Q, dropout=qtregdropout, use_t=use_t, layerwise_optim=self.layerwise_optim).to(self.device)
                        # redefine Q trainer (treg enabled)
                        qtrainer = MultiNetTrainer(net=qnet, net_type='Q', beta=1.0, iterations=qtregiters, outcome_type=output_type_Q,
                                           batch_size=qtregbatch_size, test_iter=1000, lr=qtreglr, device=self.device,
                                           weight_reg=qtregweight_reg, data_masking=self.data_masking,
                                           test_loss_plot=test_loss_plot, split=False, layerwise_optim=self.layerwise_optim)

                        # retrain Q using same x_preds which were generated ABOVE
                        train_loss_q_, val_loss_q_, stop_it_q, best_model_q, best_model_test_loss_q, eps, last_modelq, qbetas_bm, qbetas_lm = qtrainer.train(
                            x, y, z, x_pred=x_pred)
                    else:
                        qnet = QNet(input_size=input_size_Q, num_layers=qtreglayers, device=self.device,
                                    layers_size=qtreglayer_size, output_size=output_size_Q,
                                    output_type=output_type_Q, dropout=qtregdropout, use_t=use_t).to(self.device)
                        # redefine Q trainer (treg enabled)
                        qtrainer = Trainer(net=qnet, net_type='Q', beta=1.0, iterations=qtregiters, outcome_type=output_type_Q, device=self.device,
                                           batch_size=qtregbatch_size, test_iter=1000, lr=qtreglr, weight_reg=qtregweight_reg,
                                           test_loss_plot=test_loss_plot, calibration=self.calibration, split=False)

                        # retrain Q using same x_preds which were generated ABOVE
                        train_loss_q_, val_loss_q_, stop_it_q, best_model_q, best_model_test_loss_q, eps, last_modelq, qtemp_bm, qtemp_lm = qtrainer.train(
                            x, y, z, x_pred=x_pred)

                    if use_last_model:
                        best_model_q = last_modelq
                        if self.multinet:
                            qbetas_bm = qbetas_lm
                        else:
                            qtemp_bm = qtemp_lm
                    eps_.append(eps)
                    # generate counterfactual preds (treg enabled)
                    _, Q10, Q10_logits = qtrainer.test(best_model_q, x, y, z, x_pred)
                    _, Q1, Q1_logits = qtrainer.test(best_model_q, x_int1, y, z, x_pred)
                    _, Q0, Q0_logits = qtrainer.test(best_model_q, x_int0, y, z, x_pred)

                    if self.calibration:
                        print('Calibrating probs')
                        Q10 = torch.sigmoid(T_scaling(Q10_logits, qtemp_bm))
                        Q1 = torch.sigmoid(T_scaling(Q1_logits, qtemp_bm))
                        Q0 = torch.sigmoid(T_scaling(Q0_logits, qtemp_bm))

                    Q10 = Q10.detach().cpu().numpy()
                    Q1 = Q1.detach().cpu().numpy()
                    Q0 = Q0.detach().cpu().numpy()

                    if self.multinet:
                        Q10 = np.dot(Q10, qbetas_bm)
                        Q1 = np.dot(Q1, qbetas_bm)
                        Q0 = np.dot(Q0, qbetas_bm)

                    upd_psi_treg = (Q1 - Q0).mean()
                    # record estimate
                    estimates_upd_treg_halfSL.append(upd_psi_treg)

                    # ------------------------PREVIOUS TREG MODEL WITH SUBMODEL UPDATE COMBINED and SL ------------------------------------------------------------
                    print('Combining TREG with submodel update')
                    Q1_star_solve, Q0_star_solve = submodel(x_, y_, Q1, Q0, Q10, G10)
                    # record estimate
                    estimates_upd_treg_submod_halfSL.append((Q1_star_solve - Q0_star_solve).mean())

                if self.run_LR:
                    # ------------------------LOG REG------------------------------------------------------------
                    print('Running LR...')
                    Q = LogisticRegression().fit(np.concatenate([x_, z_], 1), y_[:, 0])
                    Q1 = np.clip(Q.predict_proba((np.concatenate([x_int1_, z_], 1)))[:, 1:], a_min=0.025, a_max=0.975)
                    Q0 = np.clip(Q.predict_proba((np.concatenate([x_int0_, z_], 1)))[:, 1:], a_min=0.025, a_max=0.975)
                    Q10 = np.clip(Q.predict_proba((np.concatenate([x_, z_], 1)))[:, 1:], a_min=0.025, a_max=0.975)
                    biased_psi = (Q1 - Q0).mean()

                    G = LogisticRegression().fit(z_, x_[:, 0])
                    G10 = np.clip(G.predict_proba(z_), a_min=0.025, a_max=0.975)[:, 1:]

                    # submodel approach
                    Q1_star_solve, Q0_star_solve = submodel(x_, y_, Q1, Q0, Q10, G10)

                    # one step approach
                    Q1_star, Q0_star = one_step(x_, y_, Q0, Q1, G10)

                    estimates_upd_submod_LR.append((Q1_star_solve - Q0_star_solve).mean())
                    estimates_naive_LR.append(biased_psi)
                    estimates_upd_1s_LR.append((Q1_star - Q0_star).mean())

                if self.run_SL:
                    print('Running SL...')
                    # ------------------------SUPER LEARNER------------------------------------------------------------
                    Qest_dict = {'LR': LogisticRegression(), 'SVC': SVC(probability=True),
                                 'RF': RandomForestClassifier(), 'KNN': KNeighborsClassifier(),
                                 'AB': AdaBoostClassifier(), 'poly': 'poly'}


                    QSL = SuperLearner(output=self.output, est_dict=Qest_dict, k=self.k)
                    QSL.train_combiner(np.concatenate([x_, z_], 1), y_[:, 0])
                    QSL.train_superlearner(np.concatenate([x_, z_], 1), y_[:, 0])

                    Q10 = np.clip(QSL.estimation(np.concatenate([x_, z_], 1), y_[:, 0]),
                                  a_min=0.025, a_max=0.975)
                    Q1 = np.clip(QSL.estimation(np.concatenate([x_int1_, z_], 1), y_[:, 0]),
                                 a_min=0.025, a_max=0.975)
                    Q0 = np.clip(QSL.estimation(np.concatenate([x_int0_, z_], 1), y_[:, 0]),
                                 a_min=0.025, a_max=0.975)

                    Gest_dict = {'LR': LogisticRegression(), 'SVC': SVC(probability=True),
                                 'RF': RandomForestClassifier(), 'KNN': KNeighborsClassifier(),
                                 'AB': AdaBoostClassifier(), 'poly': 'poly'}


                    GSL = SuperLearner(output=self.output, est_dict=Gest_dict, k=self.k)
                    GSL.train_combiner(z_, x_[:, 0])
                    GSL.train_superlearner(z_, x_[:, 0])
                    G10 = np.clip(GSL.estimation(z_, x_[:, 0]), a_min=0.025, a_max=0.975)

                    # submodel approach
                    Q1_star_solve, Q0_star_solve = submodel(x_, y_, Q1, Q0, Q10, G10)

                    # one step approach
                    Q1_star, Q0_star = one_step(x_, y_, Q0, Q1, G10)

                    estimates_upd_submod_SL.append((Q1_star_solve - Q0_star_solve).mean())
                    estimates_naive_SL.append(biased_psi)
                    estimates_upd_1s_SL.append((Q1_star - Q0_star).mean())

                i += 1


            except Exception as exc:
                print(traceback.format_exc())
                print(exc)
                exc
                print('Error training one of the models, restarting run')

            col_names = ['naive', 'submodel', 'one_step', 'treg', 'treg_submod',
                         'naive_LR', 'submodel_LR', 'one_step_LR',
                         'naive_SL', 'submodel_SL', 'one_step_SL',
                         'naive_halfSL', 'submodel_halfSL', 'one_step_halfSL', 'treg_halfSL', 'treg_submod_halfSL', 'sample_truth']

            if self.multinet:
                betas_q_ = np.asarray(betas_q)
                betas_g_ = np.asarray(betas_g)
                df_betas_g = pd.DataFrame(betas_g_)
                df_betas_q = pd.DataFrame(betas_q_)

                df_betas_g.to_csv(self.fn + '{}_betas_G_t_predictor_{}_datarand_{}_{}_{}_datamask_{}_{}_{}_start_iter_{}.csv'.format(self.run, use_t, self.data_rand, self.dataset,
                                                                                self.multinet, self.data_masking, current_time, self.N, self.starting_iter), index=False)
                df_betas_q.to_csv(self.fn + '{}_betas_Q_t_predictor_{}_datarand_{}_{}_{}_datamask_{}_{}_{}_start_iter_{}.csv'.format(self.run, use_t, self.data_rand, self.dataset,
                                                                                    self.multinet, self.data_masking, current_time, self.N, self.starting_iter), index=False)
            df = pd.DataFrame([estimates_naive, estimates_upd_submod, estimates_upd_1s,
                               estimates_upd_treg, estimates_upd_treg_submod,
                               estimates_naive_LR, estimates_upd_submod_LR, estimates_upd_1s_LR,
                               estimates_naive_SL, estimates_upd_submod_SL, estimates_upd_1s_SL,
                               estimates_naive_halfSL, estimates_upd_submod_halfSL, estimates_upd_1s_halfSL,
                               estimates_upd_treg_halfSL, estimates_upd_treg_submod_halfSL, sample_truth]).T
            df.columns = col_names
            df.to_csv(self.fn + '{}_Results_t_predictor_{}_datarand_{}_{}_multinet_{}_datamask_{}_layerwise_{}_{}_{}_start_iter_{}.csv'.format(self.run, use_t, self.data_rand,
                                                                                                 self.dataset,
                                                                                                 self.multinet,
                                                                                                 self.data_masking,
                                                                                                                 self.layerwise_optim,
                                                                                                                 current_time, self.N, self.starting_iter), index=False)

        return true_psi, df
