from copy import deepcopy
from os import makedirs, path as osp
from time import perf_counter
from typing import Dict, Tuple, Any
import warnings

from lightning import Fabric
from lightning.fabric.wrappers import _FabricModule
import numpy as np
from scipy.stats import pearsonr, spearmanr, linregress
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
from torch.nn.functional import mse_loss, binary_cross_entropy_with_logits
# from torch.utils.tensorboard import SummaryWriter

from .datasets import Dataset
from .pdgrapher import PDGrapher
from ._utils import get_thresholds, calculate_loss_sample_weights, DummyWriter, EarlyStopping
from time import time

class Trainer:

    def __init__(self, fabric_kwargs: Dict[str, Any] = {}, **kwargs) -> None:
        # Logger
        self.use_logging = kwargs.pop("log", False)
        self.logging_dir = osp.abspath(kwargs.pop("logging_dir", "examples/PDGrapher")) # default PDGrapher
        self.logging_name = kwargs.pop("logging_name", "")
        self.writer = DummyWriter()
        self.log_train = kwargs.pop("log_train", False)
        self.log_test = kwargs.pop("log_test", False)

        # Other training parameters
        self.use_forward_data = kwargs.pop("use_forward_data", True)
        self.use_backward_data = kwargs.pop("use_backward_data", False)
        self.use_intervention_data = kwargs.pop("use_intervention_data", True)
        self.use_supervision = kwargs.pop("use_supervision", False)
        self.supervision_multiplier = kwargs.pop("supervision_multiplier", 1)
        self.use_lr_scheduler = kwargs.pop("use_lr_scheduler", False)

        if len(kwargs):
            warnings.warn(f"Unknown kwargs: {list(kwargs.keys())}")

        # Fabric setup
        self.fabric = Fabric(**fabric_kwargs)

        # Placeholder functions for optimizers & schedulers (zero_grad and step)
        self._op1_zero_grad = lambda: None
        self._op1_step = lambda: None
        self._op2_zero_grad = lambda: None
        self._op2_step = lambda: None
        self._sc1_step = lambda: None
        self._sc2_step = lambda: None

    def logging_paths(self, *, path: str = None, name: str = None) -> None:
        if path:
            self.logging_dir = osp.abspath(path)
        if name:
            self.logging_name = name
            if not name.endswith("_"):
                self.logging_name += "_"

    def train(self, model: PDGrapher, dataset: Dataset, n_epochs: int, early_stopping_kwargs: Dict[str, Any] = {}) -> Dict[str, Dict[str, float]]:

        t0 = time()
        # Loss weights, thresholds
        sample_weights_model_2_backward = calculate_loss_sample_weights(dataset.train_dataset_backward, "intervention")
        sample_weights_model_2_backward = self.fabric.to_device(sample_weights_model_2_backward)
        pos_weight = sample_weights_model_2_backward[1] / sample_weights_model_2_backward[0]
        thresholds = get_thresholds(dataset)
        thresholds = {k: self.fabric.to_device(v) for k, v in thresholds.items()} # do we really need them?
        model.response_prediction.edge_index = self.fabric.to_device(model.response_prediction.edge_index)
        model.perturbation_discovery.edge_index = self.fabric.to_device(model.perturbation_discovery.edge_index)
        t1 = time()
        print('Time in Loss weights, thresholds: {:.3f} secs'.format(t1 - t0))


        t0 = time()
        # Optimizers & Schedulers
        model_1, model_2 = self._configure_model_with_optimizers_and_schedulers(model)
        t1 = time()
        print('Time in Optimizers & Schedulers: {:.3f} secs'.format(t1 - t0))


        if self.use_logging:
            # Log model parameters
            with open(osp.join(self.logging_dir, f"{self.logging_name}params.txt"), "w") as log_params:
                log_params.write(f"Response Prediction Model parameters:\t{sum(p.numel() for p in model_1.parameters())}\n")
                log_params.write(f"Perturbation Discovery Model parameters:\t{sum(p.numel() for p in model_2.parameters())}\n")
            # Log metrics
            log_metrics = open(osp.join(self.logging_dir, f"{self.logging_name}metrics.txt"), "w")
        makedirs(self.logging_dir, exist_ok=True)


        t0 = time()
        # Dataloaders
        # (
        #     train_loader_forward, train_loader_backward,
        #     val_loader_forward, val_loader_backward,
        #     test_loader_forward, test_loader_backward
        # ) = self.fabric.setup_dataloaders(*dataset.get_dataloaders())

        (
            train_loader_forward, train_loader_backward,
            val_loader_forward, val_loader_backward,
            test_loader_forward, test_loader_backward
        ) = dataset.get_dataloaders(num_workers = 20)


        t1 = time()
        print('Time in Dataloaders: {:.3f} secs'.format(t1 - t0))

        t0 = time()
        # Early stopping
        es_1 = EarlyStopping(model=model_1, save_path=osp.join(self.logging_dir, f"{self.logging_name}response_prediction.pt"), **early_stopping_kwargs)
        es_2 = EarlyStopping(model=model_2, save_path=osp.join(self.logging_dir, f"{self.logging_name}perturbation_discovery.pt"), **early_stopping_kwargs)
        if not model._train_response_prediction:
            es_1.is_stopped = True
        if not model._train_perturbation_discovery:
            es_2.is_stopped = True
        t1 = time()
        print('Time in Early stopping: {:.3f} secs'.format(t1 - t0))


        # Train loop
        for epoch in range(1, n_epochs+1):
            start = perf_counter()

            # TRAIN
            tic = perf_counter()
            loss, loss_f, loss_b = self._train_one_pass(
                model_1, model_2, es_1, es_2, train_loader_forward, train_loader_backward,
                thresholds, pos_weight)
            toc = perf_counter()
            print(f"Train call: {toc-tic:.2f}s")

            # VALIDATION
            tic = perf_counter()
            val_loss, val_loss_f, val_loss_b = self._val_one_pass(
                model_1, model_2, es_1, es_2, val_loader_forward, val_loader_backward,
                thresholds, pos_weight)
            toc = perf_counter()
            print(f"Validation call: {toc-tic:.2f}s")

            # Log additional metrics
            summ_train = ""
            if self.log_train:
                tic = perf_counter()
                train_performance = self._test_one_pass(
                    model_1, model_2, es_1, es_2, train_loader_forward, train_loader_backward, thresholds)
                toc = perf_counter()
                print(f"Test call (train dataset): {toc-tic:.2f}s")
                summ_train = self._test_to_str(train_performance, "TRAIN")
                self._test_to_writer(train_performance, "train", epoch)

            summ_test = ""
            if self.log_test:
                tic = perf_counter()
                test_performance = self._test_one_pass(
                    model_1, model_2, es_1, es_2, test_loader_forward, test_loader_backward, thresholds)
                toc = perf_counter()
                print(f"Test call (test dataset): {toc-tic:.2f}s")
                summ_test = self._test_to_str(test_performance, "TEST")
                self._test_to_writer(test_performance, "test", epoch)

            # Log basic numbers
            self.writer.add_scalar("Loss/total", loss, epoch)
            self.writer.add_scalar("Loss/forward", loss_f, epoch)
            self.writer.add_scalar("Loss/backward", loss_b, epoch)
            self.writer.add_scalar("Loss/val/forward", val_loss_f, epoch)
            self.writer.add_scalar("Loss/val/backward", val_loss_b, epoch)

            end = perf_counter()

            # Log epoch summary
            summary = (
                f"Epoch {epoch:03d} [{end-start:.2f}s], "
                f"Train loss: {loss:.4f} (forward: {loss_f:.4f}, backward: {loss_b:.4f}), "
                f"Val loss: {val_loss:.4f} (forward: {val_loss_f:.4f}, backward: {val_loss_b:.4f})"
            )
            summary += summ_train + summ_test
            print(summary)
            if self.use_logging:
                log_metrics.write(summary + "\n")

            # Early stopping
            if not es_1.is_stopped and es_1(val_loss_f):
                print("Early stopping model 1 (response prediction)")
            if not es_2.is_stopped and es_2(val_loss_b):
                print("Early stopping model 2 (intervention discovery)")
            if es_1.is_stopped and es_2.is_stopped:
                break

            print()

        if self.use_logging:
            log_metrics.close()

        # Restore best models
        if model._train_response_prediction:
            model.response_prediction = es_1.load_model()
            model_1 = self.fabric.setup(model.response_prediction)
        if model._train_perturbation_discovery:
            model.perturbation_discovery = es_2.load_model()
            model_2 = self.fabric.setup(model.perturbation_discovery)

        # Enable testing of the models
        es_1.is_stopped = False
        es_2.is_stopped = False
        train_perf = self._test_one_pass(model_1, model_2, es_1, es_2, train_loader_forward, train_loader_backward, thresholds)
        test_perf = self._test_one_pass(model_1, model_2, es_1, es_2, test_loader_forward, test_loader_backward, thresholds)

        model_performance = {
            "train": train_perf,
            "test": test_perf
        }

        return model_performance

    def train_kfold(self, model: PDGrapher, dataset: Dataset, n_epochs: int, early_stopping_kwargs: Dict[str, Any] = {}):
        model_performances = list()
        _prev_name = self.logging_name
        
        for fold_idx in range(1, dataset.num_of_folds + 1):
            dataset.prepare_fold(fold_idx)
            self.logging_paths(name=f"{_prev_name}_fold_{fold_idx}_")
            model_tmp = deepcopy(model)
            model_performance = self.train(model_tmp, dataset, n_epochs, early_stopping_kwargs)
            model_performances.append(model_performance)

        self.logging_paths(name=_prev_name)

        return model_performances

    def _train_one_pass(self, model_1, model_2, es_1, es_2, loader_forward, loader_backward,
                        thresholds, pos_weight) -> Tuple[float, float, float]:
        l_response = 0
        l_intervention = 0
        noptims_response = 0
        noptims_intervention = 0

        
        # self.fabric.to_device(
        # Do we train the response prediction model?
        if not es_1.is_stopped:
            model_1.train()
            if self.use_forward_data:
                for data in loader_forward:
                    self._op1_zero_grad()
                    output_forward, _ = model_1(torch.concat([self.fabric.to_device(data.healthy.view(-1, 1)), self.fabric.to_device(data.mutations.view(-1, 1))], 1), self.fabric.to_device(data.batch), binarize_intervention=False, threshold_input=thresholds["healthy"])
                    loss_forward = mse_loss(output_forward.view(-1), self.fabric.to_device(data.diseased))
                    self.fabric.backward(loss_forward)
                    self._op1_step()
                    self._sc1_step()
                    l_response += float(loss_forward)
                noptims_response += len(loader_forward)
            if self.use_backward_data:
                for data in loader_backward:
                    self._op1_zero_grad()
                    output_forward, _ = model_1(torch.concat([self.fabric.to_device(data.diseased.view(-1, 1)), self.fabric.to_device(data.intervention.view(-1, 1))], 1), self.fabric.to_device(data.batch), binarize_intervention=False, mutilate_mutations=self.fabric.to_device(data.mutations), threshold_input=thresholds["diseased"])
                    loss_forward = mse_loss(output_forward.view(-1), self.fabric.to_device(data.treated))
                    self.fabric.backward(loss_forward)
                    self._op1_step()
                    self._sc1_step()
                    l_response += float(loss_forward)
                noptims_response += len(loader_backward)

        # Do we train the perturbagen discovery model?
        if not es_2.is_stopped:
            model_1.eval()
            model_2.train()
            if self.use_intervention_data:
                for data in loader_backward:
                    self._op2_zero_grad()
                    pred_backward_m2 = model_2(torch.concat([self.fabric.to_device(data.diseased.view(-1, 1)), self.fabric.to_device(data.treated.view(-1, 1))], 1), self.fabric.to_device(data.batch), mutilate_mutations=self.fabric.to_device(data.mutations), threshold_input=thresholds)
                    # prior knowledge: number of perturbations (targets) per drug
                    topK = torch.sum(data.intervention.view(-1, int(data.num_nodes / len(torch.unique(data.batch)))), 1)
                    pred_backward_m1, in_x_binarized = model_1(torch.concat([self.fabric.to_device(data.diseased.view(-1, 1)), pred_backward_m2], 1), self.fabric.to_device(data.batch), mutilate_mutations=self.fabric.to_device(data.mutations), threshold_input=thresholds["diseased"], binarize_intervention=True, topK=topK)
                    loss_backward = mse_loss(pred_backward_m1.view(-1), self.fabric.to_device(data.treated))
                    # adds supervision
                    if self.use_supervision:
                        loss_backward += self.supervision_multiplier * binary_cross_entropy_with_logits(pred_backward_m2.view(-1), self.fabric.to_device(data.intervention), pos_weight=pos_weight)
                    # Freezing response prediction model
                    self._freeze_model(model_1)
                    self.fabric.backward(loss_backward)
                    self._op2_step()
                    self._sc2_step()
                    # Unfreezing response prediction model
                    self._unfreeze_model(model_1)
                    l_intervention += float(loss_backward)
                noptims_intervention += len(loader_backward)
            elif self.use_supervision:
                for data in loader_backward:
                    # Backward
                    self._op2_zero_grad()
                    pred_backward_m2 = model_2(torch.concat([self.fabric.to_device(data.diseased.view(-1, 1)), self.fabric.to_device(data.treated.view(-1, 1))], 1), self.fabric.to_device(data.batch), mutilate_mutations=self.fabric.to_device(data.mutations), threshold_input=thresholds)
                    loss_backward = self.supervision_multiplier * binary_cross_entropy_with_logits(pred_backward_m2.view(-1), self.fabric.to_device(data.intervention), pos_weight=pos_weight)
                    self.fabric.backward(loss_backward)
                    self._op2_step()
                    self._sc2_step()
                    l_intervention += float(loss_backward)
                noptims_intervention += len(loader_backward)

        total_loss = l_response + l_intervention
        total_noptims = noptims_response + noptims_intervention

        return (
            total_loss/total_noptims if total_noptims else total_loss,
            l_response/noptims_response if noptims_response else l_response,
            l_intervention/noptims_intervention if noptims_intervention else l_intervention
        )

    @torch.no_grad()
    def _val_one_pass(self, model_1, model_2, es_1, es_2, loader_forward,
                      loader_backward, thresholds, pos_weight) -> Tuple[float, float]:
        l_response = 0
        l_intervention = 0
        noptims_response = 0
        noptims_intervention = 0

        model_1.eval()
        model_2.eval()

        if not es_1.is_stopped:
            if self.use_forward_data:
                for data in loader_forward:
                    # Forward
                    # regression loss - learns diseased from healthy and mutations
                    output_forward, _ = model_1(torch.concat([self.fabric.to_device(data.healthy.view(-1, 1)), self.fabric.to_device(data.mutations.view(-1, 1))], 1), self.fabric.to_device(data.batch), binarize_intervention=False, threshold_input=thresholds["healthy"])
                    loss_forward = mse_loss(output_forward.view(-1), self.fabric.to_device(data.diseased))
                    l_response += float(loss_forward)
                noptims_response += len(loader_forward)
            if self.use_backward_data:
                for data in loader_backward:
                    out, _ = model_1(torch.concat([self.fabric.to_device(data.diseased.view(-1, 1)), self.fabric.to_device(data.intervention.view(-1, 1))], 1), self.fabric.to_device(data.batch), mutilate_mutations=self.fabric.to_device(data.mutations), binarize_intervention=False, threshold_input=thresholds["diseased"])
                    loss_forward = mse_loss(out.view(-1), self.fabric.to_device(data.treated))
                    l_response += float(loss_forward)
                noptims_response += len(loader_backward)

        if not es_2.is_stopped:
            if self.use_intervention_data:
                for data in loader_backward:
                    # Backward
                    # (1), (2) cycle loss with M_1 frozen
                    pred_backward_m2 = model_2(torch.concat([self.fabric.to_device(data.diseased.view(-1, 1)), self.fabric.to_device(data.treated.view(-1, 1))], 1), self.fabric.to_device(data.batch), mutilate_mutations=self.fabric.to_device(data.mutations), threshold_input=thresholds)
                    # prior knowledge: number of perturbations (targets) per drug
                    topK = torch.sum(data.intervention.view(-1, int(data.num_nodes / len(torch.unique(data.batch)))), 1)
                    pred_backward_m1, in_x_binarized = model_1(torch.concat([self.fabric.to_device(data.diseased.view(-1, 1)), pred_backward_m2], 1), self.fabric.to_device(data.batch), mutilate_mutations=self.fabric.to_device(data.mutations), threshold_input=thresholds["diseased"], binarize_intervention=True, topK=topK)
                    loss_backward = mse_loss(pred_backward_m1.view(-1), self.fabric.to_device(data.treated))
                    if self.use_supervision:
                        loss_backward += self.supervision_multiplier * binary_cross_entropy_with_logits(pred_backward_m2.view(-1), self.fabric.to_device(data.intervention), pos_weight=pos_weight)
                    l_intervention += float(loss_backward)
                noptims_intervention += len(loader_backward)
            # supervision for U'
            elif self.use_supervision:
                for data in loader_backward:
                    pred_backward_m2 = model_2(torch.concat([self.fabric.to_device(data.diseased.view(-1, 1)), self.fabric.to_device(data.treated.view(-1, 1))], 1), self.fabric.to_device(data.batch), mutilate_mutations=self.fabric.to_device(data.mutations), threshold_input=thresholds)
                    loss_backward = self.supervision_multiplier * binary_cross_entropy_with_logits(pred_backward_m2.view(-1), self.fabric.to_device(data.intervention), pos_weight=pos_weight)
                    l_intervention += float(loss_backward)
                noptims_intervention += len(loader_backward)

        total_loss = l_response + l_intervention
        total_noptims = noptims_response + noptims_intervention

        return (
            total_loss/total_noptims if total_noptims else total_loss,
            l_response/noptims_response if noptims_response else l_response,
            l_intervention/noptims_intervention if noptims_intervention else l_intervention
        )

    @torch.no_grad()
    def _test_one_pass(self, model_1, model_2, es_1, es_2, loader_forward, loader_backward, thresholds) -> Dict[str, float]:
        model_1.eval()
        model_2.eval()

        if not es_1.is_stopped:
            real_y = []
            score_y = []
            if self.use_forward_data:
                for data in loader_forward:
                    out, _ = model_1(torch.concat([self.fabric.to_device(data.healthy.view(-1, 1)), self.fabric.to_device(data.mutations.view(-1, 1))], 1), self.fabric.to_device(data.batch), binarize_intervention=False, threshold_input=thresholds["healthy"])
                    real_y += data.diseased.detach().cpu().tolist()
                    score_y += out[:, -1].detach().cpu().tolist()

            if self.use_backward_data:
                for data in loader_backward:
                    out, _ = model_1(torch.concat([self.fabric.to_device(data.diseased.view(-1, 1)), self.fabric.to_device(data.intervention.view(-1, 1))], 1), self.fabric.to_device(data.batch), mutilate_mutations=self.fabric.to_device(data.mutations), binarize_intervention=False, threshold_input=thresholds["diseased"])
                    real_y += data.treated.detach().cpu().tolist()
                    score_y += out[:, -1].detach().cpu().tolist()

            forward_mae = mean_absolute_error(real_y, score_y)
            forward_mse = mean_squared_error(real_y, score_y)
            # forward_r2 = r2_score(real_y, score_y)

            # linear model (scGen style)
            real_ys = np.array(real_y).reshape(-1, int(data.num_nodes / len(torch.unique(data.batch))))
            score_ys = np.array(score_y).reshape(-1, int(data.num_nodes / len(torch.unique(data.batch))))
            x = np.mean(score_ys, 0).ravel()
            y = np.mean(real_ys, 0).ravel()
            forward_r_value = linregress(x, y).rvalue
            forward_r2_value = forward_r_value**2

            forward_spearman = []
            forward_pearson = []
            for ry, sy in zip(real_ys, score_ys):
                forward_spearman.append(spearmanr(ry, sy).correlation)
                forward_pearson.append(pearsonr(ry, sy).statistic)
            forward_spearman = np.mean(forward_spearman)
            forward_pearson = np.mean(forward_pearson)
            forward_r2 = forward_pearson**2
        else:
            forward_mae = -1
            forward_mse = -1
            forward_r2 = -1
            forward_r2_value = -1
            forward_spearman = -1

        if not es_2.is_stopped:
            real_y = []
            score_y = []
            top_ks = []
            perturbagens = []

            for data in loader_backward:
                perturbagens += data.perturbagen_name
                num_nodes = int(data.num_nodes / len(torch.unique(data.batch)))
                # predicting interventions
                out = model_2(torch.concat([self.fabric.to_device(data.diseased.view(-1, 1)), self.fabric.to_device(data.treated.view(-1, 1))], 1), self.fabric.to_device(data.batch), mutilate_mutations=self.fabric.to_device(data.mutations), threshold_input=thresholds)

                # measure accuracy predicted U'
                where_intervention = torch.where(data.intervention.detach().cpu().view(-1, num_nodes))
                correct_interventions = tuple(zip(where_intervention[0].tolist(), where_intervention[1].tolist()))
                prepare_out = out.detach().cpu().view(-1, num_nodes)
                for (row, col) in correct_interventions:
                    top_ks.append(torch.where(torch.argsort(prepare_out[row, :], descending=True) == col)[0].item())

                # response prediction
                topK = torch.sum(data.intervention.view(-1, int(data.num_nodes / len(torch.unique(data.batch)))), 1)
                out, in_x_binarized = model_1(torch.concat([self.fabric.to_device(data.diseased.view(-1, 1)), out], 1), self.fabric.to_device(data.batch), mutilate_mutations=self.fabric.to_device(data.mutations), threshold_input=thresholds["diseased"], binarize_intervention=True, topK=topK)

                real_y += data.treated.detach().cpu().tolist()
                score_y += out[:, -1].detach().cpu().tolist()

            avg_topk = np.mean(top_ks)
            # performance metrics
            backward_mae = mean_absolute_error(real_y, score_y)
            backward_mse = mean_squared_error(real_y, score_y)

            # linear model (scGen style)
            real_ys = np.array(real_y).reshape(-1, int(data.num_nodes / len(torch.unique(data.batch))))
            score_ys = np.array(score_y).reshape(-1, int(data.num_nodes / len(torch.unique(data.batch))))

            # compute R-value perturbagen-wise and then aggregate
            backward_r2_values = []
            backward_spearman = []
            backward_pearson = []
            for perturbagen in set(perturbagens):
                sample_indices = [i == perturbagen for i in perturbagens]
                x = np.mean(score_ys[sample_indices, :], 0).ravel()
                y = np.mean(real_ys[sample_indices, :], 0).ravel()
                backward_r_value = linregress(x, y).rvalue
                backward_r2_values.append(backward_r_value**2)
            for ry, sy in zip(real_ys, score_ys):
                backward_spearman.append(spearmanr(ry, sy).correlation)
                backward_pearson.append(pearsonr(ry, sy).statistic)
            backward_r2_value = np.mean(backward_r2_values)
            backward_spearman = np.mean(backward_spearman)
            backward_pearson = np.mean(backward_pearson)
            backward_r2 = backward_pearson**2
        else:
            backward_mae = -1
            backward_mse = -1
            backward_r2 = -1
            backward_r2_value = -1
            backward_spearman = -1
            avg_topk = -1

        return {
            'forward_mae': forward_mae,
            'forward_mse': forward_mse,
            'forward_r2': forward_r2,
            'forward_r2_scgen': forward_r2_value,
            'forward_spearman': forward_spearman,
            'backward_mae': backward_mae,
            'backward_mse': backward_mse,
            'backward_r2': backward_r2,
            'backward_r2_scgen': backward_r2_value,
            'backward_spearman': backward_spearman,
            'backward_avg_topk': avg_topk
        }

    def _configure_model_with_optimizers_and_schedulers(self, model: PDGrapher) -> Tuple[_FabricModule, _FabricModule]:
        (optimizer_1, optimizer_2), (scheduler_1, scheduler_2) = model.get_optimizers_and_schedulers()

        # Setup optimizers zero_grad() and step() functions
        if isinstance(optimizer_1, list): # we have multiple optimizers for response prediction model
            model_1, optimizer_1 = self.fabric.setup(model.response_prediction, *optimizer_1)
            self._op1_zero_grad = lambda: [op1.zero_grad() for op1 in optimizer_1]
            self._op1_step = lambda: [op1.step() for op1 in optimizer_1]
        else: # we have one optimizer for response prediction model
            model_1, optimizer_1 = self.fabric.setup(model.response_prediction, optimizer_1)
            self._op1_zero_grad = lambda: optimizer_1.zero_grad()
            self._op1_step = lambda: optimizer_1.step()
        if isinstance(optimizer_2, list): # we have multiple optimizers for perturbation discovery model
            model_2, optimizer_2 = self.fabric.setup(model.perturbation_discovery, *optimizer_2)
            self._op2_zero_grad = lambda: [op2.zero_grad() for op2 in optimizer_2]
            self._op2_step = lambda: [op2.step() for op2 in optimizer_2]
        else: # we have one optimizer for perturbation discovery model
            model_2, optimizer_2 = self.fabric.setup(model.perturbation_discovery, optimizer_2)
            self._op2_zero_grad = lambda: optimizer_2.zero_grad()
            self._op2_step = lambda: optimizer_2.step()

        # Setup schedulers step() function
        if self.use_lr_scheduler and scheduler_1 is not None:
            if isinstance(scheduler_1, list):
                self._sc1_step = lambda: [sc1.step() for sc1 in scheduler_1]
            else:
                self._sc1_step = lambda: scheduler_1.step()
        else:
            self._sc1_step = lambda: None
        if self.use_lr_scheduler and scheduler_2 is not None:
            if isinstance(scheduler_2, list):
                self._sc2_step = lambda: [sc2.step() for sc2 in scheduler_2]
            else:
                self._sc2_step = lambda: scheduler_2.step()
        else:
            self._sc2_step = lambda: None

        return model_1, model_2

    def _freeze_model(self, model) -> None:
        for param in model.parameters():
            param.requires_grad = False

    def _unfreeze_model(self, model) -> None:
        for param in model.parameters():
            param.requires_grad = True

    def _test_to_str(self, perf: Dict[str, float], kind: str) -> str:
        return (
            f" | {kind} - FORWARD: MSE: {perf['forward_mse']:.4f}, MAE: {perf['forward_mae']:.4f}, "
            f"R2: {perf['forward_r2']:.4f}, R2 scgen: {perf['forward_r2_scgen']:.4f}, "
            f"Spearman: {perf['forward_spearman']:.4f} | {kind} - BACKWARD: MSE: {perf['backward_mse']:.4f}, "
            f"MAE: {perf['backward_mae']:.4f}, R2: {perf['backward_r2']:.4f}, "
            f"R2 scgen: {perf['backward_r2_scgen']:.4f}, Spearman: {perf['backward_spearman']:.4f}, TopK: {perf['backward_avg_topk']:.4f}"
        )

    def _test_to_writer(self, perf: Dict[str, float], kind: str, epoch: int) -> None:
        for k, v in perf.items():
            pre, suf = k.split("_", 1)
            self.writer.add_scalar(f"{pre}/{kind}/{suf}", v, epoch)
