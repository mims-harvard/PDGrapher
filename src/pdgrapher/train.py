from os import makedirs, path as osp
from time import perf_counter
from typing import Dict, Tuple

from lightning import Fabric
from lightning.fabric.wrappers import _FabricModule
import numpy as np
from scipy.stats import pearsonr, spearmanr, linregress
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
from torch.nn.functional import mse_loss, binary_cross_entropy_with_logits
#from torch.utils.tensorboard import SummaryWriter

from pdgrapher.datasets import Dataset
from pdgrapher.pdgrapher import PDGrapher
from pdgrapher._utils import get_thresholds, calculate_loss_sample_weights, DummyWriter, EarlyStopping, tictoc


class Trainer:

    def __init__(self, **kwargs):
        # Logger
        self.use_logging = kwargs.pop("log", False)
        self.logging_dir = osp.abspath(kwargs.pop("logging_dir", "examples/PDGrapher")) # default PDGrapher
        makedirs(self.logging_dir, exist_ok=True)
        self.writer = DummyWriter()
        self.log_train = kwargs.pop("log_train", True)
        self.log_test = kwargs.pop("log_test", True)

        # TODO other training parameters
        self.use_forward_data = kwargs.pop("use_forward_data", True)
        self.use_backward_data = kwargs.pop("use_backward_data", False)
        self.use_intervention_data = kwargs.pop("use_intervention_data", True)
        self.use_supervision = kwargs.pop("use_supervision", False)
        self.supervision_multiplier = kwargs.pop("supervision_multiplier", 1)
        self.use_lr_scheduler = kwargs.pop("use_lr_scheduler", False)

        # All remaining kwargs go to Fabric
        self.fabric = Fabric(**kwargs)

        # Placeholder functions for optimizers & schedulers (zero_grad and step)
        self._op1_zero_grad = lambda: None
        self._op1_step = lambda: None
        self._op2_zero_grad = lambda: None
        self._op2_step = lambda: None
        self._sc1_step = lambda: None
        self._sc2_step = lambda: None

        # TODO support kfold?
        # No, this can be done with multiple train calls or multiple Trainer objects

    def train(self, model: PDGrapher, dataset: Dataset, n_epochs: int, **kwargs):
        # Loss weights, thresholds
        sample_weights_model_2_backward = calculate_loss_sample_weights(dataset.train_dataset_backward, "intervention")
        pos_weight = sample_weights_model_2_backward[1] / sample_weights_model_2_backward[0]
        thresholds = get_thresholds(dataset)
        thresholds = {k: self.fabric.to_device(v) for k, v in thresholds.items()} # do we really need them?
        model.response_prediction.edge_index = self.fabric.to_device(model.response_prediction.edge_index)
        model.perturbation_discovery.edge_index = self.fabric.to_device(model.perturbation_discovery.edge_index)

        # Optimizers & Schedulers
        model_1, model_2 = self._configure_model_with_optimizers_and_schedulers(model)

        if self.use_logging:
            # Log model parameters
            with open(osp.join(self.logging_dir, "params.txt"), "w") as log_params:
                log_params.write("Response Prediction Model parameters:\t{}\n".format(sum(p.numel() for p in model_1.parameters())))
                log_params.write("Perturbation Discovery Model parameters:\t{}\n".format(sum(p.numel() for p in model_2.parameters())))
            # Log metrics
            log_metrics = open(osp.join(self.logging_dir, "metrics.txt"), "w")

        # Dataloaders
        (
            train_loader_forward, train_loader_backward,
            val_loader_forward, val_loader_backward,
            test_loader_forward, test_loader_backward
        ) = self.fabric.setup_dataloaders(*dataset.get_dataloaders())

        # Early stopping
        es_1 = EarlyStopping(model=model_1, save_path=osp.join(self.logging_dir, "model_response_prediction.pt"))
        if not model._train_response_prediction:
            es_1.is_stopped = True
        es_2 = EarlyStopping(model=model_2, save_path=osp.join(self.logging_dir, "model_perturbation_discovery.pt"))
        if not model._train_perturbation_discovery:
            es_2.is_stopped = True

        # Train loop
        for epoch in range(1, n_epochs+1):
            start = perf_counter()

            # TRAIN
            loss, loss_f, loss_b = self._train_one_pass(
                model_1, model_2, es_1, es_2, train_loader_forward, train_loader_backward,
                thresholds, pos_weight)

            # VALIDATION
            val_loss_f, val_loss_b = self._val_one_pass(
                model_1, model_2, es_1, es_2, val_loader_forward, val_loader_backward,
                thresholds, pos_weight)

            # Log additional metrics
            summ_train = ""
            if self.log_train:
                tic = perf_counter()
                train_performance = self._test_one_pass(
                    model_1, model_2, es_1, es_2, train_loader_forward, train_loader_backward, thresholds)
                toc = perf_counter()
                print(f"Test call (train dataset): {toc-tic}secs")
                summ_train = self._test_to_str(train_performance, "TRAIN")
                self._test_to_writer(train_performance, "train", epoch)

            summ_test = ""
            if self.log_test:
                tic = perf_counter()
                test_performance = self._test_one_pass(
                    model_1, model_2, es_1, es_2, test_loader_forward, test_loader_backward, thresholds)
                toc = perf_counter()
                print(f"Test call (test dataset): {toc-tic}secs")
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
            summary = f"Epoch: {epoch:03d}, {end-start:.2f} seconds, Loss: {loss:.4f}"
            summary += summ_train + summ_test + "\n"
            print(summary)
            if self.use_logging:
                log_metrics.write(summary)

            # Early stopping
            if not es_1.is_stopped and es_1(val_loss_f):
                print("Early stopping model 1 (response prediction)")
            if not es_2.is_stopped and es_2(val_loss_b):
                print("Early stopping model 2 (intervention discovery)")
            if es_1.is_stopped and es_2.is_stopped:
                break
        
        if self.use_logging:
            log_metrics.close()

        train_perf = self._test_one_pass(model_1, model_2, es_1, es_2, train_loader_forward, train_loader_backward, thresholds)
        test_perf = self._test_one_pass(model_1, model_2, es_1, es_2, test_loader_forward, test_loader_backward, thresholds)

        model_performance = {
            "train": train_perf,
            "test": test_perf
        }

        # Restore best models
        if model._train_response_prediction:
            model.response_prediction = es_1.load_model()
        if model._train_perturbation_discovery:
            model.perturbation_discovery = es_2.load_model()

        return model_performance

    @tictoc("Train call: {:.2f}secs")
    def _train_one_pass(self, model_1, model_2, es_1, es_2, train_loader_forward, train_loader_backward,
                        thresholds, pos_weight) -> Tuple[float, float, float]:
        l_response = 0
        l_intervention = 0
        noptims_response = 0
        noptims_intervention = 0

        # Do we train the response prediction model?
        if not es_1.is_stopped:
            model_1.train()
            if self.use_forward_data:
                for data in train_loader_forward:
                    self._op1_zero_grad() # optimizer_1.zero_grad()
                    output_forward, _ = model_1(torch.concat([data.healthy.view(-1, 1), data.mutations.view(-1, 1)], 1), data.batch, binarize_intervention=False, threshold_input=thresholds["healthy"])
                    loss_forward = mse_loss(output_forward.view(-1), data.diseased)
                    self.fabric.backward(loss_forward)
                    self._op1_step() # optimizer_1.step()
                    self._sc1_step() # if self.use_lr_scheduler: scheduler_1.step()
                    l_response += float(loss_forward)
                noptims_response += len(train_loader_forward)
            if self.use_backward_data:
                for data in train_loader_backward:
                    self._op1_zero_grad() # optimizer_1.zero_grad()
                    output_forward, _ = model_1(torch.concat([data.diseased.view(-1, 1), data.mutations.view(-1, 1)], 1), data.batch, binarize_intervention=False, threshold_input=thresholds["diseased"])
                    loss_forward = mse_loss(output_forward.view(-1), data.treated)
                    self.fabric.backward(loss_forward)
                    self._op1_step() # optimizer_1.step()
                    self._sc1_step() # if self.use_lr_scheduler: scheduler_1.step()
                    l_response += float(loss_forward)
                noptims_response += len(train_loader_backward)

        # Do we train the perturbagen discovery model?
        if not es_2.is_stopped:
            model_1.eval()
            model_2.train()
            if self.use_intervention_data:
                for data in train_loader_backward:
                    self._op2_zero_grad() # optimizer_2.zero_grad()
                    pred_backward_m2 = model_2(torch.concat([data.diseased.view(-1, 1), data.treated.view(-1, 1)], 1), data.batch, mutilate_mutations=data.mutations, threshold_input=thresholds)
                    # prior knowledge: number of perturbations (targets) per drug
                    topK = torch.sum(data.intervention.view(-1, int(data.num_nodes / len(torch.unique(data.batch)))), 1)
                    pred_backward_m1, in_x_binarized = model_1(torch.concat([data.diseased.view(-1, 1), pred_backward_m2], 1), data.batch, mutilate_mutations=data.mutations, threshold_input=thresholds["diseased"], binarize_intervention=True, topK=topK)
                    loss_backward = mse_loss(pred_backward_m1.view(-1), data.treated)
                    # adds supervision
                    if self.use_supervision:
                        loss_backward += self.supervision_multiplier * binary_cross_entropy_with_logits(pred_backward_m2.view(-1), data.intervention, pos_weight=pos_weight)
                    self.fabric.backward(loss_backward)
                    self._op2_step() # optimizer_2.step()
                    self._sc2_step() # if self.use_lr_scheduler: scheduler_2.step()
                    l_intervention += float(loss_backward)
                noptims_intervention += len(train_loader_backward)
            elif self.use_supervision:
                for data in train_loader_backward:
                    # Backward
                    self._op2_zero_grad() # optimizer_2.zero_grad()
                    pred_backward_m2 = model_2(torch.concat([data.diseased.view(-1, 1), data.treated.view(-1, 1)], 1), data.batch, mutilate_mutations=data.mutations, threshold_input=thresholds)
                    loss_backward = self.supervision_multiplier * binary_cross_entropy_with_logits(pred_backward_m2.view(-1), data.intervention, pos_weight=pos_weight)
                    self.fabric.backward(loss_backward)
                    self._op2_step() # optimizer_2.step()
                    self._sc2_step() # if self.use_lr_scheduler: scheduler_2.step()
                    l_intervention += float(loss_backward)
                noptims_intervention += len(train_loader_backward)

        total_loss = l_response + l_intervention
        total_noptims = noptims_response + noptims_intervention

        return (
            total_loss/total_noptims,
            l_response/noptims_response if noptims_response else l_response,
            l_intervention/noptims_intervention if noptims_intervention else l_intervention
        )

    @torch.no_grad()
    @tictoc("Validation call: {:.2f}secs")
    def _val_one_pass(self, model_1, model_2, es_1, es_2, val_loader_forward,
                      val_loader_backward, thresholds, pos_weight) -> Tuple[float, float]:
        l_response = 0
        l_intervention = 0
        noptims_response = 0
        noptims_intervention = 0

        model_1.eval()
        model_2.eval()

        if not es_1.is_stopped:
            if self.use_forward_data:
                for data in val_loader_forward:
                    # Forward
                    # regression loss - learns diseased from healthy and mutations
                    output_forward, _ = model_1(torch.concat([data.healthy.view(-1, 1), data.mutations.view(-1, 1)], 1), data.batch, binarize_intervention=False, threshold_input=thresholds["healthy"])
                    loss_forward = mse_loss(output_forward.view(-1), data.diseased)
                    l_response += float(loss_forward)
                noptims_response += len(val_loader_forward)
            if self.use_backward_data:
                for data in val_loader_backward:
                    out, _ = model_1(torch.concat([data.diseased.view(-1, 1), data.intervention.view(-1, 1)], 1), data.batch, mutilate_mutations=data.mutations, binarize_intervention=False, threshold_input=thresholds["diseased"])
                    loss_forward = mse_loss(out.view(-1), data.treated)
                    l_response += float(loss_forward)
                noptims_response += len(val_loader_backward)

        if not es_2.is_stopped:
            if self.use_interv_data:
                for data in val_loader_backward:
                    # Backward
                    # (1), (2) cycle loss with M_1 frozen
                    pred_backward_m2 = model_2(torch.concat([data.diseased.view(-1, 1), data.treated.view(-1, 1)], 1), data.batch, mutilate_mutations=data.mutations, threshold_input=thresholds)
                    # prior knowledge: number of perturbations (targets) per drug
                    topK = torch.sum(data.intervention.view(-1, int(data.num_nodes / len(torch.unique(data.batch)))), 1)
                    pred_backward_m1, in_x_binarized = model_1(torch.concat([data.diseased.view(-1, 1), pred_backward_m2], 1), data.batch, mutilate_mutations=data.mutations, threshold_input=thresholds["diseased"], binarize_intervention=True, topK=topK)
                    loss_backward = mse_loss(pred_backward_m1.view(-1), data.treated)
                    if self.use_supervision:
                        loss_backward += self.supervision_multiplier * binary_cross_entropy_with_logits(pred_backward_m2.view(-1), data.intervention, pos_weight=pos_weight)
                    l_intervention += float(loss_backward)
                noptims_intervention += len(val_loader_backward)
            # supervision for U'
            elif self.use_supervision:
                for data in val_loader_backward:
                    pred_backward_m2 = model_2(torch.concat([data.diseased.view(-1, 1), data.treated.view(-1, 1)], 1), data.batch, mutilate_mutations=data.mutations, threshold_input=thresholds)
                    loss_backward = self.supervision_multiplier * binary_cross_entropy_with_logits(pred_backward_m2.view(-1), data.intervention, pos_weight=pos_weight)
                    l_intervention += float(loss_backward)
                noptims_intervention += len(val_loader_backward)

        return (
            l_response/noptims_response if noptims_response else l_response,
            l_intervention/noptims_intervention if noptims_intervention else l_intervention
        )

    @torch.no_grad()
    def _test_one_pass(self, model_1, model_2, es_1, es_2, loader_forward, loader_backward, thresholds):
        model_1.eval()
        model_2.eval()

        if not es_1.is_stopped:
            real_y = []
            score_y = []
            inputs = []
            if self.use_forward_data:
                for data in loader_forward:
                    _in = torch.concat([data.healthy.view(-1, 1), data.mutations.view(-1, 1)], 1)
                    out, _ = model_1(_in, data.batch, binarize_intervention=False, threshold_input=thresholds["healthy"])
                    inputs.append(_in.cpu().numpy())
                    real_y += data.diseased.detach().cpu().tolist()
                    score_y += out[:, -1].detach().cpu().tolist()

            if self.use_backward_data:
                for data in loader_backward:
                    _in = torch.concat([data.diseased.view(-1, 1), data.intervention.view(-1, 1)], 1)
                    out, _ = model_1(_in, data.batch, mutilate_mutations=data.mutations, binarize_intervention=False, threshold_input=thresholds["diseased"])
                    inputs.append(_in.cpu().numpy())
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
            for i in range(len(real_ys)):
                forward_spearman.append(spearmanr(real_ys[i, :], score_ys[i, :]).correlation)
                forward_pearson.append(pearsonr(real_ys[i, :], score_ys[i, :]).statistic)
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
            inputs = []
            top_ks = []
            perturbagens = []

            for data in loader_backward:
                perturbagens += data.perturbagen_name
                num_nodes = int(data.num_nodes / len(torch.unique(data.batch)))
                # predicting interventions
                out = model_2(torch.concat([data.diseased.view(-1, 1), data.treated.view(-1, 1)], 1), data.batch, mutilate_mutations=data.mutations, threshold_input=thresholds)
                inputs.append(torch.concat([data.diseased.view(-1, 1), out], 1).cpu().numpy())

                # measure accuracy predicted U'
                correct_interventions = tuple(zip(torch.where(data.intervention.detach().cpu().view(-1, num_nodes))[0].tolist(), torch.where(data.intervention.detach().cpu().view(-1, num_nodes))[1].tolist()))

                for intervention in correct_interventions:
                    row, col = intervention
                    top_ks.append(torch.where(torch.argsort(out.detach().cpu().view(-1, num_nodes)[row, :], descending=True) == col)[0].item())

                # response prediction
                topK = torch.sum(data.intervention.view(-1, int(data.num_nodes / len(torch.unique(data.batch)))), 1)
                out, in_x_binarized = model_1(torch.concat([data.diseased.view(-1, 1), out], 1), data.batch, mutilate_mutations=data.mutations, threshold_input=thresholds["diseased"], binarize_intervention=True, topK=topK)

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
                backward_spearman.append(spearmanr(real_ys[i, :], score_ys[i, :]).correlation)
                backward_pearson.append(pearsonr(real_ys[i, :], score_ys[i, :]).statistic)
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
            'forward_spearman': forward_spearman,
            'forward_mae': forward_mae,
            'forward_mse': forward_mse,
            'forward_r2': forward_r2,
            'forward_r2_scgen': forward_r2_value,
            'backward_spearman': backward_spearman,
            'backward_mae': backward_mae,
            'backward_mse': backward_mse,
            'backward_r2': backward_r2,
            'backward_r2_scgen': backward_r2_value,
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

    def _test_to_str(self, perf: Dict[str, float], kind: str) -> str:
        return (
            f", {kind} - FORWARD: MSE: {perf['forward_mse']:.4f}, MAE: {perf['forward_mae']:.4f}, "
            f"R2: {perf['forward_r2']:.4f}, R2 scgen: {perf['forward_r2_scgen']:.4f}, "
            f"Spearman: {perf['forward_spearman']:.4f}, {kind} - BACKWARD: MSE: {perf['backward_mse']:.4f}, "
            f"MAE: {perf['backward_mae']:.4f}, R2: {perf['backward_r2']:.4f}, "
            f"R2 scgen: {perf['backward_r2_scgen']:.4f}, Spearman: {perf['backward_spearman']:.4f}, TopK: {perf['backward_avg_topk']:.4f}"
        )

    def _test_to_writer(self, perf: Dict[str, float], kind: str, epoch) -> None:
        for k, v in perf.items():
            pre, suf = k.split("_", 1)
            self.writer.add_scalar(f"{pre}/{kind}/{suf}", v, epoch)
