from typing import Any, List, Dict, Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from ._models import GCNArgs, ResponsePredictionModel, PerturbationDiscoveryModel, ResponsePredictionModelOld, PerturbationDiscoveryModelOld
from ._utils import _test_condition

__all__ = ["PDGrapher"]


class PDGrapherOld:
    """
    PDGrapher is a model that predicts therapeutic perturbagens.
    It consists of two submodels: Response Prediction Model (RP) and Perturbation Discovery Model (PD).
    """

    def __init__(self, edge_index: torch.Tensor, *, model_kwargs: Dict[str, Any] = {},
                 response_kwargs: Dict[str, Any] = {}, perturbation_kwargs: Dict[str, Any] = {}) -> None:
        """
        Initialization for PDGrapher.

        Args:
            edge_index (torch.Tensor): _description_
            model_kwargs (dict[str, Any]): Arguments for both models. They are
            overwritten if some arguments are provided in response_kwargs or
            perturbation_kwargs for that model. See bellow for all possible
            key:value pairs. Defaults to {}.
            response_kwargs (dict[str, Any]): Arguments for Response Prediction
            Model. See bellow for all possible key:value pairs. Defaults to {}.
            perturbation_kwargs (dict[str, Any]): Arguments for Perturbation
            Discovery Model. See bellow for all possible key:value pairs.
            Defaults to {}.
        Next arguments apply to model_kwargs, response_kwargs, and perturbation_kwargs.
            positional_features_dims (int): _description_. Defaults to 16.
            embedding_layer_dim (int): _description_. Defaults to 16.
            dim_gnn (int): _description_. Defaults to 16.
            out_channels (int): _description_. Defaults to 1.
            num_vars (int): _description_. Defaults to 1.
            n_layers_gnn (int): _description_. Defaults to 1.
            n_layers_nn (int): _description_. Defaults to 2.
            train (bool): Whether to train this model. Defaults to True.
        """

        # import pdb; pdb.set_trace()
        # Populate response_kwargs and perturbation_kwargs with default args from
        # model_args
        response_kwargs = {**model_kwargs, **response_kwargs}
        perturbation_kwargs = {**model_kwargs, **perturbation_kwargs}

        # Pop kwargs related to response_prediction and perturbation_discovery
        # modules
        self._train_response_prediction = response_kwargs.pop("train", True)
        self._train_perturbation_discovery = perturbation_kwargs.pop("train", True)

        rp_args = GCNArgs.from_dict(response_kwargs)
        pd_args = GCNArgs.from_dict(perturbation_kwargs)

        # Models
        self.response_prediction: nn.Module = ResponsePredictionModelOld(rp_args, edge_index)
        self.perturbation_discovery: nn.Module = PerturbationDiscoveryModelOld(pd_args, edge_index)

        # Optimizers & Schedulers
        # we use __* to set these "private"
        self.__optimizer_response_prediction = optim.Adam(self.response_prediction.parameters(), lr=0.01)
        self.__optimizer_perturbation_discovery = optim.Adam(self.perturbation_discovery.parameters(), lr=0.01)
        self.__scheduler_response_prediction = lr_scheduler.StepLR(
            self.__optimizer_response_prediction, step_size=350, gamma=0.1)
        self.__scheduler_perturbation_discovery = lr_scheduler.StepLR(
            self.__optimizer_perturbation_discovery, step_size=1500, gamma=0.1)


    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError()


    def get_optimizers_and_schedulers(self):
    # -> Tuple[
    #     List[Union[optim.Optimizer, List[optim.Optimizer]]],
    #     List[Optional[Union[lr_scheduler.LRScheduler, List[lr_scheduler.LRScheduler]]]]
    # ]:
        """
        Returns all optimizers and learning rate schedulers.

        Returns:
            tuple[list[Optimizer | list[Optimizer]],list[None | LRScheduler | list[LRScheduler]]]:
            First element in the tuple is a 2-list of optimizers, at index 0
            there are optimizers for the RP model and at index 1 for the PD
            model. Second element in the tuple is a 2-list of LR schedulers, at
            index 0 there are LR schedulers for optimizers, connected to the RP
            model, and at index 1 there are LR schedulers for optimizers,
            connected to the PD model.
        """
        optimizers = [
            self.__optimizer_response_prediction,
            self.__optimizer_perturbation_discovery
        ]
        schedulers = [
            self.__scheduler_response_prediction,
            self.__scheduler_perturbation_discovery
        ]
        return (optimizers, schedulers)


    def set_optimizers_and_schedulers(
            self, optimizers: List[Union[optim.Optimizer, List[optim.Optimizer]]],
            schedulers = [None, None]
            ) -> None:
        """
        _summary_

        Args:
            optimizers (list[Optimizer, list[Optimizer]]): _description_
            schedulers (list[None, LRScheduler, list[LRScheduler]], optional): _description_. Defaults to [None, None].
        """
        # Check if optimizers len is ok
        _test_condition(isinstance(optimizers, list), "Parameter `optimizers` needs to be a list!")
        _test_condition(len(optimizers) == 2, f"Parameter `optimizers` needs to be a list of length 2, but length {len(optimizers)} was detected!")
        # Check if schedulers len is ok
        _test_condition(isinstance(schedulers, list), "Parameter `schedulers` needs to be a list!")
        _test_condition(len(schedulers) == 2, f"Parameter `schedulers` needs to be a list of length 2, but length {len(schedulers)} was provided!")
        # Check for each optimizer if it is connected to the correct model
        _test_condition(self._check_optimizers(self.response_prediction, self.perturbation_discovery, optimizers[0]), "One of the provided optimizers for the Response Prediction Model has no association with it!")
        _test_condition(self._check_optimizers(self.perturbation_discovery, self.response_prediction, optimizers[1]), "One of the provided optimizers for the Perturbation Discovery Model has no association with it!")
        # Check if each scheduler is connected to a corresponding optimizer
        _test_condition(self._check_schedulers(optimizers[0], schedulers[0]), "One of the provided schedulers for the Response Prediction Model is not connected to any of its optimizers!")
        _test_condition(self._check_schedulers(optimizers[1], schedulers[1]), "One of the provided schedulers for the Perturbation Discovery Model is not connected to any of its optimizers!")

        self.__optimizer_response_prediction = optimizers[0]
        self.__optimizer_perturbation_discovery = optimizers[1]
        self.__scheduler_response_prediction = schedulers[0]
        self.__scheduler_perturbation_discovery = schedulers[1]


    def _check_optimizers(self, correct_model: nn.Module, wrong_model: nn.Module,
                          optimizer: Union[optim.Optimizer, List[optim.Optimizer]]) -> bool:
        # we check for the intersection of the parameters between model and optimizer
        if not optimizer:
            return False
        correct_model_parameters = set(correct_model.parameters())
        wrong_model_parameters = set(wrong_model.parameters())
        if isinstance(optimizer, list):
            for op in optimizer:
                op_parameters = set(p for group in op.param_groups for p in group["params"])
                if not op_parameters.intersection(correct_model_parameters): # no common parameters -> this optimizer does not optimize this model
                    return False
                if op_parameters.intersection(wrong_model_parameters): # common parameters with wrong model
                    return False
            return bool(len(optimizer)) # we have at least one optimizer
        op_parameters = set(p for group in optimizer.param_groups for p in group["params"])
        return bool(op_parameters.intersection(correct_model_parameters)) and not bool(op_parameters.intersection(wrong_model_parameters))


    def _check_schedulers(
            self, optimizer: Union[optim.Optimizer, List[optim.Optimizer]],
            scheduler = None) -> bool:
        if scheduler is None: # using no scheduler is permited
            return True

        if not isinstance(optimizer, list):
            optimizer = [optimizer]

        if isinstance(scheduler, list):
            for sc in scheduler:
                for op in optimizer:
                    if sc.optimizer == op:
                        break
                else: # no break was detected -> this scheduler has no optimizer
                    return False
            return True

        for op in optimizer:
            if scheduler.optimizer == op:
                return True
        return False
