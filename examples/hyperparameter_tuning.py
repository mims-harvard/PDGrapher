from ray import tune, air
from ray.air import session
from ray.tune.search.optuna import OptunaSearch
import torch
import torch.optim as optim

from pdgrapher import Dataset, PDGrapher, Trainer
from pdgrapher._utils import DummyEarlyStopping, get_thresholds, calculate_loss_sample_weights
from pdgrapher._models import GCNArgs


def main():
    dataset = Dataset(
        forward_path="data/rep-learning-approach-3/processed/real_lognorm/data_forward_A549.pt",
        backward_path="data/rep-learning-approach-3/processed/real_lognorm/data_backward_A549.pt",
        splits_path="data/splits/genetic/A549/random/1fold/splits.pt"
    )
    edge_index = torch.load("data/rep-learning-approach-3/processed/real_lognorm/edge_index_A549.pt")

    trainer = Trainer(
        fabric_kwargs={"accelerator": "cuda"},
        use_forward_data=True, use_backward_data=True, use_supervision=True, use_intervention_data=True
    )

    # Define objective function
    def objective(config, dataset, edge_index, trainer):
        # `config` is a dict[any, any]

        # Get training and testing datasets (this is also part of trainer.train
        # function, but we need its functionallity here)
        # Apparently setting all of this here is better for memory management...?
        sample_weights_model_2_backward = calculate_loss_sample_weights(dataset.train_dataset_backward, "intervention")
        sample_weights_model_2_backward = trainer.fabric.to_device(sample_weights_model_2_backward)
        pos_weight = sample_weights_model_2_backward[1] / sample_weights_model_2_backward[0]
        thresholds = get_thresholds(dataset)
        thresholds = {k: trainer.fabric.to_device(v) for k, v in thresholds.items()}
        edge_index = trainer.fabric.to_device(edge_index)
        (
            train_loader_forward, train_loader_backward,
            val_loader_forward, val_loader_backward,
            test_loader_forward, test_loader_backward
        ) = trainer.fabric.setup_dataloaders(*dataset.get_dataloaders())
        es_1 = DummyEarlyStopping()
        es_2 = DummyEarlyStopping()

        # Define the model from config and its optimizers
        model_kwargs = GCNArgs.from_dict(config).to_dict() # get values without removing them
        model_kwargs["num_vars"] = dataset.get_num_vars()
        model = PDGrapher(edge_index, model_kwargs=model_kwargs)
        op_rp = optim.Adam(model.response_prediction.parameters(), lr=config["rp_lr"])
        op_pd = optim.Adam(model.perturbation_discovery.parameters(), lr=config["pd_lr"])
        model.set_optimizers_and_schedulers([op_rp, op_pd])

        model_1, model_2 = trainer._configure_model_with_optimizers_and_schedulers(model)

        # Define supervision_multiplier
        trainer.supervision_multiplier = config["supervision_multiplier"]

        # Manual training loop from Trainer
        while True:
            loss, loss_f, loss_b = trainer._train_one_pass(
                model_1, model_2, es_1, es_2, train_loader_forward, train_loader_backward,
                thresholds, pos_weight)
            val_loss, val_loss_f, val_loss_b = trainer._val_one_pass(
                model_1, model_2, es_1, es_2, val_loader_forward, val_loader_backward,
                thresholds, pos_weight)
            session.report({"train_loss": loss, "train_loss_forward": loss_f, "train_loss_backward": loss_b,
                            "val_loss": val_loss, "val_loss_forward": val_loss_f, "val_loss_backward": val_loss_b})
    
    # Define hyperparameter search space
    search_space = {
        # Learning rates
        "rp_lr": tune.loguniform(1e-6, 1e-1),
        "rp_lr": tune.loguniform(1e-6, 1e-1),
        # Supervision loss multiplier
        "supervision_multiplier": tune.loguniform(1e-4, 1),
        # Model architecture
        "positional_features_dims": tune.choice([8, 16, 32, 48, 64, 96, 128]),
        "embedding_layer_dim": tune.choice([8, 16, 32, 48, 64, 96, 128]),
        "dim_gnn": tune.choice([8, 16, 32, 48, 64, 96, 128]),
        "n_layers_gnn": tune.choice([1, 2, 3, 4, 5]),
        "n_layers_nn": tune.choice([1, 2, 3, 4, 5]),
    }
    algo = OptunaSearch()

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(objective, dataset=dataset, edge_index=edge_index, trainer=trainer),
            resources={"cpu": 1, "gpu": 1},
        ),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            search_alg=algo,
            num_samples=100,
            max_concurrent_trials=1
        ),
        run_config=air.RunConfig(
            name="PDGrapher_tuning",
            storage_path="examples/PDGrapher",
            stop={"training_iteration": 1},
        ),
    )

    results = tuner.fit()
    print("Best config is:", results.get_best_result().config)


if __name__ == "__main__":
    main()