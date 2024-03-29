import logging
import multiprocessing
import random

import numpy
import torch
import wandb

from src.DataLoaders.RExEmbeddingLoader import RExEmbeddingLoader
from src.LLM.factory import llm_factory
from src.Model.GraphAttentionEmbedder.GraphAttentionEmbedder import GraphAttentionEmbedder
from src.Model.Trainer.SentenceTrainer import SentenceTrainer
from src.Training.pytorch import train, test, evaluate

SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
numpy.random.seed(SEED)
random.seed(SEED)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main(gpu: int):
    device = torch.device(f"cuda:{gpu}")

    run = wandb.init(project="8_search_hyperparameters")
    config = wandb.config

    llm = llm_factory(
        config.embedding_llm_type,
        config.embedding_llm_name,
        batch_size=config.batch_size,
        device=device,
        bits=config.quantization,
    )

    train_dataset, val_dataset, test_dataset = RExEmbeddingLoader.from_dataset(
        train_dataset_name=config.pretrain_dataset_name,
        graph_dataset_name=config.graph_dataset_name,
        llm=llm,
        num_neighbors=config.number_of_neighbors,
    )
    train_loader, val_loader, test_loader = RExEmbeddingLoader.to_loader(train_dataset, val_dataset, test_dataset,
                                                                         config.batch_size)

    run.config.update({
        "train_sentences": len(train_dataset),
        "val_sentences": len(val_dataset),
        "test_sentences": len(test_dataset),
    })

    graph_embedder = GraphAttentionEmbedder.from_config(config, llm)
    graph_embedder = graph_embedder.to(device)
    graph_embedder.train()

    # Initialize GraphGPT with the pretrained GCN
    model = SentenceTrainer(llm, graph_embedder, replace_subject=config.replace_subject)
    model.to(device)

    loss_function = torch.nn.CrossEntropyLoss()

    model = train(
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        llm=llm,
        model=model,
        loss_function=loss_function,
        checkpoint_dir=None,
        learning_rate=config.learning_rate,
        epochs=config.number_of_epochs,
        patience=config.patience,
        batch_size=config.batch_size,
        number_of_neighbors=config.number_of_neighbors,
        scheduler_step_size=config.scheduler_step_size,
        scheduler_gamma=config.scheduler_gamma,
        run=run,
    )

    test(
        test_loader=test_loader,
        device=device,
        llm=llm,
        model=model,
        loss_function=loss_function,
        batch_size=config.batch_size,
        number_of_neighbors=config.number_of_neighbors,
        run=run,
    )

    evaluate(test_loader, model, k=50, run=run)


def create_sweep_config(
        graph_dataset_name: str,
        pretrain_dataset_name: str,
        train_dataset_name: str,
):
    return {
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "k10"},
        "name": f"{graph_dataset_name}-{pretrain_dataset_name}",
        "parameters": {
            # Tune
            "learning_rate": {"min": 1e-6, "max": 1e-5},
            "model_layer_depth": {"min": 1, "max": 3},
            "model_layer_width_multiplier": {"min": 0.1, "max": 5.0},

            # Constant
            "scheduler_step_size": {"value": 10},
            "scheduler_gamma": {"value": 0.1},
            "quantization": {"value": 2},
            "number_of_epochs": {"value": 15},
            "patience": {"value": 3},
            "num_pseudo_words": {"value": 10},
            "replace_subject": {"value": True},
            "number_of_neighbors": {"value": 25},
            "model_layer_activation": {"value": "leaky_relu"},
            "batch_size": {"value": 32},
            "embedding_llm_type": {"value": "gpt-2"},
            "embedding_llm_name": {"value": "gpt2"},

            "graph_dataset_name": {"value": graph_dataset_name},
            "pretrain_dataset_name": {"value": pretrain_dataset_name},
            "train_dataset_name": {"value": train_dataset_name},
        },
    }

def start_agent(sweep_id: str, gpu: int):
    wandb.agent(sweep_id, function=lambda: main(gpu), project="8_search_hyperparameters")

if __name__ == "__main__":
    gpus = [0, 1, 2, 3, 4, 5, 6]
    sweep_configuration = create_sweep_config(
        graph_dataset_name="TRExStarLite",
        pretrain_dataset_name="TriRExLite",
        train_dataset_name="TRExBiteLite",
    )

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="8_search_hyperparameters")

    processes = []
    for gpu in gpus:
        p = multiprocessing.Process(target=start_agent, args=(sweep_id, gpu))
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()
