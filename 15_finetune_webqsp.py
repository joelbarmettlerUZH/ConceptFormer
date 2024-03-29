import logging
import multiprocessing
import random
from dataclasses import asdict
from pathlib import Path

import numpy
import torch
import wandb
from torch.utils.data import DataLoader

from src.BatchSampler.DynamicNeighbourBatchSampler import DynamicNeighbourBatchSampler
from src.DataLoaders.RExEmbeddingDynamicLoader import RExEmbeddingDynamicLoader
from src.Datasets.factory import web_qsp_factory, web_qsp_finetune_factory
from src.GraphAligner.BigGraphAligner import BigGraphAligner
from src.Training.wandb import init_or_recover_wandb
from src.Config.train_sentences_config import TrainSentencesConfig, gpt2_n_neighbors_search_dynamic
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


def main(config: TrainSentencesConfig, gpu=0):
    device = torch.device(f"cuda:{gpu}")

    config.batch_size = 4

    llm = llm_factory(
        config.embedding_llm_type,
        config.embedding_llm_name,
        batch_size=config.batch_size,
        device=device,
        bits=config.quanization
    )

    if not config.trained_path.exists():
        logging.info("Model not initially trained")
        return

    if config.finetune_path("WebQSP").exists():
        logging.info("Skipping already fine-tuned model")
        return

    run, run_completed = init_or_recover_wandb(
        "university-of-zurich",
        "15_finetune_webqsp",
        config.finetune_model_name("WebQSP"),
        asdict(config),
        relevant_keys=[
            'embedding_llm_name',
            'graph_dataset_name',
            'pretrain_dataset_name',
            'train_dataset_name',
            'num_pseudo_words',
            'number_of_neighbors',
            'model_layer_depth',
            'model_layer_width_multiplier',
            'batch_size',
        ],
    )

    if run_completed:
        logging.info("Skipping completed run")
        wandb.finish()
        return

    print("running", config)

    train_data, validation_data, train_graphs = web_qsp_finetune_factory()
    test_data, test_graphs = web_qsp_factory()

    train_graph_aligner = BigGraphAligner(llm, train_graphs, dataset_name='WebQSPFinetuneStar', use_untrained=config.ignore_global_alignment)
    test_graph_aligner = BigGraphAligner(llm, test_graphs, dataset_name='WebQSPStar', use_untrained=config.ignore_global_alignment)

    train_dataset = RExEmbeddingDynamicLoader(train_data, train_graphs, graph_aligner=train_graph_aligner, num_neighbors=10_000)
    validation_dataset = RExEmbeddingDynamicLoader(validation_data, train_graphs, graph_aligner=train_graph_aligner, num_neighbors=10_000)
    test_dataset = RExEmbeddingDynamicLoader(test_data, test_graphs, graph_aligner=test_graph_aligner, num_neighbors=10_000)

    train_sampler = DynamicNeighbourBatchSampler(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    validation_sampler = DynamicNeighbourBatchSampler(validation_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)
    test_sampler = DynamicNeighbourBatchSampler(test_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    validation_loader = DataLoader(validation_dataset, batch_sampler=validation_sampler)
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler)

    run.config.update({
        "train_sentences": len(train_dataset),
        "test_sentences": len(validation_dataset),
    })

    graph_embedder = GraphAttentionEmbedder.from_config(config, llm)
    graph_embedder.load_state_dict(torch.load(config.trained_path, map_location=f'cuda:{gpu}'))
    graph_embedder = graph_embedder.to(device)
    graph_embedder.train()

    model = SentenceTrainer(llm, graph_embedder, replace_subject=config.replace_subject)
    model.to(device)

    loss_function = torch.nn.CrossEntropyLoss()

    model = train(
        train_loader=train_loader,
        val_loader=validation_loader,
        device=device,
        llm=llm,
        model=model,
        loss_function=loss_function,
        checkpoint_dir=config.finetune_checkpoint_directory("WebQSP"),
        learning_rate=config.learning_rate,
        epochs=config.number_of_epochs,
        patience=config.patience,
        batch_size=config.batch_size,
        number_of_neighbors=config.number_of_neighbors,
        run=run,
    )

    test_loss = test(
        test_loader=test_loader,
        device=device,
        llm=llm,
        model=model,
        loss_function=loss_function,
        batch_size=config.batch_size,
        number_of_neighbors=config.number_of_neighbors,
        run=run,
    )

    config.finetune_path("WebQSP").parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.graph_embedder.state_dict(), config.finetune_path("WebQSP"))

    best_model = wandb.Artifact(f"model-{config.finetune_model_name('WebQSP')}", type="model")
    best_model.add_file(config.trained_path)
    run.log_artifact(best_model)

    # Link the model to the Model Registry
    run.link_artifact(best_model, f"15_finetune_webqsp/{config.finetune_model_name('WebQSP')}")

    evaluate(test_loader, model, k=50, run=run)

    wandb.finish()


def process_function(config_queue, gpu):
    while True:
        try:
            config = config_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break

        print(f"Processing config on GPU {gpu}")
        main(config, gpu)


if __name__ == "__main__":
    gpus = [7]
    configs = gpt2_n_neighbors_search_dynamic

    # Create a multiprocessing queue and add all configurations to it
    config_queue = multiprocessing.Queue()
    for config in configs:
        config_queue.put(config)

    processes = []
    for gpu in gpus:
        p = multiprocessing.Process(target=process_function, args=(config_queue, gpu))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
