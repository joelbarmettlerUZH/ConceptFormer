import logging
import multiprocessing
import random
from dataclasses import asdict

import numpy
import torch
import wandb

from src.Training.wandb import init_or_recover_wandb
from src.Config.train_sentences_config import TrainSentencesConfig, gpt2_n_neighbors_search_lite, \
    gpt2_n_neighbors_search, gpt2_n_neighbors_search_lite_dynamic, gpt2_n_neighbors_search_dynamic
from src.DataLoaders.RExEmbeddingDynamicLoader import RExEmbeddingDynamicLoader
from src.LLM.factory import llm_factory
from src.Model.GraphAttentionEmbedder.GraphAttentionEmbedder import GraphAttentionEmbedder
from src.Model.Trainer.SentenceTrainer import SentenceTrainer
from src.Training.pytorch import train, test, evaluate, generate_lookup_table

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

    config.batch_size = config.batch_size // 2

    llm = llm_factory(
        config.embedding_llm_type,
        config.embedding_llm_name,
        batch_size=config.batch_size,
        device=device,
        bits=config.quanization
    )

    if config.trained_path.exists():
        logging.info("Skipping completed run")
        return

    if not config.pretrained_path.exists():
        logging.warning("Pretraining not found")
        return

    run, run_completed = init_or_recover_wandb(
        "university-of-zurich",
        "10_train_dynamic_batching_ignore_global_alignment",
        config.trained_model_name,
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
            'replace_subject',
        ],
    )

    if run_completed:
        logging.info("Skipping completed run")
        wandb.finish()
        return

    print("running", config)

    train_dataset, val_dataset, test_dataset = RExEmbeddingDynamicLoader.from_dataset(
        train_dataset_name=config.train_dataset_name,
        graph_dataset_name=config.graph_dataset_name,
        llm=llm,
        num_neighbors=config.number_of_neighbors,
        ignore_global_alignment=config.ignore_global_alignment,
    )
    train_loader, val_loader, test_loader = RExEmbeddingDynamicLoader.to_loader(train_dataset, val_dataset, test_dataset,
                                                                         config.batch_size)

    run.config.update({
        "train_sentences": len(train_dataset),
        "val_sentences": len(val_dataset),
        "test_sentences": len(test_dataset),
    })

    graph_embedder = GraphAttentionEmbedder.from_config(config, llm)
    graph_embedder.load_state_dict(torch.load(config.pretrained_path, map_location=f'cuda:{gpu}'))
    graph_embedder = graph_embedder.to(device)
    graph_embedder.train()

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
        checkpoint_dir=config.trained_checkpoint_directory,
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

    config.trained_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.graph_embedder.state_dict(), config.trained_path)

    best_model = wandb.Artifact(f"model-{config.trained_model_name}", type="model")
    best_model.add_file(config.trained_path)
    run.log_artifact(best_model)

    # Link the model to the Model Registry
    run.link_artifact(best_model, f"10_train_dynamic_batching_ignore_global_alignment/{config.trained_model_name}")

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
