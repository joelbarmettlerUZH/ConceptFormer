import logging
import multiprocessing
from dataclasses import asdict

import torch
import wandb
from tqdm import tqdm

from src.DataLoaders.RExEmbeddingDynamicLoader import RExEmbeddingDynamicLoader
from src.LLM.factory import llm_factory
from src.Model.Trainer.SentenceTrainer import SentenceTrainer
from src.Config.train_sentences_config import gpt2_n_neighbors_search, gpt2_n_neighbors_search_lite, TrainSentencesConfig
from src.Model.GraphAttentionEmbedder.GraphAttentionEmbedder import GraphAttentionEmbedder


def run_test(config: TrainSentencesConfig, test_name, sentence, subject_marker, object_marker, gpu=0, sample_from_top=1):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    device = torch.device(f"cuda:{gpu}")

    if not config.trained_path.exists():
        return

    llm = llm_factory(
        config.embedding_llm_type,
        config.embedding_llm_name,
        batch_size=config.batch_size,
        device=device,
        bits=config.quanization
    )

    train_dataset, _, test_dataset = RExEmbeddingDynamicLoader.from_dataset(
        train_dataset_name=config.pretrain_dataset_name,
        graph_dataset_name=config.graph_dataset_name,
        llm=llm,
        num_neighbors=config.number_of_neighbors,
    )

    graph_embedder = GraphAttentionEmbedder.from_config(config, llm)
    graph_embedder.load_state_dict(torch.load(config.trained_path, map_location=f'cuda:{gpu}'))
    graph_embedder = graph_embedder.to(device)
    graph_embedder.train()

    sentence_trainer = SentenceTrainer(llm, graph_embedder, replace_subject=config.replace_subject)

    test_name_reduced = test_name.lower().replace(" ", "_")
    wandb.init(
        project="11_analyze_pseudowords_sample",
        name=f"{test_name_reduced}_{config.embedding_llm_name}_{config.number_of_neighbors}_to_{config.num_pseudo_words}",

        # track hyperparameters and run metadata
        config={
            "method": test_name,
            "sample_from_top": sample_from_top,
            **asdict(config),
        }
    )
    subject_boundary_start = sentence.index(subject_marker)
    subject_boundary_end = subject_boundary_start + len(subject_marker)

    object_boundary_start = sentence.index(object_marker)
    object_boundary_end = object_boundary_start + len(object_marker)

    examples_table = wandb.Table(columns=["Entity ID", "Entity Label", "Baseline Input", "Baseline Output", "Model Input", "Model Output"])

    processed_ids = []

    # Process data in batches
    limit = 100
    processed = 0
    for i in tqdm(range(len(test_dataset)), desc=f'Evaluating', unit='sentences'):
        if processed >= limit:
            break
        data = train_dataset[i]

        entity_id = data['subject_id']
        entity_label = data['subject_label']

        if entity_id in processed_ids:
            continue
        processed_ids.append(entity_id)

        baseline_sentence = sentence[:object_boundary_start].replace(subject_marker, entity_label).strip()

        central_node_embedding = data['central_node_embedding'].to(device, non_blocking=True).unsqueeze(0)
        node_embeddings = data['node_embeddings'].to(device, non_blocking=True).unsqueeze(0)
        edge_embeddings = data['edge_embeddings'].to(device, non_blocking=True).unsqueeze(0)

        graph_embeddings = sentence_trainer.graph_embedder(central_node_embedding, node_embeddings, edge_embeddings)
        input_embeds, target_token_ids = sentence_trainer.embed_sentence(
            sentence,
            subject_boundary_start,
            subject_boundary_end,
            object_boundary_start,
            object_boundary_end,
            graph_embeddings[0]
        )

        baseline_embeddings = llm.early_embedding(baseline_sentence)
        baseline_sequence = sentence_trainer.llm.predict_sequence(baseline_embeddings, n_tokens=64, stop_tokens=['\n'], sample_from_top=sample_from_top)
        baseline_output = baseline_sequence["output_string"]

        model_sequence = sentence_trainer.llm.predict_sequence(input_embeds, n_tokens=64, stop_tokens=['\n'], sample_from_top=sample_from_top)
        model_output = model_sequence["output_string"]
        examples_table.add_data(entity_id, entity_label, baseline_sentence, baseline_output, sentence, model_output)
        processed += 1

    wandb.log({
        "examples": examples_table,
    })
    wandb.finish()


def title_abstract_test(config, gpu, sample_from_top):
    test_name = "Title and Abstract"

    subject_marker = "<subject>"
    object_marker = "<object>"
    sentence = f"Title: {subject_marker}.\nAbstract: {object_marker}"

    run_test(config, test_name, sentence, subject_marker, object_marker, gpu, sample_from_top=sample_from_top)

def repeat_after_me_test(config, gpu, sample_from_top):
    test_name = "Repeat after me"

    subject_marker = "<subject>"
    object_marker = "<object>"
    sentence = f"Repeat after me! Me: {subject_marker}. You: {object_marker}"

    run_test(config, test_name, sentence, subject_marker, object_marker, gpu, sample_from_top=sample_from_top)

def summary_test(config, gpu, sample_from_top):
    test_name = "Summary"

    subject_marker = "<subject>"
    object_marker = "<object>"
    sentence = f"Summary of {subject_marker}: {object_marker}"

    run_test(config, test_name, sentence, subject_marker, object_marker, gpu, sample_from_top=sample_from_top)

def blank_test(config, gpu, sample_from_top):
    test_name = "Blank"

    subject_marker = "<subject>"
    object_marker = "<object>"
    sentence = f"{subject_marker} {object_marker}"

    run_test(config, test_name, sentence, subject_marker, object_marker, gpu, sample_from_top=sample_from_top)

def is_test(config, gpu, sample_from_top):
    test_name = "Is"

    subject_marker = "<subject>"
    object_marker = "<object>"
    sentence = f"{subject_marker} is {object_marker}"

    run_test(config, test_name, sentence, subject_marker, object_marker, gpu, sample_from_top=sample_from_top)

def process_function(config_queue, gpu):
    while True:
        try:
            config = config_queue.get_nowait()
        except multiprocessing.queues.Empty:
            break

        print(f"Processing config on GPU {gpu}")
        for i in range(1, 6):
            blank_test(config, gpu, sample_from_top=i)
            is_test(config, gpu, sample_from_top=i)
            title_abstract_test(config, gpu, sample_from_top=i)
            repeat_after_me_test(config, gpu, sample_from_top=i)
            summary_test(config, gpu, sample_from_top=i)


if __name__ == "__main__":
    gpus = [7]
    configs = gpt2_n_neighbors_search

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
