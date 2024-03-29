import copy
import logging
from pathlib import Path

import torch
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from wandb.sdk.wandb_run import Run

from src.GraphAligner.BigGraphAligner import BigGraphAligner
from src.LLM.LLM import LLM
from src.Model.Trainer.SentenceTrainer import SentenceTrainer


def forward_batch(
        batch,
        device: torch.device,
        llm: LLM,
        model: SentenceTrainer,
        loss_function: _Loss,
        batch_size: int,
        number_of_neighbors: int,
        run: Run,
):
    sentence = batch['sentence']
    subject_boundary_start = batch['subject_boundary_start']
    subject_boundary_end = batch['subject_boundary_end']
    object_boundary_start = batch['object_boundary_start']
    object_boundary_end = batch['object_boundary_end']

    central_node_embedding = batch['central_node_embedding'].to(device, non_blocking=True)
    node_embeddings = batch['node_embeddings'].to(device, non_blocking=True)
    edge_embeddings = batch['edge_embeddings'].to(device, non_blocking=True)

    outputs = model(
        sentence,
        subject_boundary_start,
        subject_boundary_end,
        object_boundary_start,
        object_boundary_end,
        central_node_embedding,
        node_embeddings,
        edge_embeddings,
    )

    target_ids = []
    for sent, object_start, object_end in zip(sentence, object_boundary_start, object_boundary_end):
        source_text = sent[:object_start]
        source_text = source_text.strip()

        target_text = sent[object_start:object_end]
        target_text = f" {target_text.strip()}"

        n_source_tokens = llm.text_to_input_ids(source_text).shape[1] if source_text else 0
        target_token_ids = llm.text_to_input_ids(source_text + target_text)[:, n_source_tokens:]

        target_ids.extend(target_token_ids)

    target = torch.cat(target_ids, dim=0).to(device, non_blocking=True)

    assert outputs.shape == (target.shape[0],
                             llm.tokenizer.vocab_size), f"Output shape mismatch: {target.shape[0]} vs. {llm.tokenizer.vocab_size}"

    return loss_function(outputs, target)


def validate(
        val_loader: DataLoader,
        device: torch.device,
        llm: LLM,
        model: SentenceTrainer,
        loss_function: _Loss,
        batch_size: int,
        number_of_neighbors: int,
        epoch: int,
        run: Run,
):
    val_loss = 0.0
    model.eval()

    with torch.no_grad():
        for batch in val_loader:
            loss = forward_batch(
                batch=batch,
                device=device,
                llm=llm,
                model=model,
                loss_function=loss_function,
                batch_size=batch_size,
                number_of_neighbors=number_of_neighbors,
                run=run,
            )
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    run.log({"validation/loss": avg_val_loss, "epoch": epoch})
    return avg_val_loss


def test(
        test_loader: DataLoader,
        device: torch.device,
        llm: LLM,
        model: SentenceTrainer,
        loss_function: _Loss,
        batch_size: int,
        number_of_neighbors: int,
        run: Run,
):
    test_loss = 0.0
    model.eval()

    with torch.no_grad():
        for batch in test_loader:
            loss = forward_batch(
                batch=batch,
                device=device,
                llm=llm,
                model=model,
                loss_function=loss_function,
                batch_size=batch_size,
                number_of_neighbors=number_of_neighbors,
                run=run,
            )
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")
    run.log({"test/loss": avg_test_loss})

    run.summary['test/loss'] = avg_test_loss
    return test_loss


def train(
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        llm: LLM,
        model: SentenceTrainer,
        loss_function: _Loss,
        checkpoint_dir: Path | None,
        learning_rate: float,
        epochs: int,
        patience: int,
        batch_size: int,
        number_of_neighbors: int,
        run: Run,
        scheduler_step_size=10,
        scheduler_gamma=0.1,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of training sentences", len(train_loader))
    print("Number of model parameters", params)

    run.config.update({"params": params})

    # Load epoch checkpoint
    start_epoch = 0
    if checkpoint_dir:
        for epoch in range(epochs):
            epoch_file = checkpoint_dir / f"epoch_{epoch}.pth"
            if epoch_file.exists():
                logging.info(f"Loading checkpoint: epoch {epoch}")
                start_epoch = epoch + 1
                model.graph_embedder.load_state_dict(torch.load(epoch_file, map_location=f'cuda:{device.index}'))

        for _ in range(start_epoch):
            scheduler.step()

    # Training Loop
    best_val_loss = float('inf')

    best_model_state = copy.deepcopy(model.state_dict())
    counter = 0  # Counter for early stopping

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader):
            loss = forward_batch(
                batch=batch,
                device=device,
                llm=llm,
                model=model,
                loss_function=loss_function,
                batch_size=batch_size,
                number_of_neighbors=number_of_neighbors,
                run=run,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        run.log({"train/loss": avg_train_loss, "epoch": epoch})

        # Validation
        val_loss = validate(
            val_loader=val_loader,
            device=device,
            llm=llm,
            model=model,
            loss_function=loss_function,
            batch_size=batch_size,
            number_of_neighbors=number_of_neighbors,
            epoch=epoch,
            run=run,
        )

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            counter += 1
            if counter >= patience:
                run.tags = run.tags + ("early_stop",)

                print("Early stopping triggered.")
                break

        if checkpoint_dir:
            new_epoch_file = checkpoint_dir / f"epoch_{epoch}.pth"
            new_epoch_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.graph_embedder.state_dict(), new_epoch_file)

            old_epoch_file = checkpoint_dir / f"epoch_{epoch - 1}.pth"
            if old_epoch_file.exists():
                old_epoch_file.unlink()

        scheduler.step()  # Step the learning rate scheduler

    model.load_state_dict(best_model_state)
    return model


def evaluate(
        test_loader: DataLoader,
        model: SentenceTrainer,
        k: int,
        run: Run,
):
    return model.evaluate(test_loader.dataset, k, run=run)


def generate_lookup_table(
        graph_aligner: BigGraphAligner,
        model: SentenceTrainer,
        output_directory: Path,
        num_pseudo_words: int,
):
    publish_directory = output_directory / f"{num_pseudo_words}_context_vectors"
    publish_directory.mkdir(parents=True, exist_ok=True)

    for subject_id, G in tqdm(graph_aligner.graphs.items(), desc="SentenceFormer Lookup Table"):
        subject_file_path = publish_directory / f"{subject_id}.npy"

        if subject_file_path.exists() or subject_id not in graph_aligner.entity_index:
            logging.info(f"Embedding for subject_id {subject_id} already exists. Skipping...")
            continue

        central_node_embedding = graph_aligner.node_embedding_batch([subject_id]).unsqueeze(0)

        neighbour_ids, edge_ids, ranks = [], [], []
        for central_node_id, neighbour_node_id, edge in G.edges(data=True):
            edge_ids.append(edge['id'])
            neighbour_ids.append(neighbour_node_id)
            ranks.append(G.nodes[neighbour_node_id]['rank'])

        if len(neighbour_ids) == 0:
            continue

        combined = zip(neighbour_ids, edge_ids, ranks)
        sorted_combined = sorted(combined, key=lambda x: x[2], reverse=True)
        neighbour_ids, edge_ids, _ = zip(*sorted_combined)
        neighbour_ids, edge_ids = list(neighbour_ids), list(edge_ids)

        node_embeddings = graph_aligner.node_embedding_batch(neighbour_ids).unsqueeze(0)
        edge_embeddings = graph_aligner.edge_embedding_batch(edge_ids).unsqueeze(0)

        graph_embeddings = graph_embedder(central_node_embedding, node_embeddings, edge_embeddings)

        # Save the embedding as a numpy array
        np.save(subject_file_path, graph_embeddings.cpu().detach().numpy())
