# ConceptFormer: Towards Finding the Essence of Everything in Large Language Models

This repository contains the source code corresponding to the master thesis "ConceptFormer: Towards Finding the Essence of Everything in Large Language Models," 
including all code to generate the required datasets (T-Rex Bite, Tri-Rex, T-Rex Star), the model architecture, the code to train and evaluate the model,
as well as all scripts to generate the plots used in the final thesis.

The repository employs some root-level scripts that can be executed to replicate the results shown in the ConceptFormer thesis from scratch.

## 0. Setup

Running this code requires Python 3.11 (lower or higher versions probably won't work), as well as sufficient hardware (RTX 4090 or better, at least 24GB VRAM is required, and 500+ GB of CPU RAM). 
To get started, create a new virtual environment and install the dependencies using

```shell
pip install -r requirements.txt
```

## 1. Generate T-Rex 

To get started, we need the T-Rex dataset in the Huggingface Dataset format. Running:

```shell
python3 1_generate_TREx.py
```

generates the T-Rex Dataset class. You can find the corresponding implementation
in the *src/Datasets/Trex.py* folder.

Note that all datasets come in two forms: A regular one (e.g., T-Rex) and a Lite version (e.g., T-Rex Lite).

The lite versions are subsets of the regular datasets and are useful for quick experimentation.

## 2. Generate T-Rex Star

Next, we extract all relevant entities from the T-Rex dataset, query Wikidata for their local neighborhood, and 
save the resulting subgraphs into a new dataset called "T-Rex Star":

```shell
python3 2_generate_TRExStar.py --dataset_name TRExStarLite --version 1 --edge_limit 1
```

Use the *dataset_name* argument to generate either the lite or full version of T-Rex Star. You can limit the number
of edges contained in the subgraph using the *edge_limit* argument. 

The script spawns 8 parallel workers that query Wikidata. It is not recommended to add more workers, as you
start to get rate-limited or even IP-blocked by Wikidata.

Querying Wikidata for millions of entities takes a while. To speed things up, you can run a local instance of 
Wikidata using [qEndpoint](https://github.com/the-qa-company/qEndpoint) and change the SPARQL endpoint in the
following source script: *src/GraphQueryEngine/SparqlEngine.py*.

To extract the whole neighborhood and not limit the number of neighbors, simply set the edge_limit to something
like 10,000. 

If you limit the number of neighbors, we prioritize their inclusion based on PageRank, which is automatically
downloaded from AWS.

The output of this stage is saved under *data/artifacts/TRExStarLite*, containing JSON representations of the extracted subgraphs, 
as well as an output TAR that is read and processed by the HF Datasets class.

## 3. Generate T-Rex Bite

Next, we generate the T-Rex Bite dataset, focused on next token prediction tasks:

```shell
python3 3_generate_TRExBite --dataset_name TRExBiteLite --version 1
```

This step is quite fast, as we simply loop over the Wikipedia abstracts provided by T-Rex and extract
sentences based on the defined criteria.

The output of this stage is saved under *data/artifacts/TRExBiteLite*, containing CSVs with the extracted sentences, as well as an output TAR that is read and processed
by the HF Datasets class.

## 4. Generate Tri-REx

Next, we generate the synthetic sentences for the Tri-REx dataset. To do so, we need a locally running LLM, 
such as Mistral-7B. 

Make sure you have [llama.cpp python](https://github.com/abetlen/llama-cpp-python) installed with cuBLAS support,
otherwise, this step will take forever:

```shell
python3 4_generate_TriREx --dataset_name TriRExLite --version 1 --n_sentences 100 --match_threshold 80 --gpu_indices 0 1 2 3 4 5 6 7 --seed 0 --n_processes_per_gpu 2
```

This script starts a Mistral-7B LLM on each of the GPUs defined. Moreover, if you specify n_processes_per_gpu, you can even
load multiple Mistral-7B models onto the same GPU, depending on how much VRAM you have. 

Setting the number of sentences to 100 (equal to the number of neighbors in T-Rex Star) will result in 1 sentence per neighbor.
However, you can limit this number if you want to keep T-Rex Star lighter or only want to generate sentences for the most famous neighbors (e.g., highest PageRank).

The output of this stage is saved under *data/artifacts/TriRExLite*, containing CSVs with the generated sentences, as well as an output TAR that is read and processed
by the HF Datasets class.

## 5. Align Graph (optional)

Next, we can create global graph alignment using PyTorch Big Graph:

```shell
python3 5_align_graph.py --dataset_name TRExStarLite --llm_type gpt-2 --llm_name_or_path gpt2-xl --gpu 1
```

Here, we create initial global alignments of all entities in TRExStar using a model with GPT-2 architecture, specifically
[gpt2-xl](https://huggingface.co/openai-community/gpt2-xl).

Currently, only GPT-2 based or LLama-2 based models are supported for the llm type. However, for the LLM name or path,
you can provide any llm from Huggingface, which will be downloaded and used automatically. However, for this thesis,
only gpt2 was used, hence it is not guaranteed that other models will work as expected.

The output of this stage is stored under *data/artifacts/BigGraphAlignment_v1/TRExStarLite/gpt2*, namely a file
*model_checkpoint/model.v1000.h5* containing the vector embeddings of all entities.

## 6. Evaluate Base Models

To evaluate the performance of all base models, run:

```shell
python3 6_evaluate_basemodel.py --dataset_name TriRExLite --gpu_indices 0 1 2 3 --k 50
```

This script reports the evaluation to [Weights & Biases](https://wandb.ai/), for which you'll have to first log in
through their CLI. All metrics can be seen in the W&B project called "6_eval_basemodel"

*Note: Once created in a later step, you can also evaluate WebQSP here by simply providing the dataset_name WebQSP!*

## 7. Evaluate Graph RAG

To evaluate the performance of all base models with graph textification (text injection), run:

```shell
python3 7_eval_basemodel_textinjection.py --dataset_name TriRExLite --gpu_indices 0 1 2 3 --k 50
```

All outputs are again visible on W&B.

*Note: Once created in a later step, you can also evaluate WebQSP here by simply providing the dataset_name WebQSP!*

## 8. Search Hyperparameters

Next, a hyperparameter search is conducted to find the best model parameters. This search is controlled by W&B using 
their [Sweep](https://docs.wandb.ai/guides/sweeps) feature. Hence, initialize a new Sweep online and then run this
script to give W&B the resources to optimize the parameters:

```shell
python3 8_search_hyperparameters.py
```

## 9. Pretrain

Given the hyperparameters, start the pre-training on Tri-REx as follows:

```shell
python3 9_pretrain.py
```

This script does not support command-line arguments. 
Instead, you have to manually specify a set of experiments that you want to conduct, e.g., a set of model
configs that shall be trained and reported to W&B.

A training config looks as follows:



```python
TrainSentencesConfig(
    prefix="SentenceFormer_DynamicBatches_NoGlobalAlignment",
    learning_rate=0.00006,
    number_of_epochs=3,
    patience=0,
    model_layer_depth=1,
    model_layer_width_multiplier=1.6,
    num_pseudo_words=10,
    replace_subject=False,
    number_of_neighbors=100,
    model_layer_activation="leaky_relu",
    quanization=None,
    batch_size=12,
    embedding_llm_type="gpt-2",
    embedding_llm_name="gpt2",
    graph_dataset_name="TRExStar",
    pretrain_dataset_name="TriREx",
    train_dataset_name="TRExBite",
    pretrained_model_name="",
    trained_model_name="",
    pretrained_path="",
    trained_path="",
    pretrained_checkpoint_directory="",
    trained_checkpoint_directory="",
    ignore_global_alignment=True,
)
```

In the `__main__` method of the pretrain script, specify the indices of the GPUs you want to use, and the list of
training configurations you want to train. Using multiprocessing, GPUs request new configs until all are processed.

All checkpoints and final models are stored locally under *models* and as artifacts on W&B!

See the resulting training statistics on W&B!

## 10. Train

If you run the same config again in the train script, it will load the best model from the pretrain step and 
continues to train it on T-Rex automatically:

```shell
python3 10_train.py
```

All checkpoints and final models are stored locally under *models* and as artifacts on W&B!

Again, results are reported to W&B!

## 11. Analyze Pseudowords (Optional)

Run the pseudoword analysis script with the same configs again to get a qualitative feeling for the trained models.
This script will prompt the LLM with certain tasks, like "Repeat after me: <concept_vector>" or "Summarize for me: <concept_vector>".

```shell
python3 11_analyze_pseudowords.py
```

The resulting text is logged to W&B, giving you insights into how the LLM processes the concept vectors.

However, as GPT-2 0.1B is very limited in its language capabilities, these experiments yielded no interesting results for me, 
hence they are not mentioned in the final thesis.

## 12. Generate WebQSP

This script generates the WebQSPSentences Dataset mentioned in the thesis. Simply run:

```shell
python3 12_generate_WebQSP.py
```

Resulting in a new HF Dataset containing Sentences from WebQSP. Now that you have this dataset, you can revisit
Steps 6 and 7 to evaluate the base model performance on WebQSP.

## 13. Generate WebQSPFinetune

To achieve better performance, we fine-tuned ConceptFormer on WebQSP. The dataset containing the fine-tune
training sentences is a completely different one, called "WebQSPFinetuneSentences," and it is generated here.

```shell
python3 13_generate_WebQSPFinetune.py
```

## 14. Finetune GPT-2 (Optional)

We also experimented with whether the performance of the GPT-2 base model would increase when fine-tuning the model itself
on the question-answer format of the WebQSPSentence dataset. This turned out not to be the case. However, if you
want to fine-tune a GPT-2 base model, you can do so by running:

```shell
python3 14_finetune_gpt2.py
```

The results are reported to W&B!

## 15. Finetune WebQSP

Finally, we fine-tune and evaluate ConceptFormer on WebQSP:

```shell
python3 15_finetune_webqsp.py
```

Results are reported - you guessed it - to W&B!

## 16. Generate Lookup Table

To employ ConceptFormer in a static setting, we can generate lookup tables for the T-Rex Star dataset as follows:

```shell
python3 17_generate_lookup_table.py --gpu_indices 0 1 2 3 --n_processes_per_gpu 4
```

## 17. Evaluate Lookup Table

Finally, to get the results reported in the thesis, you must evaluate the lookup table:

```shell
python3 18_evaluate_lookup_table.py --gpu_indices 0 1 2 3 --n_processes_per_gpu 4
```

The output of this script is stored in *output/ConceptFormer/ConceptFormer-n*, being a separate .npy file for every
entity in T-Rex Star.
You could modify this script to generate a concept vector for every entity in the entire Wikidata, if you'd like.

The results are reported to W&B. These metrics are also displayed as the final result metrics in the thesis.

## Generate Plots

The plots are all generated based on the data logged to W&B! To create them, run the scripts under *plots*; all plots
will be generated as both PNG and PDF files under *plots/output*.

---

Read more [about me](https://joelbarmettler.xyz/) and my [AI research](https://joelbarmettler.xyz/research/) on my Website, or dive deeper into [ConceptFormer](https://joelbarmettler.xyz/research/master-arbeit/). I also have an [AI Podcast on Spotify and Apple Podcasts](https://joelbarmettler.xyz/podcast/) and sometimes give [Keynote Speeches about AI in Zurich, Switzerland](https://joelbarmettler.xyz/auftritte/)
