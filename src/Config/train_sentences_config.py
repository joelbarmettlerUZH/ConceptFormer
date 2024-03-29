from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TrainSentencesConfig:
    prefix: str

    learning_rate: float
    quanization: Optional[int]
    batch_size: int
    number_of_epochs: int
    patience: int
    model_layer_depth: int
    model_layer_width_multiplier: float
    num_pseudo_words: int
    replace_subject: bool
    number_of_neighbors: int
    model_layer_activation: str
    embedding_llm_type: str
    embedding_llm_name: str

    graph_dataset_name: str

    pretrain_dataset_name: str
    train_dataset_name: str

    pretrained_path: Path | str
    trained_path: Path | str

    pretrained_model_name: str
    trained_model_name: str

    pretrained_checkpoint_directory: Path | str
    trained_checkpoint_directory: Path | str

    ignore_global_alignment: bool = False

    def __post_init__(self):
        script_directory = Path(__file__).parent
        data_directory = script_directory.parent.parent / "data"
        output_directory = data_directory / "models" / self.prefix / self.embedding_llm_name
        model_name = f'{self.model_layer_depth}x{round(self.model_layer_width_multiplier, 1)}_{self.number_of_neighbors}to{self.num_pseudo_words}'

        model_directory = output_directory / f"{self.pretrain_dataset_name}-{self.train_dataset_name}" / model_name
        pretrained_directory = model_directory / "pretrained"
        trained_directory = model_directory / "trained"

        self.pretrained_checkpoint_directory = pretrained_directory / "checkpoints"
        self.trained_checkpoint_directory = trained_directory / "checkpoints"

        self.pretrained_model_name = f"pretrained-{self.embedding_llm_name}-{self.pretrain_dataset_name}-{model_name}"
        self.trained_model_name = f"trained-{self.embedding_llm_name}-{self.pretrain_dataset_name}-{self.train_dataset_name}-trained-{model_name}"

        self.pretrained_path = pretrained_directory / f"{self.pretrained_model_name}.pth"
        self.trained_path = trained_directory / f"{self.trained_model_name}.pth"

    def finetune_model_name(self, finetune_name: str):
        model_name = f'{self.model_layer_depth}x{round(self.model_layer_width_multiplier, 1)}_{self.number_of_neighbors}to{self.num_pseudo_words}'
        return f"finetuned-{finetune_name}-{self.embedding_llm_name}-{self.pretrain_dataset_name}-{model_name}"

    def finetune_checkpoint_directory(self, finetune_name: str):
        script_directory = Path(__file__).parent
        data_directory = script_directory.parent.parent / "data"
        output_directory = data_directory / "models" / self.prefix / self.embedding_llm_name
        model_name = f'{self.model_layer_depth}x{round(self.model_layer_width_multiplier, 1)}_{self.number_of_neighbors}to{self.num_pseudo_words}'

        model_directory = output_directory / f"{self.pretrain_dataset_name}-{self.train_dataset_name}" / model_name
        finetune_directory = model_directory / "finetune" / finetune_name

        return finetune_directory / "checkpoints"

    def finetune_path(self, finetune_name: str):
        script_directory = Path(__file__).parent
        data_directory = script_directory.parent.parent / "data"
        output_directory = data_directory / "models" / self.prefix / self.embedding_llm_name
        model_name = f'{self.model_layer_depth}x{round(self.model_layer_width_multiplier, 1)}_{self.number_of_neighbors}to{self.num_pseudo_words}'

        model_directory = output_directory / f"{self.pretrain_dataset_name}-{self.train_dataset_name}" / model_name
        finetune_directory = model_directory / "finetune" / finetune_name

        return finetune_directory / f"{self.trained_model_name}_finetune.pth"


# LITE GPT2 NEIGHBOUR SEARCH
gpt2_n_neighbors_search_lite_nested = [
    [
        TrainSentencesConfig(
            prefix="SentenceFormer_BatchUpsampling",
            learning_rate=0.00006,
            number_of_epochs=50,
            patience=5,
            model_layer_depth=1,
            model_layer_width_multiplier=1.6,
            num_pseudo_words=num_pseudo_words,
            replace_subject=True,
            number_of_neighbors=num_neighbors,
            model_layer_activation="leaky_relu",
            quanization=None,
            batch_size=32,
            embedding_llm_type="gpt-2",
            embedding_llm_name="gpt2",
            graph_dataset_name="TRExStarLite",
            pretrain_dataset_name="TriRExLite",
            train_dataset_name="TRExBiteLite",
            pretrained_model_name="",
            trained_model_name="",
            pretrained_path="",
            trained_path="",
            pretrained_checkpoint_directory="",
            trained_checkpoint_directory="",
        ) for num_neighbors in [5, 10, 15, 25, 50, 100]
    ] for num_pseudo_words in [1, 2, 3, 4, 5, 10, 15, 25]
]
gpt2_n_neighbors_search_lite = [item for sublist in gpt2_n_neighbors_search_lite_nested for item in sublist]

# FULL GPT2 NEIGHBOUR SEARCH
gpt2_n_neighbors_search_nested = [
    [
        TrainSentencesConfig(
            prefix="SentenceFormer_BatchUpsampling",
            learning_rate=0.00006,
            number_of_epochs=50,
            patience=5,
            model_layer_depth=1,
            model_layer_width_multiplier=1.6,
            num_pseudo_words=num_pseudo_words,
            replace_subject=True,
            number_of_neighbors=num_neighbors,
            model_layer_activation="leaky_relu",
            quanization=None,
            batch_size=32,
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
        ) for num_neighbors in [5, 10, 15, 25, 50, 100]
    ] for num_pseudo_words in [1, 2, 3, 4, 5, 10, 15, 25]
]
gpt2_n_neighbors_search = [item for sublist in gpt2_n_neighbors_search_nested for item in sublist]

# LITE GPT2 DYNAMIC BATCHES NEIGHBOUR SEARCH
gpt2_n_neighbors_search_lite_dynamic_nested = [
    [
        TrainSentencesConfig(
            prefix="SentenceFormer_DynamicBatches_NoGlobalAlignment",
            learning_rate=0.00006,
            number_of_epochs=50,
            patience=5,
            model_layer_depth=1,
            model_layer_width_multiplier=1.6,
            num_pseudo_words=num_pseudo_words,
            replace_subject=False,
            number_of_neighbors=num_neighbors,
            model_layer_activation="leaky_relu",
            quanization=None,
            batch_size=32,
            embedding_llm_type="gpt-2",
            embedding_llm_name="gpt2",
            graph_dataset_name="TRExStarLite",
            pretrain_dataset_name="TriRExLite",
            train_dataset_name="TRExBiteLite",
            pretrained_model_name="",
            trained_model_name="",
            pretrained_path="",
            trained_path="",
            pretrained_checkpoint_directory="",
            trained_checkpoint_directory="",
            ignore_global_alignment=True,
        ) for num_neighbors in [100]
    ] for num_pseudo_words in [1, 2, 3, 4, 5, 10, 15, 25]
]
gpt2_n_neighbors_search_lite_dynamic = [item for sublist in gpt2_n_neighbors_search_lite_dynamic_nested for item in
                                        sublist]

gpt2_n_neighbors_search_dynamic_nested = [
    [
        TrainSentencesConfig(
            prefix="SentenceFormer_DynamicBatches_NoGlobalAlignment",
            learning_rate=0.00006,
            number_of_epochs=3,
            patience=0,
            model_layer_depth=1,
            model_layer_width_multiplier=1.6,
            num_pseudo_words=num_pseudo_words,
            replace_subject=False,
            number_of_neighbors=num_neighbors,
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
        ) for num_neighbors in [100]
    ] for num_pseudo_words in [1]
]
gpt2_n_neighbors_search_dynamic = [item for sublist in gpt2_n_neighbors_search_dynamic_nested for item in
                                   sublist]
