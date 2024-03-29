import logging
from typing import Dict, List, Tuple

import wandb
from wandb.sdk.wandb_run import Run


def init_or_recover_wandb(org: str, project: str, name: str, config: Dict, relevant_keys: List[str]) -> Tuple[Run, bool]:
    wandb.login()
    api = wandb.Api()

    filters = [
        {"$not": {"tags": "Invalid"}},
    ]
    for key in relevant_keys:
        filters.append({
            f"config.{key}": config[key]
        })
    runs = api.runs(f"{org}/{project}",
                    {"$and": filters})

    try:
        if len(runs) == 1:
            run = runs[0]
            print("Found existing run", name, run.name)
            return wandb.init(
                project=project,
                id=run.id,
                resume="must",
            ), "early_stop" in run.tags
    except ValueError:
        pass
    return wandb.init(
        project=project,
        name=name,
        config=config
    ), False
