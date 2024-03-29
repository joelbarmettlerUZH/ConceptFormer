import csv
import os

import h5py
import numpy as np
import torch
from torchbiggraph.graph_storages import FORMAT_VERSION_ATTR, FORMAT_VERSION
from tqdm import tqdm

from GraphAligner.BigGraphAligner import BigGraphAligner
from GraphQueryEngine.SparqlEngine import get_wikidata_entities
from src.LLM.LLM import LLM


class WikidataGraphBuilder(BigGraphAligner):
    def __init__(self, llm: LLM, epochs=1000):
        super().__init__(llm, graphs=dict(), dataset_name="full_wikidata", epochs=epochs)


    def prepare(self):
        if os.path.isfile(f'{self.folder}/init/embeddings_entity_0.v0.h5'):
            print("Already prepared")
            return

        os.makedirs(f'{self.folder}/init', exist_ok=True)

        entities = dict()
        relations = dict()
        triples = set()

        all_wikidata_entities = get_wikidata_entities()
        for entity in tqdm(all_wikidata_entities, desc="Processing Wikidata"):
            # Handle Properties
            if entity['type'] == 'property':
                if entity['datatype'] != 'wikibase-item':
                    continue
                predicate_id = entity['id']
                predicate_label = entity['labels'].get('en', {}).get('value', None)
                if not predicate_label:
                    continue
                relations[predicate_id] = predicate_label

            # Handle Items
            if entity['type'] == 'item':
                entity_id = entity['id']
                entity_label = entity['labels'].get('en', {}).get('value', None)
                if not entity_label:
                    continue
                entities[entity_id] = entity_label

                for predicate_id, claims in entity['claims'].items():
                    for claim in claims:
                        if claim["type"] != "statement" or not claim["mainsnak"].get("datavalue", None):
                            continue
                        if claim["mainsnak"]["datavalue"]["type"] != "wikibase-entityid":
                            continue
                        if claim["mainsnak"]["datavalue"]["value"]["entity-type"] != "item":
                            continue
                        neighbour_id = claim["mainsnak"]["datavalue"]["value"]["id"]
                        triples.add((entity_id, predicate_id, neighbour_id))

        entity_list = [(key, value) for key, value in entities.items()]
        sorted_entity_list = sorted(entity_list, key=lambda x: int(x[0][1:]))

        relation_list = [(key, value) for key, value in relations.items()]
        sorted_relation_list = sorted(relation_list, key=lambda x: int(x[0][1:]))

        triples_list = list(triples)
        sorted_triples_list = sorted(triples_list, key=lambda x: (int(x[0][1:]), int(x[1][1:]), int(x[2][1:])))

        print("Entities:", len(entity_list))
        print("Relations:", len(relation_list))
        print("Triples:", len(triples_list))

        with open(f'{self.folder}/graph_edges.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['Subject', 'Predicate', 'Object'])
            writer.writerows(sorted_triples_list)

        # Write entities to a file
        with open(f'{self.folder}/entities.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['ID', 'Label'])
            for entity in tqdm(sorted_entity_list, desc='Writing entities'):
                writer.writerow(entity)

        # Write relations to a file
        with open(f'{self.folder}/relations.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['ID', 'Label'])
            for relation in tqdm(sorted_relation_list, desc='Writing relations'):
                writer.writerow(relation)

        cpu = torch.device("cpu")
        dataset = torch.zeros((len(sorted_entity_list), self.llm.embedding_length), dtype=torch.float32).to(cpu)
        for i, entity in tqdm(list(enumerate(sorted_entity_list)), desc='Embedding entities'):
            dataset[i, :] = self.llm.late_embedding(entity[1]).squeeze(0).squeeze(0).to(cpu)

        with h5py.File(f'{self.folder}/init/embeddings_entity_0.v0.h5', 'w') as hf:
            hf.create_dataset("embeddings", data=dataset.cpu().numpy(), dtype=np.float32)
            hf.attrs[FORMAT_VERSION_ATTR] = FORMAT_VERSION
