import bz2
import json
import logging
import time
from pathlib import Path
from typing import Dict, Generator, Any
from urllib.error import HTTPError

import networkx as nx
import requests
from SPARQLWrapper import SPARQLWrapper, JSON


artifacts_directory = Path(__file__).parent.parent.parent / "data" / "input"
artifacts_directory.mkdir(parents=True, exist_ok=True)

# Can also be local qEndpoint: https://hub.docker.com/r/qacompany/qendpoint-wikidata
endpoint_url = "https://query.wikidata.org/bigdata/namespace/wdq/sparql"


def clean(label: str):
    label = label.replace(";", " ")
    label = label.replace("\t", " ")
    label = label.replace("\n", " ")
    label = label.strip()
    return label if label else None


def extract_entity_id(uri: str) -> str:
    return uri.split("/")[-1]

def get_pagerank_map() -> Dict[str, float]:
    """
    Download and process pagerank file. The map contains a pagerank value for every wikidata entity
    :return: Dict mapping wikidata IDs to their pagerank
    """
    file_name = '2023-12-04.allwiki.links.rank'
    bz2_file_name = file_name + '.bz2'
    file_path = artifacts_directory / file_name
    bz2_file_path = artifacts_directory / bz2_file_name
    url = 'https://danker.s3.amazonaws.com/' + bz2_file_name

    # Check if the file exists
    if not file_path.exists():
        # Download the bz2 file
        response = requests.get(url)
        with open(bz2_file_path, 'wb') as file:
            file.write(response.content)

        # Unpack the bz2 file
        with bz2.open(bz2_file_path, 'rb') as f_in:
            with open(file_path, 'wb') as f_out:
                f_out.write(f_in.read())

        # Remove the bz2 file
        bz2_file_path.unlink()

    entity_ranks = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split('\t')
            entity_ranks[key] = float(value)
    return entity_ranks

def get_wikidata_entities() -> Generator[Dict[str, Any], None, None]:
    """
    Download and process pagerank file. The map contains a pagerank value for every wikidata entity
    :return: Dict mapping wikidata IDs to their pagerank
    """
    file_name = 'latest-all.json'
    bz2_file_name = file_name + '.bz2'
    file_path = artifacts_directory / file_name
    bz2_file_path = artifacts_directory / bz2_file_name
    url = 'https://dumps.wikimedia.org/wikidatawiki/entities/' + bz2_file_name

    # Check if the file exists
    if not file_path.exists():
        # Download the bz2 file
        response = requests.get(url)
        with open(bz2_file_path, 'wb') as file:
            file.write(response.content)

        # Unpack the bz2 file
        with bz2.open(bz2_file_path, 'rb') as f_in:
            with open(file_path, 'wb') as f_out:
                f_out.write(f_in.read())

        # Remove the bz2 file
        bz2_file_path.unlink()

    with open(file_path, 'r') as file:
        next(file)
        for line in file:
            try:
                yield json.loads(line[:-2])
            except json.decoder.JSONDecodeError:
                continue

def get_entity_label(entity_id: str) -> str | None:
    """
    Returns the label of an entity given its Q-ID
    :param entity_id: Entity Q-ID
    :return: Entity label (in english) or None
    """
    query = f"""
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX wikibase: <http://wikiba.se/ontology#>
        PREFIX bd: <http://www.bigdata.com/rdf#>

        SELECT ?label WHERE {{
            wd:{entity_id} rdfs:label ?label.
            FILTER(LANG(?label) = "en")
        }}
    """

    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    results = None

    while True:
        try:
            results = sparql.query().convert()
        except Exception as e:
            # If exception is HTTPException and code is 429
            if isinstance(e, HTTPError) and e.code == 429:
                retry_after = int(e.headers.get("Retry-After", 30))
                time.sleep(retry_after)
            else:
                return None
        if results:
            break

    # Extract the label from the query results
    if results["results"]["bindings"]:
        label = results["results"]["bindings"][0]["label"]["value"]
        return clean(label)
    else:
        return None


def fetch_neighbors(pageranks: Dict[str, float], entity_id: str, edge_limit: int) -> nx.Graph | None:
    """
    Fetches the local neighbourhood star subgraph of a given entity ID. Note that the star subgraph is a subgraph
    in which the entity is in the center and edges only exists between the central entity and its neighbours.
    We only extract a limited number of edges, as some entities (like the United States of America) have a tremendous
    amount of neighbours. To select the most important ones, we use their pagerank.
    :param pageranks: Pagerank for every Entity.
    :param entity_id: ID of the central entity
    :param edge_limit: Maximum number of edges to extract
    :return: nx.Graph or None
    """
    query = f"""
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX bd: <http://www.bigdata.com/rdf#>

        SELECT ?relation ?propertyLabel ?neighbor ?neighborLabel 
        WHERE {{
          wd:{entity_id} ?relation ?neighbor .
          ?property wikibase:directClaim ?relation .
          FILTER(STRSTARTS(STR(?neighbor), "http://www.wikidata.org/entity/Q"))
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
    """

    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    results = None

    while True:
        try:
            results = sparql.query().convert()
        except Exception as e:
            # If exception is HTTPException and code is 429
            if isinstance(e, HTTPError) and e.code == 429:
                retry_after = int(e.headers.get("Retry-After", 30))
                time.sleep(retry_after)
            else:
                return
        if results:
            break

    entity_label = get_entity_label(entity_id)
    if not entity_label:
        return

    #  Create an empty NetworkX Graph
    rank = pageranks.get(entity_id, 0.5)
    G = nx.DiGraph()
    G.graph['central_node'] = entity_id
    G.graph['central_node_label'] = entity_label
    G.graph['central_node_rank'] = rank
    G.add_node(entity_id, label=entity_label, rank=rank)

    edge_candidats = []

    for result in results["results"]["bindings"]:
        # Check if both neighborLabel and propertyLabel are available
        if "neighborLabel" not in result or "propertyLabel" not in result:
            continue

        # Neighbour entity ID
        neighbor_id = result["neighbor"]["value"].split("/")[-1]
        neighbor_label = result["neighborLabel"]["value"] if result['neighbor']['type'] == 'uri' else \
            result['neighbor']['value']
        relation_label = result["propertyLabel"]["value"]
        relation_id = result["relation"]["value"].split("/")[-1]  # Extracting relation ID

        neighbor_label = clean(neighbor_label)
        relation_label = clean(relation_label)

        if not neighbor_label or not relation_label:
            continue

        if (neighbor_label.startswith("Category:") or
                neighbor_id.startswith("Template:") or
                neighbor_id.startswith("Wikipedia:") or
                neighbor_label == neighbor_id):
            continue

        edge_candidats.append({
            "neighbor_id": neighbor_id,
            "neighbor_label": neighbor_label,
            "relation_label": relation_label,
            "relation_id": relation_id,
            "rank": pageranks.get(neighbor_id, 0.5),
        })

    edge_candidats = sorted(edge_candidats, key=lambda x: x['rank'], reverse=True)

    if len(edge_candidats) > edge_limit:
        edge_candidats = edge_candidats[:edge_limit]

    for edge in edge_candidats:
        G.add_node(edge['neighbor_id'], label=edge['neighbor_label'], rank=edge['rank'])
        G.add_edge(entity_id, edge['neighbor_id'], id=edge['relation_id'], label=edge['relation_label'])

    logger = logging.getLogger(__name__)

    logger.debug("Central Entity and its Properties:")
    central_node = G.graph['central_node']
    central_node_label = G.graph['central_node_label']
    logger.debug(f"Central Entity: {central_node} (Label: {central_node_label})")
    logger.debug(f"Properties of Central Entity: {G.nodes[central_node]}")

    logger.debug("Edges and their Nodes:")
    for edge in G.edges(data=True):
        node1, node2, data = edge
        logger.debug(f"Edge between '{node1}' and '{node2}'")
        logger.debug(f"  - Relation ID: {data.get('id', 'N/A')}")
        logger.debug(f"  - Relation Label: {data.get('label', 'N/A')}")

        # Get neighbor properties
        neighbor_properties = G.nodes[node2]
        logger.debug(f"  - Neighbor Label: {neighbor_properties.get('label', 'N/A')}")
        logger.debug(f"  - Rank: {neighbor_properties.get('rank', 'N/A')}")

    return G

def mid_to_qid(mid: str) -> str | None:
    """
    Returns the Wikidata Q-ID of an entity given its Freebase ID (P646)
    :param freebase_id: Freebase ID
    :return: Wikidata Q-ID or None
    """
    freebase_id = f"/m/{mid[2:]}"

    query = f"""
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX wikibase: <http://wikiba.se/ontology#>
            PREFIX bd: <http://www.bigdata.com/rdf#>

            SELECT ?entity WHERE {{
                ?entity wdt:P646 "{freebase_id}".
            }}
        """

    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    results = None

    while True:
        try:
            results = sparql.query().convert()
        except Exception as e:
            # If exception is HTTPException and code is 429
            if isinstance(e, HTTPError) and e.code == 429:
                retry_after = int(e.headers.get("Retry-After", 30))
                time.sleep(retry_after)
            else:
                return None
        if results:
            break

    # Extract the Wikidata ID from the query results
    if results["results"]["bindings"]:
        wikidata_id = results["results"]["bindings"][0]["entity"]["value"]
        # Extracting the Q-ID from the full URI
        return wikidata_id.split('/')[-1]
    else:
        return None

