"""ConceptFormer v2 CLI (typer)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:  # heavy infer-group types, imported lazily at runtime
    from conceptformer.model.chat import ChatModel
    from conceptformer.train.trainer import ConceptTrainer

import typer
from rich import print as rprint
from rich.table import Table

from conceptformer.config import settings
from conceptformer.data.benchmarks import load_popqa, subject_qids
from conceptformer.data.coverage import audit_coverage
from conceptformer.data.snapshot import build_snapshot_parallel
from conceptformer.data.wikidata import WikidataClient
from conceptformer.schemas import Subgraph

app = typer.Typer(add_completion=False, help="ConceptFormer v2 data pipeline.")


@app.command()
def entity(
    qid: Annotated[str, typer.Argument(help="Wikidata entity id, e.g. Q42")],
    max_neighbors: Annotated[int, typer.Option(help="Truncate display (0 = all)")] = 0,
) -> None:
    """Fetch and print one entity's complete 1-hop neighborhood (sanity check)."""
    with WikidataClient() as client:
        sg = client.fetch_neighborhood(qid, max_neighbors=max_neighbors or None)
    if sg is None:
        rprint(f"[red]Entity {qid} not found.[/red]")
        raise typer.Exit(1)
    rprint(f"[bold]{sg.center.qid}[/bold] — {sg.center.label} ({sg.center.description})")
    rprint(f"edges: {len(sg.edges)} kept / {sg.n_edges_total} total (capped={sg.capped})")
    table = Table("property", "→ neighbor", "rank")
    for e in sg.edges:
        table.add_row(
            f"{e.property_label} ({e.property_id})",
            f"{e.neighbor.label} ({e.neighbor.qid})",
            str(int(e.neighbor.rank or 0)),
        )
    rprint(table)


@app.command("popqa-info")
def popqa_info(
    limit: Annotated[int, typer.Option(help="Rows to print")] = 5,
) -> None:
    """Load PopQA and show a few normalized rows + subject coverage."""
    examples = load_popqa()
    qids = subject_qids(iter(examples))
    rprint(f"[bold]PopQA[/bold]: {len(examples)} rows, {len(qids)} unique subject QIDs")
    for ex in examples[:limit]:
        rprint(json.loads(ex.model_dump_json()))


@app.command("build-popqa-snapshot")
def build_popqa_snapshot(
    limit: Annotated[int, typer.Option(help="0 = all subjects")] = 0,
    name: Annotated[str, typer.Option(help="Snapshot name")] = "popqa",
) -> None:
    """Snapshot the complete 1-hop neighborhoods of PopQA subject entities (concurrent)."""
    settings.ensure_dirs()
    examples = load_popqa()
    qids = subject_qids(iter(examples))
    if limit:
        qids = qids[:limit]
    out = build_snapshot_parallel(name, qids)
    manifest = json.loads((out / "manifest.json").read_text())
    rprint(f"[green]Wrote snapshot[/green] {out}")
    rprint(manifest)


@app.command("audit-popqa")
def audit_popqa(
    name: Annotated[str, typer.Option(help="Snapshot name to audit against")] = "popqa",
) -> None:
    """Report answer-in-graph coverage of PopQA against a snapshot."""
    examples = load_popqa()
    report = audit_coverage(examples, settings.snapshots_dir / name)
    rprint(f"[bold]PopQA coverage vs snapshot '{name}'[/bold]")
    rprint(report)


@app.command("eval-popqa")
def eval_popqa(
    condition: Annotated[str, typer.Option(help="base | rag")] = "base",
    model: Annotated[str, typer.Option(help="HF model id")] = "Qwen/Qwen3-0.6B",
    snapshot: Annotated[str, typer.Option(help="Defines the example set")] = "popqa_sample",
    n: Annotated[int, typer.Option(help="0 = all covered examples")] = 0,
    device: Annotated[str, typer.Option(help="cuda / cuda:1 / cpu")] = "cuda",
    batch_size: Annotated[int, typer.Option(help="Generation batch size")] = 64,
) -> None:
    """Evaluate a condition on PopQA. All conditions use the SAME snapshot-covered example set."""
    import random

    from conceptformer.cache import KVCache
    from conceptformer.data.snapshot import iter_subgraphs
    from conceptformer.eval.harness import is_graph_supported, save_predictions, save_report, score
    from conceptformer.eval.predictors import (
        TextPredictor,
        base_prompt,
        make_rag_prompt,
        prompt_spec,
    )
    from conceptformer.eval.subclass import SUBCLASS_RELATIONS, SubclassExpander
    from conceptformer.eval.tokens import measure_text_prompt_tokens, token_report
    from conceptformer.model.chat import ChatModel

    # The snapshot defines the universe of evaluable examples for EVERY condition, so base /
    # rag / conceptformer are scored on identical examples (fair comparison).
    subgraphs = {sg.center.qid: sg for sg in iter_subgraphs(settings.snapshots_dir / snapshot)}
    examples = [e for e in load_popqa() if e.subject_qid in subgraphs]
    if n:
        random.Random(0).shuffle(examples)
        examples = examples[:n]

    # subclass-expanded aliases for the `fair` metric (type-like relations only)
    with SubclassExpander(settings) as expander:
        gold_qids = {
            e.answer_qid for e in examples if e.answer_qid and e.relation in SUBCLASS_RELATIONS
        }
        rprint(f"[dim]expanding subclasses for {len(gold_qids)} gold class entities…[/dim]")
        expanded_aliases = expander.expand(gold_qids)

    gen_cache = KVCache(settings.generation_cache_path)  # resumable, skips re-generation
    chat = ChatModel(model, device=device, cache=gen_cache)
    builder = (
        make_rag_prompt(chat.count_tokens, settings.rag_context_tokens)
        if condition == "rag"
        else base_prompt
    )
    predictor = TextPredictor(chat, builder, condition, batch_size=batch_size)
    predictions = predictor.predict_batch(examples, subgraphs)

    # Token-efficiency accounting (the paper's headline): exact input length per example, the
    # marginal knowledge payload (facts for RAG, 0 for base), and the FULL (un-budgeted)
    # neighborhood text cost — all vs ConceptFormer's constant k.
    from conceptformer.verbalize import verbalize

    active_prompts = [builder(e, subgraphs.get(e.subject_qid)) for e in examples]
    base_prompts = [base_prompt(e, None) for e in examples]
    uncapped_facts = [
        verbalize(subgraphs[e.subject_qid]) if e.subject_qid in subgraphs else None
        for e in examples
    ]
    token_records = measure_text_prompt_tokens(chat, active_prompts, base_prompts, uncapped_facts)
    gen_cache.close()

    graph_supported = [is_graph_supported(e, subgraphs[e.subject_qid]) for e in examples]
    supported = [
        (e, p) for e, p, gs in zip(examples, predictions, graph_supported, strict=True) if gs
    ]

    full_score = score(examples, predictions, expanded_aliases=expanded_aliases)
    gs_score = score(
        [e for e, _ in supported], [p for _, p in supported], expanded_aliases=expanded_aliases
    )
    report = {
        "condition": condition,
        "model": model,
        "snapshot": snapshot,
        "prompt": prompt_spec(condition),
        "rag_context_tokens": settings.rag_context_tokens if condition == "rag" else None,
        "tokens": token_report(token_records),
        "n_total": len(examples),
        "n_graph_supported": len(supported),
        "full": full_score,
        "graph_supported": gs_score,
    }
    report_path = save_report(report, settings.results_dir)
    pred_path = save_predictions(
        examples, predictions, graph_supported, settings.results_dir,
        condition=condition, model=model, expanded_aliases=expanded_aliases,
        token_records=token_records,
    )
    rprint(
        f"[bold]PopQA {condition} — {model}[/bold] on '{snapshot}' "
        f"(n={len(examples)}, graph-supported={len(supported)})"
    )
    for view, blk in (("full", full_score), ("graph_supported", gs_score)):
        cells = "  ".join(f"{m}={b['accuracy_pct']}%" for m, b in blk["metrics"].items())
        rprint(f"[bold]{view}[/bold]: {cells}")
    tok = report["tokens"]
    uncapped = tok.get("uncapped_knowledge_tokens", {}).get("mean")
    uncapped_str = f"  uncapped-knowledge mean={uncapped}" if uncapped is not None else ""
    rprint(
        f"[bold]tokens[/bold]: input mean={tok['input_tokens']['mean']} "
        f"(p95={tok['input_tokens']['p95']})  knowledge mean={tok['knowledge_tokens']['mean']}"
        f"{uncapped_str}"
    )
    rprint(f"saved: {report_path}  |  {pred_path}")


@app.command("select-entities")
def select_entities_cmd(
    n: Annotated[int, typer.Option(help="How many entities to select")] = 20000,
    name: Annotated[str, typer.Option(help="Output set name")] = "cf_train_pilot",
    balanced: Annotated[bool, typer.Option(help="cover full popularity spectrum (vs tail-heavy)")] = False,  # noqa: E501
    seed: Annotated[int, typer.Option(help="Sampling seed")] = 0,
) -> None:
    """Select a CF-Train entity pool from the danker PageRank file (eval-excluded)."""
    from conceptformer.data.select import (
        BALANCED_FRACS,
        SAMPLE_FRACS,
        download_pagerank,
        load_pagerank,
        manifest,
        save_entities,
        select_entities,
    )

    path = download_pagerank()
    entries = load_pagerank(path)
    exclude = subject_qids(iter(load_popqa()))  # never train on eval subjects
    fracs = BALANCED_FRACS if balanced else SAMPLE_FRACS
    rprint(f"sampling {'balanced' if balanced else 'tail-heavy'} low/mid/high={fracs}")
    selected = select_entities(entries, n=n, exclude=exclude, seed=seed, sample_fracs=fracs)

    out_dir = settings.data_root / "cf_train" / name
    save_entities(selected, out_dir / "entities.jsonl")
    m = manifest(selected, n_requested=n, source=settings.pagerank_url)
    (out_dir / "manifest.json").write_text(json.dumps(m, indent=2))

    rprint(f"pool={len(entries):,} sitelinked entities | excluded {len(exclude):,} eval subjects")
    rprint(m)
    for tier in ("low", "mid", "high"):
        ex = next((e for e in selected if e.tier == tier), None)
        if ex:
            rprint(f"  [{tier}] e.g. {ex.qid} (rank {ex.rank:.1f})")
    rprint(f"saved: {out_dir}")


@app.command("build-cftrain-snapshot")
def build_cftrain_snapshot_cmd(
    candidates: Annotated[str, typer.Option(help="select-entities output name")] = "cf_train_pilot",
    name: Annotated[str, typer.Option(help="Snapshot name")] = "cftrain_pilot",
    target: Annotated[int, typer.Option(help="Target count of USABLE entities")] = 20000,
    min_edges: Annotated[int, typer.Option(help="Min facts to keep an entity")] = 6,
    no_pagerank: Annotated[bool, typer.Option(help="Skip PageRank ranking (faster)")] = False,
) -> None:
    """Snapshot CF-Train candidates, filtering junk/thin entities until `target` usable."""
    from conceptformer.data.select import download_pagerank, load_entities, load_pagerank_map
    from conceptformer.data.snapshot import build_cftrain_snapshot

    cand_path = settings.data_root / "cf_train" / candidates / "entities.jsonl"
    qids = [e.qid for e in load_entities(cand_path)]
    pagerank = None
    if not no_pagerank:
        rprint("[dim]loading PageRank map for neighbor ranking…[/dim]")
        pagerank = load_pagerank_map(download_pagerank())

    out = build_cftrain_snapshot(name, qids, target=target, min_edges=min_edges, pagerank=pagerank)
    m = json.loads((out / "manifest.json").read_text())
    rprint(f"[bold]CF-Train snapshot '{name}'[/]")
    rprint(m)
    if not m["complete"]:
        rprint(
            f"[yellow]incomplete: {m['n_usable']}/{target} usable from {m['n_candidates_seen']} "
            f"candidates (junk {m['n_wikimedia_junk']}, thin {m['n_thin']}, miss {m['n_missing']}) "
            f"— select more candidates[/yellow]"
        )


@app.command("extract-teacher-paths")
def extract_teacher_paths(
    dataset: Annotated[str, typer.Option(help="tiered CF-Train dataset name")] = "cftrain_qa_smoke",
    snapshot: Annotated[str, typer.Option(help="matching snapshot")] = "cftrain_smoke",
    model: Annotated[str, typer.Option(help="frozen teacher backbone")] = "Qwen/Qwen3-0.6B",
    max_new_tokens: Annotated[int, typer.Option()] = 64,
    device: Annotated[str, typer.Option()] = "cuda",
    batch_size: Annotated[int, typer.Option()] = 32,
) -> None:
    """Stage 5: store each example's frozen-teacher greedy target path (graph-in-context)."""
    import statistics
    from datetime import UTC, datetime

    from conceptformer.cache import KVCache
    from conceptformer.data.snapshot import iter_subgraphs
    from conceptformer.generate.dataset import (
        dataset_counts,
        load_cftrain_qa,
        save_cftrain_qa,
        update_manifest,
    )
    from conceptformer.generate.teacher import (
        CFTRAIN_PROMPT_VERSION,
        attach_teacher_paths,
        drop_degenerate_paths,
        teacher_path_key,
        teacher_prompt,
    )
    from conceptformer.model.chat import ChatModel
    from conceptformer.verbalize import verbalize_with_answer

    qa_dir = settings.data_root / "cf_train" / dataset
    rows = load_cftrain_qa(qa_dir / "qa_tiered.jsonl")
    sg_by_qid = {sg.center.qid: sg for sg in iter_subgraphs(settings.snapshots_dir / snapshot)}

    chat = ChatModel(model, device=device, cache=KVCache(settings.generation_cache_path))
    budget = settings.rag_context_tokens
    prompts: list[tuple[str, str]] = []  # (system, user) per row
    for r in rows:
        sg = sg_by_qid.get(r.subject_qid)
        # Guarantee the answer's edge is in the teacher's facts (large neighborhoods can cut it).
        facts = verbalize_with_answer(sg, r.answer_qid, chat.count_tokens, budget) if sg else ""
        prompts.append(teacher_prompt(r.question, facts))

    # RESUMABLE: per-row cache the greedy path, generate only uncached rows, FLUSH the cache after
    # each chunk — a kill mid-run loses at most one chunk, a re-run skips what's done (the full
    # extract is ~12h on 100k, ~40h on 1M, so this is essential, not optional).
    path_cache = KVCache(settings.generation_cache_path)
    keys = [teacher_path_key(model, max_new_tokens, f"{sys}\n{user}") for sys, user in prompts]
    ids: list[list[int]] = [[] for _ in prompts]
    todo: list[tuple[int, tuple[str, str]]] = []
    for i, k in enumerate(keys):
        hit = path_cache.get(k)
        if hit is not None:
            ids[i] = hit["ids"]
        else:
            todo.append((i, prompts[i]))
    rprint(f"[dim]teacher paths: {len(prompts) - len(todo)}/{len(prompts)} cached; "
           f"generating {len(todo)}…[/dim]")
    flush = max(batch_size, 1) * 16  # cache-flush granularity (~16 batches)
    for c in range(0, len(todo), flush):
        chunk = todo[c : c + flush]
        gen = chat.generate_batch_ids(
            [p for _, p in chunk], max_new_tokens=max_new_tokens, batch_size=batch_size
        )
        for (i, _), g in zip(chunk, gen, strict=True):
            ids[i] = list(g)
        path_cache.put_many({keys[i]: {"ids": ids[i]} for i, _ in chunk})
    texts = [chat.decode(i) for i in ids]
    enriched = attach_teacher_paths(rows, ids, texts)
    enriched, n_degenerate = drop_degenerate_paths(enriched)  # no empty/whitespace targets
    out_path = save_cftrain_qa(enriched, qa_dir / "qa_distill.jsonl")

    lengths = [len(r.teacher_target_ids or []) for r in enriched]
    update_manifest(
        qa_dir,
        "teacher",
        {
            "model": model,
            "prompt_version": CFTRAIN_PROMPT_VERSION,
            "snapshot": snapshot,
            "max_new_tokens": max_new_tokens,
            "n_degenerate_dropped": n_degenerate,
            "path_len_mean": round(statistics.mean(lengths), 1) if lengths else 0.0,
            "path_len_max": max(lengths) if lengths else 0,
            **dataset_counts(enriched),
        },
        stamp=datetime.now(UTC).isoformat(),
    )

    rprint(f"[green]teacher paths for {len(enriched)} examples[/] → {out_path}")
    if n_degenerate:
        rprint(f"  [yellow]dropped {n_degenerate} degenerate (empty) teacher paths[/]")
    if lengths:
        rprint(f"  path length: mean {statistics.mean(lengths):.1f}, max {max(lengths)}")
    for r in enriched[:4]:
        target = (r.teacher_target_text or "")[:60]
        rprint(f"  [{r.task_type}] {r.question[:50]!r} → [dim]{target!r}[/]")


@app.command("tier-cftrain")
def tier_cftrain(
    dataset: Annotated[str, typer.Option(help="CF-Train QA dataset name")] = "cftrain_qa_smoke",
    snapshot: Annotated[str, typer.Option(help="matching snapshot")] = "cftrain_smoke",
    model: Annotated[str, typer.Option(help="backbone for base/RAG")] = "Qwen/Qwen3-0.6B",
    keep_easy: Annotated[float, typer.Option(help="fraction of base-known to keep")] = 0.3,
    device: Annotated[str, typer.Option(help="cuda / cpu")] = "cuda",
    batch_size: Annotated[int, typer.Option()] = 64,
) -> None:
    """Signal-tier CF-Train QA (base-fails-RAG-succeeds = keep) and drop hard examples."""
    from collections import Counter
    from datetime import UTC, datetime

    from conceptformer.cache import KVCache
    from conceptformer.data.snapshot import iter_subgraphs
    from conceptformer.generate.dataset import (
        dataset_counts,
        load_cftrain_qa,
        save_cftrain_qa,
        update_manifest,
    )
    from conceptformer.generate.signal import answer_ok, apply_keep_policy, assign_tier
    from conceptformer.generate.teacher import CFTRAIN_PROMPT_VERSION, cftrain_prompt
    from conceptformer.model.chat import ChatModel
    from conceptformer.verbalize import verbalize_with_answer

    qa_dir = settings.data_root / "cf_train" / dataset
    rows = load_cftrain_qa(qa_dir / "qa.jsonl")
    qa = [r for r in rows if r.task_type in ("single", "compositional")]
    preserved = [r for r in rows if r.task_type in ("descriptive", "control")]  # always kept
    sg_by_qid = {sg.center.qid: sg for sg in iter_subgraphs(settings.snapshots_dir / snapshot)}

    chat = ChatModel(model, device=device, cache=KVCache(settings.generation_cache_path))
    budget = settings.rag_context_tokens
    # Same CF-Train convention the teacher (Stage 5) + student (training) use, so the tier
    # signal reflects the actual training-time prompt — not the PopQA eval format.
    base_prompts = [cftrain_prompt(r.question) for r in qa]
    rag_prompts = []
    for r in qa:
        sg = sg_by_qid.get(r.subject_qid)
        # Answer-guaranteed facts: the tier signal must reflect "graph has the fact", not
        # "the fact survived the top-PageRank budget cut".
        facts = verbalize_with_answer(sg, r.answer_qid, chat.count_tokens, budget) if sg else ""
        rag_prompts.append(cftrain_prompt(r.question, facts))

    base_preds = chat.generate_batch(base_prompts, batch_size=batch_size)
    rag_preds = chat.generate_batch(rag_prompts, batch_size=batch_size)
    tiered = [
        (r, assign_tier(answer_ok(b, r.accepted_answers), answer_ok(g, r.accepted_answers)))
        for r, b, g in zip(qa, base_preds, rag_preds, strict=True)
    ]
    raw_tiers = Counter(t or "hard(dropped)" for _, t in tiered)

    kept_qa = apply_keep_policy(tiered, keep_base_known_frac=keep_easy)
    kept_preserved = [r.model_copy(update={"tier": r.task_type}) for r in preserved]
    final = kept_qa + kept_preserved
    out_path = save_cftrain_qa(final, qa_dir / "qa_tiered.jsonl")
    update_manifest(
        qa_dir,
        "tier",
        {
            "model": model,
            "prompt_version": CFTRAIN_PROMPT_VERSION,
            "snapshot": snapshot,
            "keep_easy": keep_easy,
            "raw_tiers": dict(raw_tiers),
            **dataset_counts(final),
        },
        stamp=datetime.now(UTC).isoformat(),
    )

    rprint(f"[bold]QA tiering[/] (of {len(qa)} QA examples): {dict(raw_tiers)}")
    rprint(
        f"[green]kept[/] {len(final)} = {len(kept_qa)} QA + {len(kept_preserved)} preserved "
        f"→ {out_path}"
    )
    rprint(dict(Counter(r.tier for r in final)))


@app.command("generate-cftrain")
def generate_cftrain(
    snapshot: Annotated[str, typer.Option(help="usable-entity snapshot")] = "cftrain_smoke",
    name: Annotated[str, typer.Option(help="Output QA dataset name")] = "cftrain_qa",
    limit: Annotated[int, typer.Option(help="0 = all entities in the snapshot")] = 0,
    n_questions: Annotated[int, typer.Option(help="Gemma QA questions per entity")] = 8,
    descriptive: Annotated[int, typer.Option(help="Descriptive tasks per entity")] = 2,
    control: Annotated[int, typer.Option(help="Control (no-hijack) tasks per entity")] = 2,
    base_url: Annotated[str, typer.Option(help="vLLM server")] = "http://localhost:8000/v1",
) -> None:
    """Generate the CF-Train task mix: Gemma QA + templated descriptive + control tasks."""
    import asyncio

    from conceptformer.data.snapshot import iter_subgraphs
    from conceptformer.generate.control import control_tasks
    from conceptformer.generate.dataset import (
        dataset_counts,
        save_cftrain_qa,
        update_manifest,
        validate_questions,
    )
    from conceptformer.generate.descriptive import descriptive_tasks
    from conceptformer.generate.gemma import GemmaClient

    sgs = list(iter_subgraphs(settings.snapshots_dir / snapshot))
    if limit:
        sgs = sgs[:limit]
    rprint(
        f"generating for {len(sgs)} entities "
        f"({n_questions} QA + {descriptive} desc + {control} ctrl)…"
    )

    # Per-entity cache → the run resumes after a crash instead of redoing completed entities
    # (the 100k corpus is a ~10h generation; a single save-at-end would lose it all on a failure).
    from conceptformer.cache import KVCache

    gen_cache = KVCache(settings.generation_cache_path)
    client = GemmaClient(base_url=base_url, n_questions=n_questions, cache=gen_cache)
    results = asyncio.run(client.generate(sgs))
    gen_cache.close()
    rows = validate_questions(sgs, results)
    for sg in sgs:  # descriptive + control need no LLM; add for every entity
        rows.extend(descriptive_tasks(sg, n=descriptive))
        rows.extend(control_tasks(sg, n=control))

    out_dir = settings.data_root / "cf_train" / name
    out_path = save_cftrain_qa(rows, out_dir / "qa.jsonl")
    n_failed = sum(r is None for r in results)
    from collections import Counter
    from datetime import UTC, datetime

    update_manifest(
        out_dir,
        "generate",
        {
            "snapshot": snapshot,
            "n_entities": len(sgs),
            "n_questions_per_entity": n_questions,
            "n_descriptive": descriptive,
            "n_control": control,
            "n_gen_failed": n_failed,
            **dataset_counts(rows),
        },
        stamp=datetime.now(UTC).isoformat(),
    )

    by_type = Counter(r.task_type for r in rows)
    rprint(f"[green]saved[/] {len(rows)} tasks → {out_path}  [dim]({n_failed} gen-failed)[/dim]")
    rprint(dict(by_type))


@app.command("prototype-questions")
def prototype_questions(
    snapshot: Annotated[str, typer.Option(help="Snapshot to pull entities from")] = "popqa_full",
    entities: Annotated[int, typer.Option(help="How many entities to try")] = 20,
    n_questions: Annotated[int, typer.Option(help="Questions per entity")] = 8,
    min_edges: Annotated[int, typer.Option(help="Skip entities with fewer neighbors")] = 10,
    show: Annotated[int, typer.Option(help="Entities to print questions for")] = 3,
    base_url: Annotated[str, typer.Option(help="vLLM server")] = "http://localhost:8000/v1",
) -> None:
    """Generate CF-Train questions via Gemma (vLLM) and report quality metrics."""
    import asyncio
    from collections import Counter

    from conceptformer.data.snapshot import iter_subgraphs
    from conceptformer.generate.gemma import GemmaClient
    from conceptformer.generate.grounding import passes, quality_flags

    sgs = []
    for sg in iter_subgraphs(settings.snapshots_dir / snapshot):
        if len(sg.edges) >= min_edges:
            sgs.append(sg)
        if len(sgs) >= entities:
            break

    client = GemmaClient(base_url=base_url, n_questions=n_questions)
    results = asyncio.run(client.generate(sgs))

    n_q = n_grounded = n_subj = n_pass = n_failed = 0
    by_type: Counter[str] = Counter()
    pass_by_type: Counter[str] = Counter()
    for i, (sg, res) in enumerate(zip(sgs, results, strict=True)):
        if res is None:
            n_failed += 1
            continue
        if i < show:
            rprint(f"\n[bold cyan]{sg.center.label} ({sg.center.qid})[/] — {len(sg.edges)} nbrs")
        for q in res.questions:
            flags = quality_flags(q, sg)
            ok = passes(q, sg)
            n_q += 1
            n_grounded += flags["grounded"]
            n_subj += flags["mentions_subject"]
            n_pass += ok
            by_type[q.task_type] += 1
            pass_by_type[q.task_type] += ok
            if i < show:
                color = "green" if ok else "yellow"
                rprint(f"  [{color}]{q.task_type:13}[/] {q.question}  [dim]→ {q.answer}[/dim]")

    if not n_q:
        rprint("[red]no questions generated[/red]")
        return
    pct = lambda x: f"{100 * x / n_q:.0f}%"  # noqa: E731
    rprint(f"\n[bold]Quality over {len(sgs)} entities ({n_failed} parse-failed)[/]")
    rprint(
        f"  questions={n_q}  grounded={pct(n_grounded)}  "
        f"names_subject={pct(n_subj)}  pass={pct(n_pass)}"
    )
    for t in by_type:
        share = f"{100 * by_type[t] / n_q:.0f}%"
        tpass = f"{100 * pass_by_type[t] / by_type[t]:.0f}%" if by_type[t] else "-"
        rprint(f"  {t:13} {by_type[t]:3} ({share} of mix)  pass={tpass}")


@app.command("validate-teacher")
def validate_teacher(
    name: Annotated[str, typer.Option(help="Snapshot name")] = "popqa_sample",
    n: Annotated[int, typer.Option(help="Examples to evaluate")] = 100,
    model: Annotated[str, typer.Option(help="HF model id")] = "Qwen/Qwen3-0.6B",
    device: Annotated[str, typer.Option(help="cuda / cuda:1 / cpu")] = "cuda",
    show: Annotated[int, typer.Option(help="Sample rows to print")] = 8,
) -> None:
    """Base vs graph-in-context accuracy on PopQA (validates the distillation teacher)."""
    from conceptformer.eval.teacher import run_teacher_validation

    report = run_teacher_validation(snapshot_name=name, n=n, model_id=model, device=device)
    rows = report.pop("sample_rows")
    rprint(f"[bold]Teacher validation: {report['model']}[/bold]")
    rprint(report)
    for r in rows[:show]:
        mark = "✓" if r["rag_ok"] else "✗"
        rprint(
            f"[{'green' if r['rag_ok'] else 'red'}]{mark}[/] "
            f"q={r['question']!r} gold={r['gold']!r} "
            f"base={r['base']!r} rag={r['rag']!r} (#vals={r['values_under_relation']})"
        )


@app.command("cf-overfit")
def cf_overfit(
    dataset: Annotated[str, typer.Option(help="distilled CF-Train dataset")] = "cftrain_qa_smoke",
    snapshot: Annotated[str, typer.Option(help="matching snapshot")] = "cftrain_smoke",
    model: Annotated[str, typer.Option(help="frozen backbone")] = "Qwen/Qwen3-0.6B",
    n: Annotated[int, typer.Option(help="examples to overfit")] = 16,
    steps: Annotated[int, typer.Option()] = 300,
    k: Annotated[int, typer.Option(help="concept tokens")] = 8,
    lr: Annotated[float, typer.Option()] = 1e-4,
    device: Annotated[str, typer.Option()] = "cuda",
) -> None:
    """Milestone: overfit a handful of examples to near-zero KL (validates the mechanism)."""
    from conceptformer.data.snapshot import iter_subgraphs
    from conceptformer.generate.dataset import load_cftrain_qa
    from conceptformer.model.backbone import Backbone
    from conceptformer.model.chat import ChatModel
    from conceptformer.train.trainer import ConceptTrainer, TrainConfig

    qa_dir = settings.data_root / "cf_train" / dataset
    rows = [r for r in load_cftrain_qa(qa_dir / "qa_distill.jsonl") if r.teacher_target_ids]
    sg_by_qid = {sg.center.qid: sg for sg in iter_subgraphs(settings.snapshots_dir / snapshot)}
    batch = [
        (sg_by_qid[r.subject_qid], r.question, r.teacher_target_ids or [], r.answer_qid)
        for r in rows[:n]
        if r.subject_qid in sg_by_qid
    ]
    rprint(f"overfitting {len(batch)} examples, k={k}, {steps} steps…")

    backbone = Backbone(ChatModel(model, device=device))
    trainer = ConceptTrainer(backbone, TrainConfig(k=k, lr=lr))
    first = trainer.step(batch)
    rprint(f"  step   0: KL={first:.4f}")
    for s in range(1, steps + 1):
        loss = trainer.step(batch)
        if s % max(1, steps // 15) == 0 or s == steps:
            raw = trainer.raw_gate()
            cnorm = trainer.concept_norm(batch[0][0])
            rprint(
                f"  step {s:3d}: KL={loss:.4f}  "
                f"raw_gate∈[{min(raw):+.3f},{max(raw):+.3f}]  |concept|≈{cnorm:.2f}"
            )
    rprint(f"[green]done[/] — KL {first:.4f} → {loss:.4f}")


def log_artifact_resilient(wb: object, art: object, label: str, retries: int = 3,
                           base_delay: float = 5.0) -> bool:
    """Log a W&B artifact, retrying transient failures; on final failure WARN instead of raising.

    A flaky W&B upload (service process timing out — seen 2026-07-02) must NEVER crash an already-
    completed training run: the checkpoint is saved LOCALLY before this is called, so on failure we
    just warn and the run finishes cleanly (re-upload later). Returns True on success."""
    import time

    for attempt in range(retries):
        try:
            wb.log_artifact(art)  # ty: ignore[unresolved-attribute]
            return True
        except Exception as e:  # broad on purpose — a flaky upload must not kill a finished run
            if attempt + 1 < retries:
                time.sleep(base_delay * (attempt + 1))
            else:
                rprint(f"[yellow]W&B artifact upload failed for {label} after {retries} tries "
                       f"({type(e).__name__}); checkpoint saved LOCALLY — reupload later.[/]")
    return False


@app.command("cf-train")
def cf_train(
    dataset: Annotated[str, typer.Option(help="distilled CF-Train dataset")] = "cftrain_qa_smoke",
    snapshot: Annotated[str, typer.Option(help="matching snapshot")] = "cftrain_smoke",
    model: Annotated[str, typer.Option(help="frozen backbone")] = "Qwen/Qwen3-0.6B",
    k: Annotated[int, typer.Option(help="concept tokens")] = 8,
    d_model: Annotated[int, typer.Option(help="encoder width")] = 512,
    n_layers: Annotated[int, typer.Option(help="resampler layers")] = 2,
    lr: Annotated[float, typer.Option(help="encoder learning rate")] = 1e-4,
    weight_decay: Annotated[float, typer.Option(help="AdamW weight decay")] = 0.01,
    warmup_frac: Annotated[float, typer.Option(help="warmup as a fraction of total steps")] = 0.05,
    schedule: Annotated[str, typer.Option(help="post-warmup LR decay: cosine|constant")] = "cosine",
    temperature: Annotated[float, typer.Option(help="KL distillation temperature")] = 1.0,
    steps: Annotated[int, typer.Option()] = 600,
    batch: Annotated[int, typer.Option(help="minibatch size")] = 8,
    val_frac: Annotated[float, typer.Option(help="held-out question fraction")] = 0.3,
    split_mode: Annotated[
        str, typer.Option(help="held-out unit: fact (no paraphrase leakage) | question (legacy)")
    ] = "fact",
    eval_every: Annotated[int, typer.Option()] = 100,
    eval_n: Annotated[int, typer.Option(help="held-out examples scored per eval (capped)")] = 120,
    eval_gen_batch: Annotated[
        int, typer.Option(help="generation batch for eval brackets; lower for >=1.7B backbones")
    ] = 48,
    popqa_eval: Annotated[int, typer.Option(help="after training, score N unseen PopQA")] = 0,
    popqa_snapshot: Annotated[str, typer.Option(help="snapshot with PopQA neighborhoods")] = "popqa_full",  # noqa: E501
    checkpoint: Annotated[str, typer.Option(help="save trained encoder under this name")] = "",
    augment: Annotated[bool, typer.Option(help="distill under many system prompts")] = False,
    subsample: Annotated[bool, typer.Option(help="re-sample teacher distractors/step")] = False,
    cache_teacher: Annotated[
        bool,
        typer.Option(help="cache teacher hidden upfront; --no for large-data/few-epoch"),
    ] = True,
    placement: Annotated[
        str,
        typer.Option(help="concept slot: prefix|before_entity|after_entity|replace_entity"),
    ] = "prefix",
    injection_port: Annotated[
        str, typer.Option(help="concept interface: text | vision (multimodal backbones only)")
    ] = "text",
    grad_clip: Annotated[float, typer.Option(help="max grad-norm (0=off)")] = 0.0,
    grad_accum: Annotated[
        int, typer.Option(help="micro-batches/step (effective batch = batch*grad_accum)")
    ] = 1,
    ema_decay: Annotated[float, typer.Option(help="EMA decay for eval/ckpt weights (0=off)")] = 0.0,
    gate_mode: Annotated[str, typer.Option(help="concept gate: tanh|none")] = "tanh",
    metrics_out: Annotated[str, typer.Option(help="write final metrics JSON to this path")] = "",
    wandb: Annotated[bool, typer.Option(help="log to Weights & Biases")] = False,
    wandb_project: Annotated[str, typer.Option()] = "conceptformer-v2",
    wandb_group: Annotated[str, typer.Option(help="group runs (e.g. a sweep name)")] = "",
    seed: Annotated[int, typer.Option()] = 0,
    device: Annotated[str, typer.Option()] = "cuda",
) -> None:
    """Generalization test: train on a question split, eval on HELD-OUT questions per entity."""
    import random

    from conceptformer.data.snapshot import iter_subgraphs
    from conceptformer.eval.evalsets import sample_rows
    from conceptformer.generate.dataset import load_cftrain_qa
    from conceptformer.model.backbone import Backbone
    from conceptformer.model.chat import ChatModel
    from conceptformer.train.harness import (
        split_by_held_out_facts,
        split_by_held_out_questions,
    )
    from conceptformer.train.trainer import (
        AUGMENT_SYSTEMS,
        HELD_OUT_EVAL_SYSTEM,
        TEACHER_SYSTEM,
        ConceptTrainer,
        TrainConfig,
        split_microbatches,
    )

    qa_dir = settings.data_root / "cf_train" / dataset
    rows = load_cftrain_qa(qa_dir / "qa_distill.jsonl")
    sg_by_qid = {sg.center.qid: sg for sg in iter_subgraphs(settings.snapshots_dir / snapshot)}
    # "fact" groups paraphrases of the same (entity, fact) so they never straddle train/val
    # (question-level splitting leaked paraphrases into "held-out"); "question" = legacy mode,
    # kept only to reproduce old checkpoints' splits.
    splitter = split_by_held_out_facts if split_mode == "fact" else split_by_held_out_questions
    train_rows, val_rows = splitter(rows, val_frac=val_frac, seed=seed)
    train_tuples = [
        (sg_by_qid[r.subject_qid], r.question, r.teacher_target_ids, r.answer_qid)
        for r in train_rows
        if r.subject_qid in sg_by_qid and r.teacher_target_ids
    ]
    val_rows = [r for r in val_rows if r.subject_qid in sg_by_qid]
    rng = random.Random(seed)
    # Fixed-seed eval subset (NOT the training seed): every run/config/seed scores the SAME
    # held-out questions, so trajectories compare and per-item outputs pair (evalsets.py).
    eval_val = sample_rows(val_rows, eval_n)
    rprint(
        f"train {len(train_tuples)} examples / val {len(val_rows)} held-out questions "
        f"(eval on {len(eval_val)}; k={k}, {steps} steps, batch {batch})"
    )

    backbone = Backbone(ChatModel(model, device=device))
    # Cap the teacher facts budget for training: tail entities have small neighborhoods so this
    # rarely truncates, and it bounds the padded (B, L, V) logits tensor's memory.
    cfg = TrainConfig(
        k=k,
        d_model=d_model,
        n_layers=n_layers,
        lr=lr,
        weight_decay=weight_decay,
        schedule=schedule,
        temperature=temperature,
        warmup_steps=max(10, int(steps * warmup_frac)),
        total_steps=steps,
        rag_context_tokens=1024,
        augment_systems=AUGMENT_SYSTEMS if augment else (),
        subsample_neighbors=subsample,
        cache_teacher=cache_teacher,
        placement=placement,
        injection_port=injection_port,
        grad_clip=grad_clip,
        grad_accum=grad_accum,
        ema_decay=ema_decay,
        gate_mode=gate_mode,
        seed=seed,
    )
    trainer = ConceptTrainer(backbone, cfg)

    wb = None
    if wandb:
        import wandb as _wandb

        wb = _wandb.init(
            project=wandb_project,
            group=wandb_group or None,
            name=checkpoint or None,
            config={
                "k": k, "d_model": d_model, "n_layers": n_layers, "lr": lr,
                "weight_decay": weight_decay, "schedule": schedule, "warmup_frac": warmup_frac,
                "gate_lr": cfg.gate_lr, "temperature": temperature, "ce_weight": cfg.ce_weight,
                "steps": steps, "batch": batch, "warmup_steps": cfg.warmup_steps,
                "rag_context_tokens": cfg.rag_context_tokens, "val_frac": val_frac,
                "split_mode": split_mode,
                "eval_n": eval_n, "augment": augment, "subsample": subsample,
                "cache_teacher": cache_teacher,
                "placement": placement, "injection_port": injection_port,
                "grad_clip": grad_clip, "grad_accum": grad_accum,
                "effective_batch": batch * grad_accum,
                "ema_decay": ema_decay,
                "gate_mode": gate_mode, "seed": seed, "model": model,
                "dataset": dataset, "snapshot": snapshot,
                "n_train": len(train_tuples), "n_val": len(val_rows),
                "trainable_params": sum(
                    p.numel() for p in trainer.model.parameters() if p.requires_grad
                ),
            },
        )

    # When augmenting, evaluate under a HELD-OUT prompt (decoupling test); else the training prompt.
    eval_system = HELD_OUT_EVAL_SYSTEM if augment else TEACHER_SYSTEM
    if augment:
        rprint(f"[cyan]prompt augmentation ON[/] ({len(AUGMENT_SYSTEMS)} systems); "
               f"eval under HELD-OUT prompt: {eval_system!r}")

    last_metrics: dict = {}

    # Build the auxiliary eval sets ONCE up front so report() can score them EVERY checkpoint
    # (trajectories, not just end-of-run scalars):
    #  - held-IN: a sample of TRAINED questions -> watch the generalization gap (overfit) develop.
    #  - PopQA: UNSEEN entities -> watch external entity-generalization over training.
    from conceptformer.train.harness import is_answerable

    held_in_rows = [r for r in train_rows if is_answerable(r) and r.subject_qid in sg_by_qid]
    train_sample = rng.sample(held_in_rows, min(len(eval_val), len(held_in_rows)))
    held_in_eval = (
        trainer.build_eval(
            train_sample, sg_by_qid, eval_system=eval_system, gen_batch=eval_gen_batch
        )
        if train_sample
        else None
    )

    popqa_items: list = []
    if popqa_eval:
        from conceptformer.data.benchmarks import load_popqa
        from conceptformer.eval.evalsets import popqa_eval_items

        popqa_sgs = {
            sg.center.qid: sg for sg in iter_subgraphs(settings.snapshots_dir / popqa_snapshot)
        }
        # Fixed-seed frozen subset (same questions for every run/seed/config → pairable).
        popqa_items = popqa_eval_items(load_popqa(), popqa_sgs, n=popqa_eval)

    def report(tag: str, step: int = 0) -> dict:
        m = trainer.evaluate()  # held-out (the default stored set)
        last_metrics.update(m)
        log: dict = {
            "held_out/concept_acc": m["concept_acc"],
            "held_out/base_acc": m["base_acc"],
            "held_out/teacher_acc": m["teacher_acc"],
            "held_out/val_kl": m["val_kl"],
        }
        line = (
            f"  [{tag}] val_KL={m['val_kl']:.3f}  [bold]concept_acc={m['concept_acc']:.1%}[/]  "
            f"base={m['base_acc']:.1%}  teacher(RAG)={m['teacher_acc']:.1%}  (n={m['n_acc']})"
        )
        if held_in_eval is not None:
            mi = trainer.evaluate(held_in_eval)
            last_metrics["held_in_concept_acc"] = mi["concept_acc"]
            last_metrics["held_in_val_kl"] = mi["val_kl"]
            log["held_in/concept_acc"] = mi["concept_acc"]
            log["held_in/base_acc"] = mi["base_acc"]
            log["held_in/teacher_acc"] = mi["teacher_acc"]
            log["held_in/val_kl"] = mi["val_kl"]
            # gen-gap > 0 => fits trained questions better than held-out (overfitting signal).
            log["gen_gap/concept_acc"] = mi["concept_acc"] - m["concept_acc"]
            log["gen_gap/val_kl"] = m["val_kl"] - mi["val_kl"]
            line += f"  | held_in={mi['concept_acc']:.1%}"
        if popqa_items:
            pm = trainer.evaluate_popqa(
                popqa_items, eval_system=eval_system, cache_brackets=True,
                gen_batch=eval_gen_batch,
            )
            for kk in ("concept_acc", "base_acc", "teacher_acc"):
                last_metrics[f"popqa_{kk}"] = pm[kk]
                log[f"popqa/{kk}"] = pm[kk]
            line += f"  | popqa={pm['concept_acc']:.1%}"
        rprint(line)
        if wb:
            gates = trainer.gate_values()
            if gates:  # empty when gate_mode="none"
                log["gate/max"] = max(gates)
                log["gate/min"] = min(gates)
                log["gate/mean"] = sum(gates) / len(gates)
                log["gate/abs_mean"] = sum(abs(g) for g in gates) / len(gates)
                for i, g in enumerate(gates):  # per-token gate config (was only in checkpoints)
                    log[f"gate/t{i}"] = g
            # concept-vector norm on a fixed eval entity (encoder output magnitude over training).
            if eval_val:
                log["concept/norm"] = trainer.concept_norm(sg_by_qid[eval_val[0].subject_qid])
            # both param-group LRs (encoder + gate may differ / be scheduled independently).
            for grp in trainer.optimizer.param_groups:
                log[f"train/lr_{grp.get('name', 'g')}"] = grp["lr"]
            log["train/lr"] = trainer.optimizer.param_groups[0]["lr"]
            wb.log(log, step=step)
        return m

    accum = max(1, grad_accum)
    if subsample:
        rprint("[dim]subsample mode: features cached, teacher facts re-sampled each step…[/dim]")
        train_pool: list = train_tuples  # raw 4-tuples; teacher rebuilt live per step
        step_fn = trainer.step
        accum_fn = trainer.step_accum
    else:
        # The teacher-hidden cache costs ~path_len x d_llm x 4 B per row in HOST memory
        # (~160 KB/row at d=2048): fine at 10k-corpus scale, but at ~800k rows it exceeds the
        # 124 GB box and the kernel OOM-kills the run 2h into preprocessing, silently. Refuse
        # rather than warn: every large-corpus run must pass --no-cache-teacher deliberately.
        if cache_teacher:
            est_gb = len(train_tuples) * 20 * trainer.bb.d_model * 4 / 1e9
            if est_gb > 24:
                raise typer.BadParameter(
                    f"teacher-hidden cache would need ~{est_gb:.0f} GB host RAM for "
                    f"{len(train_tuples)} rows; rerun with --no-cache-teacher"
                )
        rprint("[dim]preprocessing (featurize + tokenize once)…[/dim]")
        train_pool = trainer.prepare(train_tuples)  # hoists CPU work out of the training loop
        step_fn = trainer.step_prepared
        accum_fn = trainer.step_prepared_accum
    eff_batch = batch * accum  # sample the whole effective batch, then split into `accum` micros
    trainer.setup_eval(  # brackets computed once
        eval_val, sg_by_qid, eval_system=eval_system, gen_batch=eval_gen_batch
    )
    report("init", 0)
    # Best-held-out checkpoint: at a generous horizon the model can overfit PAST its peak (F4), so
    # the FINAL weights may be worse than the best. Save the best-so-far separately (overwrites one
    # file) when held-out improves; that is the "checkpoint-selected" model downstream evals need.
    best_ho = -1.0
    best_path = settings.data_root / "checkpoints" / f"{checkpoint}_best.pt" if checkpoint else None
    # Stored in checkpoints so eval-final can reconstruct this run's exact train/val split
    # (legacy checkpoints without it are assumed question-mode split with the training seed).
    # "model" guards against evaluating with the wrong backbone: different backbones can share
    # d_llm (Qwen3-0.6B and Qwen3.5-0.8B are both 1024), so a mismatch loads silently.
    split_meta = {
        "split_mode": split_mode, "split_seed": seed, "val_frac": val_frac,
        "dataset": dataset, "snapshot": snapshot, "model": model,
    }
    for s in range(1, steps + 1):
        sample = rng.sample(train_pool, min(eff_batch, len(train_pool)))
        loss = accum_fn(split_microbatches(sample, accum)) if accum > 1 else step_fn(sample)
        if wb and s % 25 == 0:
            step_log = {
                "train/loss": loss,
                "train/grad_norm": trainer.last_grad_norm,
                "train/lr": trainer.optimizer.param_groups[0]["lr"],
            }
            for grp in trainer.optimizer.param_groups:
                step_log[f"train/lr_{grp.get('name', 'g')}"] = grp["lr"]
            wb.log(step_log, step=s)
        if s % eval_every == 0 or s == steps:
            m = report(f"step {s}", s)
            if best_path is not None and m["concept_acc"] > best_ho:
                best_ho = m["concept_acc"]
                # EMA off for this config → current == evaluated
                trainer.save_checkpoint(best_path, meta=split_meta)

    # held-in / PopQA were scored every checkpoint inside report() (full trajectories in W&B).
    # Echo the final overfit-vs-underfit read and mirror the converged values into the summary.
    hi_final = last_metrics.get("held_in_concept_acc")
    if hi_final is not None:
        rprint(
            f"  [held-IN final] concept_acc={hi_final:.1%}  "
            f"KL={last_metrics.get('held_in_val_kl', 0.0):.3f} — "
            "high held-in + low held-out = overfit; both low = undertrained"
        )
    if popqa_items:
        rprint(
            f"  [PopQA final, UNSEEN n={len(popqa_items)}] "
            f"concept_acc={last_metrics.get('popqa_concept_acc', 0.0):.1%}  "
            f"base={last_metrics.get('popqa_base_acc', 0.0):.1%}  "
            f"teacher(RAG)={last_metrics.get('popqa_teacher_acc', 0.0):.1%}"
        )
    if wb:
        wb.summary["held_out/concept_acc_final"] = last_metrics.get("concept_acc")
        if best_path is not None:
            wb.summary["held_out/concept_acc_best"] = best_ho  # the checkpoint-selected value
        if hi_final is not None:
            wb.summary["held_in/concept_acc"] = hi_final
            wb.summary["held_in/val_kl"] = last_metrics.get("held_in_val_kl")
        if popqa_items:
            for kk in ("concept_acc", "base_acc", "teacher_acc"):
                wb.summary[f"popqa/{kk}"] = last_metrics.get(f"popqa_{kk}")

    if checkpoint:
        ckpt_path = settings.data_root / "checkpoints" / f"{checkpoint}.pt"
        trainer.save_checkpoint(ckpt_path, meta=split_meta)
        rprint(f"[green]saved checkpoint[/] → {ckpt_path}")
        if wb:
            # Push the trained encoder+gate to W&B as a durable, versioned model artifact so the
            # checkpoint behind every result survives this machine (data/ is git-ignored & local).
            art = _wandb.Artifact(
                checkpoint,
                type="model",
                metadata={
                    "k": k, "d_model": d_model, "n_layers": n_layers, "steps": steps,
                    "dataset": dataset, "snapshot": snapshot, "subsample": subsample,
                    "augment": augment, "placement": placement, "seed": seed,
                    "held_out_concept_acc": last_metrics.get("concept_acc"),
                    "held_in_concept_acc": last_metrics.get("held_in_concept_acc"),
                    "popqa_concept_acc": last_metrics.get("popqa_concept_acc"),
                },
            )
            art.add_file(str(ckpt_path))
            if log_artifact_resilient(wb, art, f"model:{checkpoint}"):
                rprint(f"[green]logged W&B artifact[/] model:{checkpoint}")
        # The checkpoint-selected (best-held-out) model is the one downstream evals should use; push
        # it as a separate artifact so it is durable independently of the final-step weights.
        if wb and best_path is not None and best_path.exists():
            best_art = _wandb.Artifact(
                f"{checkpoint}_best", type="model",
                metadata={"k": k, "held_out_concept_acc_best": best_ho, "selected": "best_ho"},
            )
            best_art.add_file(str(best_path))
            if log_artifact_resilient(wb, best_art, f"model:{checkpoint}_best"):
                rprint(f"[green]logged W&B artifact[/] model:{checkpoint}_best (ho={best_ho:.3f})")

    if wb:
        wb.finish()

    if metrics_out:
        ho_keys = ("concept_acc", "base_acc", "teacher_acc", "val_kl")
        result = {
            "config": {"k": k, "steps": steps, "batch": batch, "augment": augment, "seed": seed},
            "held_out": {kk: last_metrics.get(kk) for kk in ho_keys},
            "held_in": {
                "concept_acc": last_metrics.get("held_in_concept_acc"),
                "val_kl": last_metrics.get("held_in_val_kl"),
            },
            "popqa": (
                {
                    kk: last_metrics.get(f"popqa_{kk}")
                    for kk in ("concept_acc", "base_acc", "teacher_acc")
                }
                if popqa_items
                else None
            ),
        }
        out_path = Path(metrics_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        rprint(f"[green]metrics[/] → {out_path}")

    rprint("[green]done[/] — concept_acc on HELD-OUT questions is the generalization signal.")


def _load_trained_checkpoint(
    checkpoint: str, model: str, device: str, *, use_generation_cache: bool = False
) -> tuple:
    """Load a trained checkpoint into an eval-ready trainer (pulls the W&B artifact if needed).

    Returns ``(trainer, blob, chat)``: ``blob`` carries ``config`` + optional ``meta`` (split
    provenance for reconstructing the run's exact train/val split); ``chat`` is the underlying
    ``ChatModel`` for text-bracket generation. ``use_generation_cache`` wires the sqlite greedy
    cache in, so checkpoint-independent base/RAG brackets are paid once across re-evals.
    """
    import torch

    from conceptformer.cache import KVCache
    from conceptformer.model.backbone import Backbone
    from conceptformer.model.chat import ChatModel
    from conceptformer.train.trainer import ConceptTrainer, TrainConfig

    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        ckpt_path = settings.data_root / "checkpoints" / f"{checkpoint}.pt"
    if not ckpt_path.exists():
        # Fall back to the W&B model artifact so a result is reproducible without the local file.
        import wandb as _wandb

        rprint(f"[dim]checkpoint not local; pulling W&B artifact model:{checkpoint}:latest…[/dim]")
        art = _wandb.Api().artifact(
            f"university-of-zurich/conceptformer-v2/{checkpoint}:latest", type="model"
        )
        ckpt_path = Path(art.download()) / f"{checkpoint}.pt"
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = TrainConfig(**blob["config"])
    trained_model = (blob.get("meta") or {}).get("model")
    if trained_model and trained_model != model:
        # Different backbones can share d_llm, so the state dict would load silently — refuse.
        raise typer.BadParameter(
            f"checkpoint was trained on {trained_model!r} but --model is {model!r}"
        )
    cache = KVCache(settings.generation_cache_path) if use_generation_cache else None
    chat = ChatModel(model, device=device, cache=cache)
    trainer = ConceptTrainer(Backbone(chat), cfg)
    trainer.model.load_state_dict(blob["model"])
    trainer.model.eval()
    rprint(
        f"[bold]loaded[/] {ckpt_path.name}  "
        f"(k={cfg.k}, d_model={cfg.d_model}, n_layers={cfg.n_layers})"
    )
    return trainer, blob, chat


def _reconstruct_split(
    rows: list, blob: dict, fallback_seed: int, fallback_val_frac: float
) -> tuple[list, list, dict]:
    """Rebuild the train/val split a checkpoint was trained under, from its stored meta.

    Legacy checkpoints (no ``meta``) used the question-level split seeded with the training
    seed; new ones record mode/seed/frac explicitly. Getting this wrong silently scores
    trained questions as "held-out", so it lives in one place.
    """
    from conceptformer.train.harness import (
        split_by_held_out_facts,
        split_by_held_out_questions,
    )

    meta = blob.get("meta") or {}
    cfg_seed = int(blob.get("config", {}).get("seed", fallback_seed))
    split_seed = int(meta.get("split_seed", cfg_seed))
    val_frac = float(meta.get("val_frac", fallback_val_frac))
    mode = str(meta.get("split_mode", "question"))
    splitter = split_by_held_out_facts if mode == "fact" else split_by_held_out_questions
    train_rows, val_rows = splitter(rows, val_frac=val_frac, seed=split_seed)
    return train_rows, val_rows, {"split_mode": mode, "split_seed": split_seed,
                                  "val_frac": val_frac}


@app.command("eval-final")
def eval_final(
    checkpoint: Annotated[str, typer.Option(help="trained checkpoint name or path")],
    dataset: Annotated[str, typer.Option()] = "cftrain_qa_100k",
    snapshot: Annotated[str, typer.Option()] = "cftrain_100k",
    popqa_snapshot: Annotated[str, typer.Option()] = "popqa_full",
    held_out_n: Annotated[int, typer.Option(help="strict held-out questions (0 = all)")] = 2000,
    popqa_n: Annotated[int, typer.Option(help="PopQA questions (0 = FULL benchmark)")] = 0,
    gen_batch: Annotated[int, typer.Option()] = 64,
    max_new: Annotated[int, typer.Option()] = 32,
    model: Annotated[str, typer.Option()] = "Qwen/Qwen3-0.6B",
    device: Annotated[str, typer.Option()] = "cuda",
    out_dir: Annotated[str, typer.Option(help="report dir (default data/analysis/eval_final)")]
    = "",
) -> None:
    """Definitive post-hoc eval of a checkpoint: strict held-out + FULL PopQA, per-item dumps.

    Fixes the historical eval defects in one place (methods note M7): frozen fixed-seed eval
    sets shared by every checkpoint (so per-item dumps pair across runs — eval/stats.py),
    Wilson CIs instead of bare points, a paraphrase-leakage-free held-out subset
    (strict_val_subset), and the WHOLE PopQA benchmark (n=200 sampling noise ~3.5 pt shrinks
    to ~0.4 pt at n~14k). Base/RAG brackets are checkpoint-independent and cached on disk
    (generations.sqlite), so re-evaluating the next checkpoint only pays the concept pass.
    """
    from conceptformer.generate.dataset import load_cftrain_qa

    trainer, blob, chat = _load_trained_checkpoint(
        checkpoint, model, device, use_generation_cache=True
    )
    rows = load_cftrain_qa(settings.data_root / "cf_train" / dataset / "qa_distill.jsonl")
    train_rows, val_rows, split_info = _reconstruct_split(rows, blob, 0, 0.3)
    _definitive_eval(
        trainer, chat, train_rows, val_rows, split_info,
        dataset=dataset, snapshot=snapshot, popqa_snapshot=popqa_snapshot,
        held_out_n=held_out_n, popqa_n=popqa_n, gen_batch=gen_batch, max_new=max_new,
        report_name=checkpoint, report_extra={"checkpoint": checkpoint,
                                              "config": blob["config"]},
        out_dir=out_dir,
    )


def _definitive_eval(
    trainer: ConceptTrainer, chat: ChatModel,
    train_rows: list, val_rows: list, split_info: dict, *,
    dataset: str, snapshot: str, popqa_snapshot: str,
    held_out_n: int, popqa_n: int, gen_batch: int, max_new: int,
    report_name: str, report_extra: dict, out_dir: str,
) -> None:
    """Shared eval core for eval-final / eval-untrained-injection (same sets, same dumps).

    Scores concept/base/RAG on the strict (fact-leakage-free) held-out subset + the frozen
    PopQA set, writes per-item JSONL (for paired stats) and a summary with Wilson CIs.
    """
    from conceptformer.data.snapshot import iter_subgraphs
    from conceptformer.eval.evalsets import EVAL_SAMPLE_SEED, popqa_eval_items, sample_rows
    from conceptformer.eval.metrics import popqa_official
    from conceptformer.eval.stats import summarize_accuracy
    from conceptformer.generate.signal import answer_ok
    from conceptformer.train.harness import is_answerable, strict_val_subset
    from conceptformer.train.trainer import TEACHER_SYSTEM
    from conceptformer.verbalize import verbalize_with_answer

    sg_by_qid = {sg.center.qid: sg for sg in iter_subgraphs(settings.snapshots_dir / snapshot)}
    answerable_val = [
        r for r in val_rows
        if is_answerable(r) and r.subject_qid in sg_by_qid and r.accepted_answers
    ]
    strict = strict_val_subset(train_rows, answerable_val)
    leak_frac = 1.0 - (len(strict) / max(1, len(answerable_val)))
    held_out = [
        (sg_by_qid[r.subject_qid], r.question, r.accepted_answers, r.answer_qid)
        for r in sample_rows(strict, held_out_n)
    ]

    popqa_sgs = {
        sg.center.qid: sg for sg in iter_subgraphs(settings.snapshots_dir / popqa_snapshot)
    }
    popqa = popqa_eval_items(load_popqa(), popqa_sgs, n=popqa_n)
    rprint(
        f"[bold]eval[/] {report_name}: strict held-out n={len(held_out)} "
        f"(fact-leaky val rows removed: {leak_frac:.1%}), PopQA n={len(popqa)}"
    )

    def facts_for(sg: Subgraph, answer_qid: str | None) -> str:
        budget = int(trainer.cfg.rag_context_tokens)
        return verbalize_with_answer(sg, answer_qid, chat.count_tokens, budget)

    def eval_items(name: str, items: list, official: bool) -> tuple[dict, list[dict]]:
        concept_preds: list[str] = []
        for i in range(0, len(items), gen_batch):
            chunk = items[i : i + gen_batch]
            concept_preds += trainer.generate_student_batch(
                [(sg, q) for sg, q, _, _ in chunk], max_new, TEACHER_SYSTEM
            )
        base_preds = chat.generate_batch(
            [(TEACHER_SYSTEM, q) for _, q, _, _ in items],
            max_new_tokens=max_new, batch_size=gen_batch,
        )
        rag_preds = chat.generate_batch(
            [(TEACHER_SYSTEM, f"{facts_for(sg, aq)}\n\n{q}") for sg, q, _, aq in items],
            max_new_tokens=max_new, batch_size=gen_batch,
        )
        per_item: list[dict] = []
        for (sg, q, gold, _), cp, bp, rp in zip(
            items, concept_preds, base_preds, rag_preds, strict=True
        ):
            row = {
                "set": name, "subject_qid": sg.center.qid, "question": q,
                "gold": list(gold),
                "concept": answer_ok(cp, gold), "base": answer_ok(bp, gold),
                "rag": answer_ok(rp, gold), "concept_pred": cp,
            }
            if official:  # the published PopQA metric, for literature comparability
                row["concept_official"] = popqa_official(cp, gold)
                row["base_official"] = popqa_official(bp, gold)
            per_item.append(row)
        summary = {
            "n": len(per_item),
            "concept": summarize_accuracy([r["concept"] for r in per_item]),
            "base": summarize_accuracy([r["base"] for r in per_item]),
            "rag": summarize_accuracy([r["rag"] for r in per_item]),
        }
        if official:
            summary["concept_official"] = summarize_accuracy(
                [r["concept_official"] for r in per_item]
            )
        return summary, per_item

    report_dir = (
        Path(out_dir) if out_dir
        else settings.data_root / "analysis" / "eval_final" / report_name
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        **report_extra, "split": split_info,
        "eval_sample_seed": EVAL_SAMPLE_SEED, "dataset": dataset,
        "strict_leak_frac_removed": round(leak_frac, 4),
    }
    for name, items, official in (("held_out", held_out, False), ("popqa", popqa, True)):
        if not items:
            continue
        summary, per_item = eval_items(name, items, official)
        report[name] = summary
        with (report_dir / f"{name}_items.jsonl").open("w", encoding="utf-8") as fh:
            for row in per_item:
                fh.write(json.dumps(row) + "\n")
        c, b, r = summary["concept"], summary["base"], summary["rag"]
        rprint(
            f"  [bold]{name}[/] (n={summary['n']}): concept={c['acc']:.1%} "
            f"[{c['ci95'][0]:.1%}, {c['ci95'][1]:.1%}]  base={b['acc']:.1%}  rag={r['acc']:.1%}"
        )
    (report_dir / "summary.json").write_text(json.dumps(report, indent=2))
    rprint(f"[green]wrote[/] {report_dir}/summary.json (+ per-item jsonl for paired stats)")


@app.command("build-metaqa-snapshot")
def build_metaqa_snapshot(
    kb: Annotated[str, typer.Option(help="path to MetaQA kb.txt (subject|relation|object)")],
    name: Annotated[str, typer.Option(help="snapshot name")] = "metaqa",
) -> None:
    """Convert the MetaQA movie KB into a ConceptFormer snapshot (cross-graph transfer)."""
    import hashlib

    from conceptformer.data.metaqa import build_subgraphs

    out_dir = settings.snapshots_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256()
    n = 0
    with Path(kb).open(encoding="utf-8") as kb_fh, \
            (out_dir / "subgraphs.jsonl").open("w", encoding="utf-8") as fh:
        for sg in build_subgraphs(kb_fh):
            line = sg.model_dump_json()
            fh.write(line + "\n")
            sha.update(line.encode("utf-8"))
            n += 1
    manifest = {"name": name, "source": "MetaQA kb.txt (CC BY 3.0)", "n_subgraphs": n,
                "sha256": sha.hexdigest()}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    rprint(f"[green]wrote[/] {out_dir} ({n} subgraphs, sha {sha.hexdigest()[:12]})")


@app.command("translate-eval-set")
def translate_eval_set(
    source: Annotated[str, typer.Option(help="popqa | a MetaQA-format QA file path")],
    out: Annotated[str, typer.Option(help="output TranslatedQA JSONL path")],
    lang: Annotated[str, typer.Option(help="target language code")] = "de",
    n: Annotated[int, typer.Option(help="questions to translate (0 = all)")] = 0,
    base_url: Annotated[str, typer.Option(help="vLLM server")] = "http://localhost:8000/v1",
) -> None:
    """Translate an eval set to ``lang`` for the multilingual transfer eval (eval-only).

    Keeps the entity mention in English (mention-anchored splice needs a locatable span), then
    for PopQA (Wikidata-linked) fetches the target-language subject label and answer aliases so
    the localized-mention and answer-language conditions are available. MetaQA entities are not
    Wikidata-linked, so its rows get the reduced set (English mention + English answer aliases).
    """
    import asyncio

    from conceptformer.cache import KVCache
    from conceptformer.data.benchmarks import load_popqa
    from conceptformer.data.metaqa import load_metaqa_qa
    from conceptformer.data.multilingual import (
        TranslatedQA,
        localize_mention,
        mention_preserved,
    )
    from conceptformer.data.wikidata import WikidataClient
    from conceptformer.eval.evalsets import sample_rows
    from conceptformer.generate.translate import GemmaTranslator

    if source == "popqa":
        rows = load_popqa()
        is_wikidata = True
    else:
        with Path(source).open(encoding="utf-8") as fh:
            rows = load_metaqa_qa(fh)
        is_wikidata = False
    rows = sample_rows(rows, n)  # prefix-consistent frozen subset; 0 = all
    rprint(f"translating {len(rows)} {source} questions -> {lang}…")

    # Entity mention per row: PopQA questions embed the subject as a QID-linked label; MetaQA's
    # subject_qid IS the surface string. We tell the translator the English mention so it can keep
    # it verbatim, and later verify it survived (mention-anchored splice needs the span).
    gen_cache = KVCache(settings.generation_cache_path)
    translator = GemmaTranslator(base_url=base_url, lang=lang, cache=gen_cache)
    results = asyncio.run(translator.translate([r.question for r in rows]))
    gen_cache.close()

    de_label: dict[str, str] = {}
    de_answer: dict[str, list[str]] = {}
    if is_wikidata:
        subj = sorted({r.subject_qid for r in rows})
        ans = sorted({r.answer_qid for r in rows if r.answer_qid})
        with WikidataClient() as wd:
            de_label = {q: (v[0] if v else "") for q, v in wd.surface_forms(subj, lang).items()}
            de_answer = wd.surface_forms(ans, lang)

    written = kept = 0
    with Path(out).open("w", encoding="utf-8") as fh:
        for r, tr in zip(rows, results, strict=True):
            written += 1
            if tr is None:
                continue
            # The translator returns the entity's surface form it kept in the sentence; we anchor
            # on that span (verified present below) rather than parsing the question text.
            mention_en = tr.mention_translated
            if not mention_preserved(tr.question_translated, mention_en):
                continue  # span lost in translation -> cannot anchor; drop
            mention_loc = de_label.get(r.subject_qid, mention_en) or mention_en
            kept += 1
            row = TranslatedQA(
                subject_qid=r.subject_qid, lang=lang, question_source=r.question,
                question_en_mention=tr.question_translated,
                question_localized=localize_mention(
                    tr.question_translated, mention_en, mention_loc
                ),
                mention_en=mention_en, mention_localized=mention_loc,
                answer_labels_en=r.answer_labels,
                answer_labels_localized=(
                    de_answer.get(r.answer_qid or "", []) if is_wikidata else r.answer_labels
                ),
            )
            fh.write(row.model_dump_json() + "\n")
    rprint(f"[green]wrote[/] {out}: {kept}/{written} rows (dropped {written - kept} "
           f"where the entity span was lost in translation)")


@app.command("eval-transfer")
def eval_transfer(
    checkpoint: Annotated[str, typer.Option(help="trained checkpoint name or path")],
    qa: Annotated[str, typer.Option(help="path to MetaQA-format QA file (bracketed subject)")],
    snapshot: Annotated[str, typer.Option()] = "metaqa",
    benchmark: Annotated[str, typer.Option(help="benchmark tag for the report")] = "metaqa_1hop",
    n: Annotated[int, typer.Option(help="questions to score (0 = all)")] = 2000,
    gen_batch: Annotated[int, typer.Option()] = 32,
    max_new: Annotated[int, typer.Option()] = 32,
    model: Annotated[str, typer.Option()] = "Qwen/Qwen3-0.6B",
    device: Annotated[str, typer.Option()] = "cuda",
) -> None:
    """ZERO-SHOT cross-graph transfer: score a (e.g. Wikidata-trained) checkpoint on another KG.

    The encoder only ever consumes label-string embeddings, so it should transfer to any
    labeled graph if it learned graph->concept rather than source-graph idioms. Scores the
    concept condition against the frozen base (floor) and budgeted text-RAG (reference; NO
    answer guarantee -- the answer edge is whatever the plain top-of-neighborhood budget
    keeps). Per-item dumps + Wilson CIs, same conventions as eval-final.
    """
    from conceptformer.data.metaqa import load_metaqa_qa
    from conceptformer.data.snapshot import iter_subgraphs
    from conceptformer.eval.evalsets import EVAL_SAMPLE_SEED, popqa_eval_items
    from conceptformer.eval.stats import summarize_accuracy
    from conceptformer.generate.signal import answer_ok
    from conceptformer.train.trainer import TEACHER_SYSTEM
    from conceptformer.verbalize import verbalize_budgeted

    trainer, blob, chat = _load_trained_checkpoint(
        checkpoint, model, device, use_generation_cache=True
    )
    sgs = {sg.center.qid: sg for sg in iter_subgraphs(settings.snapshots_dir / snapshot)}
    with Path(qa).open(encoding="utf-8") as qa_fh:
        examples = load_metaqa_qa(qa_fh)
    items = popqa_eval_items(examples, sgs, n=n)  # frozen fixed-seed subset, pairable
    rprint(f"[bold]zero-shot transfer[/] {checkpoint} -> {benchmark}: n={len(items)} "
           f"(of {len(examples)} questions, {len(sgs)} entities in graph)")

    concept_preds: list[str] = []
    for i in range(0, len(items), gen_batch):
        chunk = items[i : i + gen_batch]
        concept_preds += trainer.generate_student_batch(
            [(sg, q) for sg, q, _, _ in chunk], max_new, TEACHER_SYSTEM
        )
    base_preds = chat.generate_batch(
        [(TEACHER_SYSTEM, q) for _, q, _, _ in items],
        max_new_tokens=max_new, batch_size=gen_batch,
    )
    budget = int(trainer.cfg.rag_context_tokens)
    rag_preds = chat.generate_batch(
        [(TEACHER_SYSTEM, f"{verbalize_budgeted(sg, chat.count_tokens, budget)}\n\n{q}")
         for sg, q, _, _ in items],
        max_new_tokens=max_new, batch_size=gen_batch,
    )
    per_item: list[dict] = []
    concept_flags: list[bool] = []
    base_flags: list[bool] = []
    rag_flags: list[bool] = []
    for (sg, q, gold, _), cp, bp, rp in zip(
        items, concept_preds, base_preds, rag_preds, strict=True
    ):
        c_ok, b_ok, r_ok = answer_ok(cp, gold), answer_ok(bp, gold), answer_ok(rp, gold)
        concept_flags.append(c_ok)
        base_flags.append(b_ok)
        rag_flags.append(r_ok)
        per_item.append(
            {"benchmark": benchmark, "subject": sg.center.qid, "question": q,
             "gold": list(gold), "concept": c_ok, "base": b_ok, "rag": r_ok,
             "concept_pred": cp}
        )
    report = {
        "checkpoint": checkpoint, "benchmark": benchmark, "snapshot": snapshot,
        "eval_sample_seed": EVAL_SAMPLE_SEED, "config": blob["config"],
        "n": len(per_item),
        "concept": summarize_accuracy(concept_flags),
        "base": summarize_accuracy(base_flags),
        "rag": summarize_accuracy(rag_flags),
    }
    out_dir = settings.data_root / "analysis" / "transfer" / f"{checkpoint}__{benchmark}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "items.jsonl").open("w", encoding="utf-8") as fh:
        for row in per_item:
            fh.write(json.dumps(row) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2))
    c, b, r = report["concept"], report["base"], report["rag"]
    rprint(f"  concept={c['acc']:.1%} [{c['ci95'][0]:.1%}, {c['ci95'][1]:.1%}]  "
           f"base={b['acc']:.1%}  rag={r['acc']:.1%}")
    rprint(f"[green]wrote[/] {out_dir}/summary.json")


@app.command("eval-untrained-injection")
def eval_untrained_injection(
    k: Annotated[int, typer.Option(help="concept slots to fill with top-k edge embeddings")] = 8,
    dataset: Annotated[str, typer.Option()] = "cftrain_qa_100k",
    snapshot: Annotated[str, typer.Option()] = "cftrain_100k",
    popqa_snapshot: Annotated[str, typer.Option()] = "popqa_full",
    held_out_n: Annotated[int, typer.Option(help="strict held-out questions (0 = all)")] = 2000,
    popqa_n: Annotated[int, typer.Option(help="PopQA questions (0 = FULL benchmark)")] = 0,
    split_seed: Annotated[int, typer.Option(help="split to eval against (match eval-final)")] = 0,
    split_mode: Annotated[str, typer.Option(help="fact | question (match the checkpoint)")]
    = "question",
    val_frac: Annotated[float, typer.Option()] = 0.3,
    gen_batch: Annotated[int, typer.Option()] = 64,
    max_new: Annotated[int, typer.Option()] = 32,
    model: Annotated[str, typer.Option()] = "Qwen/Qwen3-0.6B",
    device: Annotated[str, typer.Option()] = "cuda",
    out_dir: Annotated[str, typer.Option()] = "",
) -> None:
    """The no-encoder control: top-k edge label-embeddings spliced in WITHOUT any training.

    Fills the same k concept slots with mean-pooled (property, neighbor) embeddings of the
    entity's top-k PageRank edges (model/baselines.py). The trained resampler must beat this,
    or the learned graph->concept mapping isn't earning its parameters. Same frozen eval sets
    and per-item dumps as eval-final, so the comparison is paired.
    """
    from conceptformer.cache import KVCache
    from conceptformer.generate.dataset import load_cftrain_qa
    from conceptformer.model.backbone import Backbone
    from conceptformer.model.baselines import TopKMeanEdgeBaseline
    from conceptformer.model.chat import ChatModel
    from conceptformer.train.harness import (
        split_by_held_out_facts,
        split_by_held_out_questions,
    )
    from conceptformer.train.trainer import ConceptTrainer, TrainConfig

    chat = ChatModel(model, device=device, cache=KVCache(settings.generation_cache_path))
    trainer = ConceptTrainer(
        Backbone(chat), TrainConfig(k=k, seed=split_seed),
        concept_model=TopKMeanEdgeBaseline(k),
    )
    trainer.model.eval()

    rows = load_cftrain_qa(settings.data_root / "cf_train" / dataset / "qa_distill.jsonl")
    splitter = split_by_held_out_facts if split_mode == "fact" else split_by_held_out_questions
    train_rows, val_rows = splitter(rows, val_frac=val_frac, seed=split_seed)
    split_info = {"split_mode": split_mode, "split_seed": split_seed, "val_frac": val_frac}
    name = f"untrained_topk_mean_k{k}"
    _definitive_eval(
        trainer, chat, train_rows, val_rows, split_info,
        dataset=dataset, snapshot=snapshot, popqa_snapshot=popqa_snapshot,
        held_out_n=held_out_n, popqa_n=popqa_n, gen_batch=gen_batch, max_new=max_new,
        report_name=name, report_extra={"baseline": name, "k": k},
        out_dir=out_dir,
    )


@app.command("eval-prompt-robustness")
def eval_prompt_robustness(
    checkpoint: Annotated[str, typer.Option(help="checkpoint name under data/checkpoints or path")],
    dataset: Annotated[str, typer.Option()] = "cftrain_qa_10k",
    snapshot: Annotated[str, typer.Option()] = "cftrain_10k",
    popqa_snapshot: Annotated[str, typer.Option()] = "popqa_full",
    eval_n: Annotated[int, typer.Option(help="held-out questions scored per prompt")] = 200,
    popqa_n: Annotated[int, typer.Option(help="PopQA (unseen entity) questions per prompt")] = 200,
    val_frac: Annotated[float, typer.Option()] = 0.3,
    seed: Annotated[int, typer.Option()] = 0,
    model: Annotated[str, typer.Option()] = "Qwen/Qwen3-0.6B",
    device: Annotated[str, typer.Option()] = "cuda",
) -> None:
    """Score a trained concept-token checkpoint under MANY system prompts to measure robustness.

    A single-prompt accuracy hides whether the concept vectors are coupled to the training prompt.
    This evals held-out questions AND unseen-entity PopQA under each augmentation prompt plus a
    held-OUT prompt, and reports the spread (mean / std / worst-case) -- low variance and a high
    floor = prompt-robust vectors. Use it to compare two checkpoints (e.g. subsample on vs off).
    """
    import statistics

    from conceptformer.data.benchmarks import load_popqa
    from conceptformer.data.snapshot import iter_subgraphs
    from conceptformer.eval.evalsets import popqa_eval_items, sample_rows
    from conceptformer.generate.dataset import load_cftrain_qa
    from conceptformer.train.trainer import (
        AUGMENT_SYSTEMS,
        HELD_OUT_EVAL_SYSTEM,
        TEACHER_SYSTEM,
    )

    trainer, blob, _ = _load_trained_checkpoint(checkpoint, model, device)

    qa_dir = settings.data_root / "cf_train" / dataset
    rows = load_cftrain_qa(qa_dir / "qa_distill.jsonl")
    sg_by_qid = {sg.center.qid: sg for sg in iter_subgraphs(settings.snapshots_dir / snapshot)}
    # Reconstruct THIS checkpoint's split, then sample the frozen fixed-seed eval subset from
    # ITS val side (identical across checkpoints -> results pair).
    _, val_rows, _split_info = _reconstruct_split(rows, blob, seed, val_frac)
    val_rows = [r for r in val_rows if r.subject_qid in sg_by_qid]
    eval_val = sample_rows(val_rows, eval_n)

    popqa_sgs = {
        sg.center.qid: sg for sg in iter_subgraphs(settings.snapshots_dir / popqa_snapshot)
    }
    # Frozen fixed-seed PopQA subset — identical across checkpoints, so results pair.
    popqa_items = popqa_eval_items(load_popqa(), popqa_sgs, n=popqa_n)

    # TEACHER_SYSTEM + the 5 augmentation prompts were all SEEN-style training prompts here (this
    # checkpoint trained under TEACHER_SYSTEM only); HELD_OUT_EVAL_SYSTEM is a never-seen phrasing.
    prompts = [("teacher", TEACHER_SYSTEM), ("held_out_prompt", HELD_OUT_EVAL_SYSTEM)]
    prompts += [(f"aug{i+1}", s) for i, s in enumerate(AUGMENT_SYSTEMS)]

    rprint(f"[bold]prompt-robustness[/] over {len(prompts)} system prompts "
           f"(held-out n={len(eval_val)}, PopQA n={len(popqa_items)}):")
    ho_accs, pq_accs = [], []
    rprint(f"  {'prompt':>16} {'held_out':>9} {'popqa':>7}")
    for name, system in prompts:
        es = trainer.build_eval(eval_val, sg_by_qid, eval_system=system)
        ho = trainer.evaluate(es)["concept_acc"]
        pq = trainer.evaluate_popqa(popqa_items, eval_system=system)["concept_acc"]
        ho_accs.append(ho)
        pq_accs.append(pq)
        rprint(f"  {name:>16} {ho:>9.1%} {pq:>7.1%}")

    def stats(xs: list[float]) -> str:
        sd = statistics.pstdev(xs) if len(xs) > 1 else 0.0
        return f"mean={statistics.mean(xs):.1%}  std={sd:.1%}  min={min(xs):.1%}  max={max(xs):.1%}"

    rprint(f"  [bold]held_out[/]: {stats(ho_accs)}")
    rprint(f"  [bold]popqa[/]:    {stats(pq_accs)}")
    rprint("[dim]low std + high min across prompts = prompt-robust concept vectors[/dim]")


@app.command("cf-graph-faithfulness")
def cf_graph_faithfulness(
    checkpoint: Annotated[str, typer.Option(help="trained checkpoint name or path")],
    dataset: Annotated[str, typer.Option()] = "cftrain_qa_100k",
    snapshot: Annotated[str, typer.Option()] = "cftrain_100k",
    n: Annotated[int, typer.Option(help="held-out single-fact questions to probe")] = 300,
    val_frac: Annotated[float, typer.Option()] = 0.3,
    seed: Annotated[int, typer.Option()] = 0,
    model: Annotated[str, typer.Option()] = "Qwen/Qwen3-0.6B",
    device: Annotated[str, typer.Option()] = "cuda",
) -> None:
    """Prove the model READS THE GRAPH (not just compresses text) via causal graph interventions.

    On baseline-correct single-fact questions, two interventions on the INPUT graph (re-encoded each
    time): (1) counterfactual SWAP of the answer edge's neighbor to a type-plausible FALSE entity —
    a graph-faithful model follows it to the false answer (a text-memoriser, or one leaning on the
    frozen LLM's parametric knowledge, would not); (2) ABLATE the answer edge — the model should
    lose THAT answer while a removed UNRELATED edge leaves it intact (edges encoded separably).
    """
    import random

    from conceptformer.data.snapshot import iter_subgraphs
    from conceptformer.eval.counterfactual import (
        ablate_neighbor,
        answer_edge_property,
        build_swap_pool,
        matches,
        pick_swap_target,
        swap_edge_neighbor,
    )
    from conceptformer.eval.probes import faithfulness_summary
    from conceptformer.generate.dataset import load_cftrain_qa
    from conceptformer.generate.signal import answer_ok
    from conceptformer.train.trainer import TEACHER_SYSTEM

    trainer, blob, _ = _load_trained_checkpoint(checkpoint, model, device)
    cfg = trainer.cfg

    rows = load_cftrain_qa(settings.data_root / "cf_train" / dataset / "qa_distill.jsonl")
    sg_by_qid = {sg.center.qid: sg for sg in iter_subgraphs(settings.snapshots_dir / snapshot)}
    _, val_rows, _split_info = _reconstruct_split(rows, blob, seed, val_frac)
    # Single-fact questions only: the answer maps to exactly one edge, so "the answer edge" is
    # well-defined for swap/ablate. (Compositional questions span >1 edge — a separate probe.)
    probes = [
        r for r in val_rows
        if r.task_type == "single" and r.answer_qid and r.subject_qid in sg_by_qid
        and any(e.neighbor.qid == r.answer_qid for e in sg_by_qid[r.subject_qid].edges)
    ]
    rng = random.Random(seed)
    rng.shuffle(probes)
    probes = probes[:n]
    pool = build_swap_pool(sg_by_qid.values())
    rprint(f"[bold]graph-faithfulness[/] k={cfg.k}; {len(probes)} single-fact probes")

    def gen(items: list[tuple]) -> list[str]:
        return trainer.generate_student_batch(items, max_new=32, system=TEACHER_SYSTEM)

    base_items = [(sg_by_qid[r.subject_qid], r.question) for r in probes]
    base_out = gen(base_items)
    correct = [answer_ok(o, r.accepted_answers) for o, r in zip(base_out, probes, strict=True)]
    # The frozen LLM with NO concepts: where it already knows the answer (memory), the swap test is
    # confounded. Splitting by base-known vs base-UNKNOWN isolates pure graph-reading.
    base_text = trainer._generate_text_batch([(TEACHER_SYSTEM, r.question) for r in probes], 32)
    base_known = [answer_ok(o, r.accepted_answers) for o, r in zip(base_text, probes, strict=True)]

    # Build the three interventions, but only on baseline-CORRECT probes (otherwise the deltas are
    # meaningless). swap_meta carries the base-known flag so swap-follow can be split by memory.
    swap_items, swap_meta = [], []  # (probe, true_neighbor, swapped_neighbor, base_known)
    abl_ans_items, abl_ans_meta = [], []
    abl_oth_items, abl_oth_meta = [], []
    for r, ok, bk in zip(probes, correct, base_known, strict=True):
        if not ok:
            continue
        aq = r.answer_qid
        if aq is None:  # guaranteed by the probe filter; this narrows it for the manipulators
            continue
        sg = sg_by_qid[r.subject_qid]
        true_n = next(e.neighbor for e in sg.edges if e.neighbor.qid == aq)
        prop = answer_edge_property(sg, aq)
        tgt = pick_swap_target(pool, prop, sg, aq, rng) if prop else None
        if tgt is not None:
            swap_items.append((swap_edge_neighbor(sg, aq, tgt), r.question))
            swap_meta.append((r, true_n, tgt, bk))
        abl_ans_items.append((ablate_neighbor(sg, aq), r.question))
        abl_ans_meta.append(r)
        others = [e.neighbor.qid for e in sg.edges if e.neighbor.qid != r.answer_qid]
        if others:
            abl_oth_items.append((ablate_neighbor(sg, rng.choice(others)), r.question))
            abl_oth_meta.append(r)

    swap_out = gen(swap_items)
    follow = [matches(o, m[2]) for o, m in zip(swap_out, swap_meta, strict=True)]
    stick = [matches(o, m[1]) for o, m in zip(swap_out, swap_meta, strict=True)]
    swap_known = [m[3] for m in swap_meta]
    abl_ans_out = gen(abl_ans_items)
    abl_ans_ok = [
        answer_ok(o, r.accepted_answers) for o, r in zip(abl_ans_out, abl_ans_meta, strict=True)
    ]
    abl_oth_out = gen(abl_oth_items)
    abl_oth_ok = [
        answer_ok(o, r.accepted_answers) for o, r in zip(abl_oth_out, abl_oth_meta, strict=True)
    ]

    summary = faithfulness_summary(
        correct, base_known, follow, stick, swap_known, abl_ans_ok, abl_oth_ok
    )
    report = {
        "checkpoint": checkpoint, "probe": "graph_faithfulness", "dataset": dataset,
        "snapshot": snapshot, "seed": seed, "config": blob["config"], **summary,
    }
    per_item: list[dict] = [
        {"probe": "baseline", "subject": r.subject_qid, "question": r.question,
         "correct": ok, "base_known": bk, "output": o}
        for r, ok, bk, o in zip(probes, correct, base_known, base_out, strict=True)
    ]
    per_item += [
        {"probe": "swap", "subject": m[0].subject_qid, "question": m[0].question,
         "true_neighbor": m[1].label or m[1].qid, "swap_target": m[2].label or m[2].qid,
         "base_known": m[3], "followed": f, "stuck": s, "output": o}
        for m, f, s, o in zip(swap_meta, follow, stick, swap_out, strict=True)
    ]
    per_item += [
        {"probe": "ablate_answer", "subject": r.subject_qid, "question": r.question,
         "correct": ok, "output": o}
        for r, ok, o in zip(abl_ans_meta, abl_ans_ok, abl_ans_out, strict=True)
    ]
    per_item += [
        {"probe": "ablate_other", "subject": r.subject_qid, "question": r.question,
         "correct": ok, "output": o}
        for r, ok, o in zip(abl_oth_meta, abl_oth_ok, abl_oth_out, strict=True)
    ]
    out_dir = settings.data_root / "analysis" / "probes" / f"{checkpoint}__faithfulness"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "items.jsonl").open("w", encoding="utf-8") as fh:
        for row in per_item:
            fh.write(json.dumps(row) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2))

    def pct(block: dict) -> str:
        return f"{block['acc']:.1%} ({block['correct']}/{block['n']})"

    rprint(f"  baseline correct: {pct(summary['baseline_correct'])}  "
           f"| base-only (no concepts) knows: {pct(summary['base_knows'])}")
    rprint("  [bold]counterfactual swap[/] (answer edge → false neighbor):")
    rprint(f"    ALL: follows swap (→ FALSE) [bold]{pct(summary['swap_follow'])}[/]  "
           f"sticks to original {pct(summary['swap_stick'])}")
    rprint(f"    [bold]base-UNKNOWN[/] (no parametric memory → clean graph test): "
           f"follows swap [bold]{pct(summary['swap_follow_base_unknown'])}[/]  "
           f"sticks {pct(summary['swap_stick_base_unknown'])}")
    rprint("  [bold]edge ablation[/] (of baseline-correct):")
    rprint(f"    correct after removing ANSWER edge: "
           f"[bold]{pct(summary['ablate_answer_correct'])}[/]  — want LOW")
    rprint(f"    correct after removing OTHER edge:  "
           f"[bold]{pct(summary['ablate_other_correct'])}[/]  — want HIGH (~baseline)")
    rprint("[dim]high swap-follow + (low answer-ablation, high other-ablation) = reads the graph "
           "edge-by-edge, not entangled text / parametric memory.[/dim]")
    rprint(f"[green]wrote[/] {out_dir}/summary.json")


@app.command("cf-capability-preservation")
def cf_capability_preservation(
    checkpoint: Annotated[str, typer.Option(help="trained checkpoint name or path")],
    dataset: Annotated[str, typer.Option()] = "cftrain_qa_100k",
    snapshot: Annotated[str, typer.Option()] = "cftrain_100k",
    n: Annotated[int, typer.Option(help="held-out CONTROL tasks to probe")] = 300,
    val_frac: Annotated[float, typer.Option()] = 0.3,
    seed: Annotated[int, typer.Option()] = 0,
    model: Annotated[str, typer.Option()] = "Qwen/Qwen3-0.6B",
    device: Annotated[str, typer.Option()] = "cuda",
) -> None:
    """Does injecting concept tokens DEGRADE the frozen LLM's normal generation? (is it preserved?)

    On held-out CONTROL tasks (the entity is named but the task is NOT about its facts, so concepts
    SHOULD be inert), compares the frozen model WITH concepts vs WITHOUT (base): greedy-agreement
    (same continuation?) and the mean KL of the next-token distribution. High agreement + low KL =
    the concept tokens don't disturb the model's normal behaviour = capability preserved.
    """
    import random

    import torch

    from conceptformer.data.snapshot import iter_subgraphs
    from conceptformer.eval.probes import capability_summary
    from conceptformer.generate.dataset import load_cftrain_qa
    from conceptformer.model.featurizer import featurize_subgraph
    from conceptformer.model.injection import build_position_ids
    from conceptformer.train.trainer import TEACHER_SYSTEM

    trainer, blob, _ = _load_trained_checkpoint(checkpoint, model, device)

    rows = load_cftrain_qa(settings.data_root / "cf_train" / dataset / "qa_distill.jsonl")
    sg_by = {sg.center.qid: sg for sg in iter_subgraphs(settings.snapshots_dir / snapshot)}
    _, val, _split_info = _reconstruct_split(rows, blob, seed, val_frac)
    ctrl = [r for r in val if r.task_type == "control" and r.subject_qid in sg_by]
    rng = random.Random(seed)
    if ctrl:
        rng.shuffle(ctrl)
        items = [(sg_by[r.subject_qid], r.question) for r in ctrl[:n]]
    else:
        # Corpora past 10k skip control-task generation; synthesize the same off-topic prompts
        # (control.py's template bank) over held-out entities so the test still runs.
        from conceptformer.generate.control import control_tasks

        val_qids = list(dict.fromkeys(r.subject_qid for r in val if r.subject_qid in sg_by))
        rng.shuffle(val_qids)
        items = []
        for qid in val_qids:
            for t in control_tasks(sg_by[qid], n=1, seed=seed):
                items.append((sg_by[qid], t.question))
            if len(items) >= n:
                break
        items = items[:n]

    # greedy: does injecting concepts change the continuation vs the frozen model alone?
    concept_out = trainer.generate_student_batch(items, max_new=32, system=TEACHER_SYSTEM)
    base_out = trainer._generate_text_batch([(TEACHER_SYSTEM, q) for _, q in items], 32)
    agree = [c.strip() == b.strip() for c, b in zip(concept_out, base_out, strict=True)]

    # next-token KL(concept || base) at the prompt end — how far concepts perturb the distribution.
    @torch.no_grad()
    def last_logits(with_concepts: bool) -> torch.Tensor:
        feats = [featurize_subgraph(sg, trainer.bb.embed_labels) for sg, _ in items]
        concepts = trainer._encode_concepts(feats) if with_concepts else None
        seqs = []
        for i, (sg, q) in enumerate(items):
            label = sg.center.label or sg.center.qid
            head, tail = trainer._student_split(TEACHER_SYSTEM, q, label)
            parts = [trainer._embed(head)]
            if concepts is not None:
                parts.append(concepts[i])
            parts.append(trainer._embed(tail))
            seqs.append(torch.cat(parts, dim=0))
        emb, attn = trainer._left_pad_embeds(seqs)  # right-aligned → last col is the prompt end
        hid = trainer.bb.forward_hidden(emb, attn, build_position_ids(attn))
        return trainer.bb.lm_head(hid[:, -1, :]).float()

    lc, lb = last_logits(True), last_logits(False)
    # KL(base || concept) per row — how far concepts pull the next-token dist from the frozen model.
    kl = torch.nn.functional.kl_div(
        torch.log_softmax(lc, -1), torch.log_softmax(lb, -1),
        reduction="none", log_target=True,
    ).sum(-1)

    summary = capability_summary(agree, kl.tolist())
    report = {
        "checkpoint": checkpoint, "probe": "capability_preservation", "dataset": dataset,
        "snapshot": snapshot, "seed": seed, "config": blob["config"], **summary,
    }
    out_dir = settings.data_root / "analysis" / "probes" / f"{checkpoint}__capability"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "items.jsonl").open("w", encoding="utf-8") as fh:
        for (sg, q), a, x, c, b in zip(
            items, agree, kl.tolist(), concept_out, base_out, strict=True
        ):
            fh.write(json.dumps(
                {"subject": sg.center.qid, "question": q, "agree": a, "kl": round(x, 6),
                 "concept_out": c, "base_out": b}
            ) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2))

    ga = summary["greedy_agreement"]
    rprint(
        f"[bold]capability preservation[/] {checkpoint} k={trainer.cfg.k}; {len(items)} controls"
    )
    rprint(f"  greedy-agreement (concept == base output): [bold]{ga['acc']:.1%}[/]")
    rprint(f"  next-token KL(base-concept): mean [bold]{summary['kl']['mean']:.4f}[/] "
           f"median {summary['kl']['median']:.4f} max {summary['kl']['max']:.3f}")
    rprint("[dim]high agreement + low KL = concept tokens inert on non-fact tasks = preserved.[/]")
    rprint(f"[green]wrote[/] {out_dir}/summary.json")


@app.command("cf-rag-budget-curve")
def cf_rag_budget_curve(
    dataset: Annotated[str, typer.Option()] = "cftrain_qa_100k",
    snapshot: Annotated[str, typer.Option()] = "cftrain_100k",
    popqa_snapshot: Annotated[str, typer.Option()] = "popqa_full",
    budgets: Annotated[str, typer.Option(help="fact-token budgets; 0 = base/no-facts")] =
    "0,8,16,32,64,128,2048",
    retrieval: Annotated[
        str, typer.Option(help="fact selection: pagerank | question | summary")
    ] = "pagerank",
    eval_n: Annotated[int, typer.Option()] = 300,
    popqa_n: Annotated[int, typer.Option()] = 300,
    val_frac: Annotated[float, typer.Option()] = 0.3,
    split_seed: Annotated[int, typer.Option(help="split to eval against (match eval-final)")] = 0,
    split_mode: Annotated[str, typer.Option(help="fact | question (match the checkpoint)")]
    = "question",
    model: Annotated[str, typer.Option()] = "Qwen/Qwen3-0.6B",
    device: Annotated[str, typer.Option()] = "cuda",
    out: Annotated[str, typer.Option(help="write JSON here")] = "",
) -> None:
    """Text-RAG accuracy vs its INPUT-TOKEN budget — the baseline half of the token-efficiency plot.

    Three retrieval modes so the figure cannot be attacked as a strawman at small budgets:
    - ``pagerank``: query-INDEPENDENT top-PageRank truncation — the apples-to-apples competitor
      for query-independent concept tokens (no answer guarantee; small budgets can drop the fact).
    - ``question``: query-AWARE — facts ranked by embedding similarity to the question (the frozen
      LLM's own label embeddings), so a tiny budget can hold the one relevant fact. The strongest
      per-question text baseline.
    - ``summary``: query-independent LLM-written compression — the frozen LLM summarizes the full
      neighborhood into <= budget tokens (enforced by max_new_tokens), cached per (entity, budget).

    Held-out uses the STRICT (fact-leakage-free) val side of the given split and the frozen
    fixed-seed subset, so items align with eval-final's per-item dumps for paired comparisons.
    budget 0 = base (no facts).
    """
    import json

    from conceptformer.data.benchmarks import load_popqa
    from conceptformer.data.snapshot import iter_subgraphs
    from conceptformer.eval.evalsets import popqa_eval_items, sample_rows
    from conceptformer.eval.retrieval import rank_edges_by_question, summary_prompt
    from conceptformer.generate.dataset import load_cftrain_qa
    from conceptformer.generate.signal import answer_ok
    from conceptformer.generate.teacher import cftrain_prompt
    from conceptformer.model.backbone import Backbone
    from conceptformer.model.chat import ChatModel
    from conceptformer.train.harness import (
        is_answerable,
        split_by_held_out_facts,
        split_by_held_out_questions,
        strict_val_subset,
    )
    from conceptformer.verbalize import verbalize, verbalize_budgeted

    if retrieval not in ("pagerank", "question", "summary"):
        raise typer.BadParameter(f"unknown retrieval mode {retrieval!r}")
    from conceptformer.cache import KVCache

    chat = ChatModel(model, device=device, cache=KVCache(settings.generation_cache_path))
    backbone = Backbone(chat) if retrieval == "question" else None
    budget_list = [int(b) for b in budgets.split(",")]

    # held-out: (subgraph, question, accepted_answers) from the strict val side of the split.
    rows = load_cftrain_qa(settings.data_root / "cf_train" / dataset / "qa_distill.jsonl")
    sg_by = {sg.center.qid: sg for sg in iter_subgraphs(settings.snapshots_dir / snapshot)}
    splitter = split_by_held_out_facts if split_mode == "fact" else split_by_held_out_questions
    train_rows, val = splitter(rows, val_frac=val_frac, seed=split_seed)
    answerable = [
        r for r in val if is_answerable(r) and r.subject_qid in sg_by and r.accepted_answers
    ]
    ho_rows = sample_rows(strict_val_subset(train_rows, answerable), eval_n)
    ho = [(sg_by[r.subject_qid], r.question, r.accepted_answers) for r in ho_rows]

    # PopQA (unseen entities) — same frozen subset eval-final scores.
    pq_sg = {sg.center.qid: sg for sg in iter_subgraphs(settings.snapshots_dir / popqa_snapshot)}
    pq = [(sg, q, a) for sg, q, a, _ in popqa_eval_items(load_popqa(), pq_sg, n=popqa_n)]

    def summaries_for(items: list[tuple], budget: int) -> dict[str, str]:
        """One <= budget-token summary per unique entity (cached greedy generation)."""
        uniq: dict[str, tuple[str, str]] = {}
        for sg, _, _ in items:
            if sg.center.qid not in uniq:
                label = sg.center.label or sg.center.qid
                uniq[sg.center.qid] = summary_prompt(verbalize(sg), label, budget)
        qids = list(uniq)
        texts = chat.generate_batch(
            [uniq[q] for q in qids], max_new_tokens=budget, batch_size=64
        )
        return dict(zip(qids, texts, strict=True))

    def facts_at(items: list[tuple], budget: int) -> list[str]:
        if budget == 0:
            return ["" for _ in items]
        if retrieval == "summary":
            by_qid = summaries_for(items, budget)
            return [by_qid[sg.center.qid] for sg, _, _ in items]
        if retrieval == "question":
            if backbone is None:  # narrowed for ty; construction above guarantees it
                raise RuntimeError("question retrieval requires the backbone embedder")
            return [
                verbalize_budgeted(
                    rank_edges_by_question(sg, q, backbone.embed_labels),
                    chat.count_tokens, budget,
                )
                for sg, q, _ in items
            ]
        return [verbalize_budgeted(sg, chat.count_tokens, budget) for sg, _, _ in items]

    def curve(items: list[tuple], tag: str) -> list[dict]:
        rprint(f"[bold]RAG-budget — {tag} — retrieval={retrieval}[/] (n={len(items)}):")
        rprint(f"  {'budget':>7} {'acc':>7} {'med_tokens':>11}")
        rows_out = []
        for b in budget_list:
            facts = facts_at(items, b)
            prompts = [cftrain_prompt(q, f or None)
                       for (_, q, _), f in zip(items, facts, strict=True)]
            gen = chat.generate_batch_ids(prompts, max_new_tokens=32, batch_size=64)
            preds = [chat.decode(g) for g in gen]
            correct = [answer_ok(p, ans) for p, (_, _, ans) in zip(preds, items, strict=True)]
            acc = sum(correct) / len(items)
            med = sorted(chat.count_tokens(f) for f in facts)[len(facts) // 2]
            rprint(f"  {b:>7} {acc:>7.1%} {med:>11}")
            rows_out.append({
                "budget": b, "acc": round(acc, 4), "median_knowledge_tokens": med,
                "correct": [int(c) for c in correct],  # per-item, paired vs eval-final dumps
            })
        return rows_out

    report = {
        "held_out": curve(ho, "held-out"), "popqa": curve(pq, "PopQA (unseen)"),
        "model": model, "dataset": dataset, "retrieval": retrieval,
        "split": {"split_mode": split_mode, "split_seed": split_seed, "val_frac": val_frac},
        "held_out_items": [
            {"subject_qid": sg.center.qid, "question": q} for sg, q, _ in ho
        ],
        "popqa_items": [{"subject_qid": sg.center.qid, "question": q} for sg, q, _ in pq],
    }
    if out:
        Path(out).write_text(json.dumps(report, indent=2))
        rprint(f"[green]wrote[/] {out}")


@app.command("cf-sweep")
def cf_sweep(
    params: Annotated[str, typer.Option(help="spec: 'd-model=512,768,1024;n-layers=2,3,4'")],
    dataset: Annotated[str, typer.Option()] = "cftrain_qa_1k",
    snapshot: Annotated[str, typer.Option()] = "cftrain_1k",
    steps: Annotated[int, typer.Option()] = 16000,
    batch: Annotated[int, typer.Option()] = 32,
    eval_n: Annotated[int, typer.Option()] = 96,
    eval_every: Annotated[int, typer.Option(help="0 = quarter of steps (for curves)")] = 0,
    popqa_eval: Annotated[int, typer.Option(help="PopQA examples per run (0=skip)")] = 0,
    augment: Annotated[bool, typer.Option()] = False,
    subsample: Annotated[bool, typer.Option(help="re-sample teacher distractors each step")] = True,
    cache_teacher: Annotated[
        bool, typer.Option(help="cache teacher hidden upfront; --no for large-data/few-epoch")
    ] = True,
    method: Annotated[str, typer.Option(help="grid / random / bayes")] = "bayes",
    count: Annotated[int, typer.Option(help="max trials per agent (0=until stopped)")] = 0,
    devices: Annotated[str, typer.Option(help="comma-separated GPUs")] = "cuda:0,cuda:1",
    name: Annotated[str, typer.Option(help="sweep name")] = "sweep",
    project: Annotated[str, typer.Option()] = "conceptformer-v2",
    entity: Annotated[str, typer.Option()] = "university-of-zurich",
) -> None:
    """Create a W&B sweep over one or more cf-train hyperparameters; one agent per GPU.

    ``params`` is ``;``-separated ``name=v1,v2,...`` groups (cf-train flag name, hyphenated), e.g.
    ``--params "d-model=512,768,1024;n-layers=2,3,4"`` for a capacity grid.
    """
    import os
    import subprocess

    import wandb

    def _coerce(v: str) -> object:
        for cast in (int, float):
            try:
                return cast(v)
            except ValueError:
                pass
        return v

    parameters = {}
    for group in params.split(";"):
        name_, _, vals = group.partition("=")
        parameters[name_.strip()] = {"values": [_coerce(v) for v in vals.split(",")]}

    fixed = [
        "--dataset", dataset, "--snapshot", snapshot,
        "--steps", str(steps), "--batch", str(batch), "--eval-n", str(eval_n),
        "--popqa-eval", str(popqa_eval), "--eval-every", str(eval_every or max(1, steps // 4)),
        "--wandb", "--device", "cuda:0",
    ]
    if augment:
        fixed.append("--augment")
    if subsample:
        fixed.append("--subsample")
    if not cache_teacher:
        fixed.append("--no-cache-teacher")
    sweep_config = {
        "name": name,
        "method": method,
        "metric": {"name": "held_out/concept_acc", "goal": "maximize"},
        "parameters": parameters,
        "command": ["${env}", "python", "-m", "conceptformer.cli", "cf-train", *fixed, "${args}"],
    }
    sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)
    url = f"https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}"
    rprint(f"[green]created sweep[/] {url}")

    log_dir = settings.data_root / "sweeps" / name
    log_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    for dev in devices.split(","):
        gpu = dev.rsplit(":", 1)[-1]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
        cmd = ["wandb", "agent", *(["--count", str(count)] if count else [])]
        cmd.append(f"{entity}/{project}/{sweep_id}")
        log = (log_dir / f"agent_gpu{gpu}.log").open("w")
        procs.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT))
        rprint(f"  launched agent on GPU {gpu}")
    rprint(f"[bold]running {len(procs)} agents[/] — live at {url}")
    for p in procs:
        p.wait()
    rprint(f"[green]sweep done[/] {url}")


if __name__ == "__main__":
    app()
