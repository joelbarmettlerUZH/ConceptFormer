"""ConceptFormer v2 CLI (typer)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich import print as rprint
from rich.table import Table

from conceptformer.config import settings
from conceptformer.data.benchmarks import load_popqa, subject_qids
from conceptformer.data.coverage import audit_coverage
from conceptformer.data.snapshot import build_snapshot_parallel
from conceptformer.data.wikidata import WikidataClient

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
        teacher_prompt,
    )
    from conceptformer.model.chat import ChatModel
    from conceptformer.verbalize import verbalize_with_answer

    qa_dir = settings.data_root / "cf_train" / dataset
    rows = load_cftrain_qa(qa_dir / "qa_tiered.jsonl")
    sg_by_qid = {sg.center.qid: sg for sg in iter_subgraphs(settings.snapshots_dir / snapshot)}

    chat = ChatModel(model, device=device, cache=KVCache(settings.generation_cache_path))
    budget = settings.rag_context_tokens
    prompts = []
    for r in rows:
        sg = sg_by_qid.get(r.subject_qid)
        # Guarantee the answer's edge is in the teacher's facts (large neighborhoods can cut it).
        facts = verbalize_with_answer(sg, r.answer_qid, chat.count_tokens, budget) if sg else ""
        prompts.append(teacher_prompt(r.question, facts))

    ids = chat.generate_batch_ids(prompts, max_new_tokens=max_new_tokens, batch_size=batch_size)
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

    client = GemmaClient(base_url=base_url, n_questions=n_questions)
    results = asyncio.run(client.generate(sgs))
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


@app.command("cf-train")
def cf_train(
    dataset: Annotated[str, typer.Option(help="distilled CF-Train dataset")] = "cftrain_qa_smoke",
    snapshot: Annotated[str, typer.Option(help="matching snapshot")] = "cftrain_smoke",
    model: Annotated[str, typer.Option(help="frozen backbone")] = "Qwen/Qwen3-0.6B",
    k: Annotated[int, typer.Option(help="concept tokens")] = 8,
    d_model: Annotated[int, typer.Option(help="encoder width")] = 512,
    n_layers: Annotated[int, typer.Option(help="resampler layers")] = 2,
    lr: Annotated[float, typer.Option(help="encoder learning rate")] = 1e-4,
    temperature: Annotated[float, typer.Option(help="KL distillation temperature")] = 1.0,
    steps: Annotated[int, typer.Option()] = 600,
    batch: Annotated[int, typer.Option(help="minibatch size")] = 8,
    val_frac: Annotated[float, typer.Option(help="held-out question fraction")] = 0.3,
    eval_every: Annotated[int, typer.Option()] = 100,
    eval_n: Annotated[int, typer.Option(help="held-out examples scored per eval (capped)")] = 120,
    popqa_eval: Annotated[int, typer.Option(help="after training, score N unseen PopQA")] = 0,
    popqa_snapshot: Annotated[str, typer.Option(help="snapshot with PopQA neighborhoods")] = "popqa_full",  # noqa: E501
    checkpoint: Annotated[str, typer.Option(help="save trained encoder under this name")] = "",
    augment: Annotated[bool, typer.Option(help="distill under many system prompts")] = False,
    subsample: Annotated[bool, typer.Option(help="re-sample teacher distractors/step")] = False,
    placement: Annotated[
        str,
        typer.Option(help="concept slot: prefix|before_entity|after_entity|replace_entity"),
    ] = "prefix",
    grad_clip: Annotated[float, typer.Option(help="max grad-norm (0=off)")] = 0.0,
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
    from conceptformer.generate.dataset import load_cftrain_qa
    from conceptformer.model.backbone import Backbone
    from conceptformer.model.chat import ChatModel
    from conceptformer.train.harness import split_by_held_out_questions
    from conceptformer.train.trainer import (
        AUGMENT_SYSTEMS,
        HELD_OUT_EVAL_SYSTEM,
        TEACHER_SYSTEM,
        ConceptTrainer,
        TrainConfig,
    )

    qa_dir = settings.data_root / "cf_train" / dataset
    rows = load_cftrain_qa(qa_dir / "qa_distill.jsonl")
    sg_by_qid = {sg.center.qid: sg for sg in iter_subgraphs(settings.snapshots_dir / snapshot)}
    train_rows, val_rows = split_by_held_out_questions(rows, val_frac=val_frac, seed=seed)
    train_tuples = [
        (sg_by_qid[r.subject_qid], r.question, r.teacher_target_ids, r.answer_qid)
        for r in train_rows
        if r.subject_qid in sg_by_qid and r.teacher_target_ids
    ]
    val_rows = [r for r in val_rows if r.subject_qid in sg_by_qid]
    rng = random.Random(seed)
    # Fixed eval subset so the curve is comparable across reports (full val would be too slow).
    eval_val = rng.sample(val_rows, min(eval_n, len(val_rows)))
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
        temperature=temperature,
        warmup_steps=max(10, steps // 20),
        total_steps=steps,
        rag_context_tokens=1024,
        augment_systems=AUGMENT_SYSTEMS if augment else (),
        subsample_neighbors=subsample,
        placement=placement,
        grad_clip=grad_clip,
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
                "temperature": temperature, "steps": steps, "batch": batch,
                "augment": augment, "subsample": subsample, "placement": placement,
                "grad_clip": grad_clip, "ema_decay": ema_decay, "gate_mode": gate_mode,
                "seed": seed,
                "dataset": dataset, "snapshot": snapshot,
                "trainable_params": sum(p.numel() for p in trainer.model.parameters()),
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
        trainer.build_eval(train_sample, sg_by_qid, eval_system=eval_system)
        if train_sample
        else None
    )

    popqa_items: list = []
    if popqa_eval:
        from conceptformer.data.benchmarks import load_popqa

        popqa_sgs = {
            sg.center.qid: sg for sg in iter_subgraphs(settings.snapshots_dir / popqa_snapshot)
        }
        examples = [e for e in load_popqa() if e.subject_qid in popqa_sgs and e.answer_labels]
        rng.shuffle(examples)
        for e in examples[: popqa_eval * 2]:  # over-sample; some neighborhoods may be empty
            sg = popqa_sgs[e.subject_qid]
            if sg.edges:
                popqa_items.append((sg, e.question, e.answer_labels, e.answer_qid))
            if len(popqa_items) >= popqa_eval:
                break

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
            log["held_in/val_kl"] = mi["val_kl"]
            # gen-gap > 0 => fits trained questions better than held-out (overfitting signal).
            log["gen_gap/concept_acc"] = mi["concept_acc"] - m["concept_acc"]
            line += f"  | held_in={mi['concept_acc']:.1%}"
        if popqa_items:
            pm = trainer.evaluate_popqa(popqa_items, eval_system=eval_system, cache_brackets=True)
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
            log["train/lr"] = trainer.opt.param_groups[0]["lr"]
            wb.log(log, step=step)
        return m

    if subsample:
        rprint("[dim]subsample mode: features cached, teacher facts re-sampled each step…[/dim]")
        train_pool: list = train_tuples  # raw 4-tuples; teacher rebuilt live per step
        step_fn = trainer.step
    else:
        rprint("[dim]preprocessing (featurize + tokenize once)…[/dim]")
        train_pool = trainer.prepare(train_tuples)  # hoists CPU work out of the training loop
        step_fn = trainer.step_prepared
    trainer.setup_eval(eval_val, sg_by_qid, eval_system=eval_system)  # brackets computed once
    report("init", 0)
    for s in range(1, steps + 1):
        loss = step_fn(rng.sample(train_pool, min(batch, len(train_pool))))
        if wb and s % 25 == 0:
            wb.log({"train/loss": loss, "train/lr": trainer.opt.param_groups[0]["lr"]}, step=s)
        if s % eval_every == 0 or s == steps:
            report(f"step {s}", s)

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
        if hi_final is not None:
            wb.summary["held_in/concept_acc"] = hi_final
            wb.summary["held_in/val_kl"] = last_metrics.get("held_in_val_kl")
        if popqa_items:
            for kk in ("concept_acc", "base_acc", "teacher_acc"):
                wb.summary[f"popqa/{kk}"] = last_metrics.get(f"popqa_{kk}")

    if checkpoint:
        ckpt_path = settings.data_root / "checkpoints" / f"{checkpoint}.pt"
        trainer.save_checkpoint(ckpt_path)
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
            wb.log_artifact(art)
            rprint(f"[green]logged W&B artifact[/] model:{checkpoint}")

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
    import random
    import statistics

    import torch

    from conceptformer.data.benchmarks import load_popqa
    from conceptformer.data.snapshot import iter_subgraphs
    from conceptformer.generate.dataset import load_cftrain_qa
    from conceptformer.model.backbone import Backbone
    from conceptformer.model.chat import ChatModel
    from conceptformer.train.harness import split_by_held_out_questions
    from conceptformer.train.trainer import (
        AUGMENT_SYSTEMS,
        HELD_OUT_EVAL_SYSTEM,
        TEACHER_SYSTEM,
        ConceptTrainer,
        TrainConfig,
    )

    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        ckpt_path = settings.data_root / "checkpoints" / f"{checkpoint}.pt"
    if not ckpt_path.exists():
        # Fall back to the W&B model artifact so a result is reproducible without the local file.
        import wandb as _wandb

        rprint(f"[dim]checkpoint not local; pulling W&B artifact model:{checkpoint}:latest…[/dim]")
        api = _wandb.Api()
        art = api.artifact(
            f"university-of-zurich/conceptformer-v2/{checkpoint}:latest", type="model"
        )
        ckpt_path = Path(art.download()) / f"{checkpoint}.pt"
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = TrainConfig(**blob["config"])
    backbone = Backbone(ChatModel(model, device=device))
    trainer = ConceptTrainer(backbone, cfg)
    trainer.model.load_state_dict(blob["model"])
    trainer.model.eval()
    rprint(
        f"[bold]loaded[/] {ckpt_path.name}  "
        f"(k={cfg.k}, d_model={cfg.d_model}, n_layers={cfg.n_layers})"
    )

    qa_dir = settings.data_root / "cf_train" / dataset
    rows = load_cftrain_qa(qa_dir / "qa_distill.jsonl")
    sg_by_qid = {sg.center.qid: sg for sg in iter_subgraphs(settings.snapshots_dir / snapshot)}
    _, val_rows = split_by_held_out_questions(rows, val_frac=val_frac, seed=seed)
    val_rows = [r for r in val_rows if r.subject_qid in sg_by_qid]
    rng = random.Random(seed)
    eval_val = rng.sample(val_rows, min(eval_n, len(val_rows)))

    popqa_sgs = {
        sg.center.qid: sg for sg in iter_subgraphs(settings.snapshots_dir / popqa_snapshot)
    }
    examples = [e for e in load_popqa() if e.subject_qid in popqa_sgs and e.answer_labels]
    rng.shuffle(examples)
    popqa_items = []
    for e in examples[: popqa_n * 2]:
        sg = popqa_sgs[e.subject_qid]
        if sg.edges:
            popqa_items.append((sg, e.question, e.answer_labels, e.answer_qid))
        if len(popqa_items) >= popqa_n:
            break

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
