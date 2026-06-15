"""ConceptFormer v2 CLI (typer)."""

from __future__ import annotations

import json
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
    seed: Annotated[int, typer.Option(help="Sampling seed")] = 0,
) -> None:
    """Select a tail-heavy CF-Train entity pool from the danker PageRank file (eval-excluded)."""
    from conceptformer.data.select import (
        download_pagerank,
        load_pagerank,
        manifest,
        save_entities,
        select_entities,
    )

    path = download_pagerank()
    entries = load_pagerank(path)
    exclude = subject_qids(iter(load_popqa()))  # never train on eval subjects
    selected = select_entities(entries, n=n, exclude=exclude, seed=seed)

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
    from conceptformer.verbalize import verbalize_budgeted

    qa_dir = settings.data_root / "cf_train" / dataset
    rows = load_cftrain_qa(qa_dir / "qa_tiered.jsonl")
    sg_by_qid = {sg.center.qid: sg for sg in iter_subgraphs(settings.snapshots_dir / snapshot)}

    chat = ChatModel(model, device=device, cache=KVCache(settings.generation_cache_path))
    budget = settings.rag_context_tokens
    prompts = []
    for r in rows:
        sg = sg_by_qid.get(r.subject_qid)
        facts = verbalize_budgeted(sg, chat.count_tokens, budget) if sg else ""
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
    from conceptformer.verbalize import verbalize_budgeted

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
        facts = verbalize_budgeted(sg, chat.count_tokens, budget) if sg else ""
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
        (sg_by_qid[r.subject_qid], r.question, r.teacher_target_ids or [])
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
    steps: Annotated[int, typer.Option()] = 600,
    batch: Annotated[int, typer.Option(help="minibatch size")] = 8,
    val_frac: Annotated[float, typer.Option(help="held-out question fraction")] = 0.3,
    eval_every: Annotated[int, typer.Option()] = 100,
    eval_n: Annotated[int, typer.Option(help="held-out examples scored per eval (capped)")] = 120,
    popqa_eval: Annotated[int, typer.Option(help="after training, score N unseen PopQA")] = 0,
    popqa_snapshot: Annotated[str, typer.Option(help="snapshot with PopQA neighborhoods")] = "popqa_full",  # noqa: E501
    checkpoint: Annotated[str, typer.Option(help="save trained encoder under this name")] = "",
    augment: Annotated[bool, typer.Option(help="distill under many system prompts")] = False,
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
        (sg_by_qid[r.subject_qid], r.question, r.teacher_target_ids)
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
        warmup_steps=max(10, steps // 20),
        total_steps=steps,
        rag_context_tokens=1024,
        augment_systems=AUGMENT_SYSTEMS if augment else (),
    )
    trainer = ConceptTrainer(backbone, cfg)
    # When augmenting, evaluate under a HELD-OUT prompt (decoupling test); else the training prompt.
    eval_system = HELD_OUT_EVAL_SYSTEM if augment else TEACHER_SYSTEM
    if augment:
        rprint(f"[cyan]prompt augmentation ON[/] ({len(AUGMENT_SYSTEMS)} systems); "
               f"eval under HELD-OUT prompt: {eval_system!r}")

    def report(tag: str) -> None:
        m = trainer.evaluate()
        rprint(
            f"  [{tag}] val_KL={m['val_kl']:.3f}  "
            f"[bold]concept_acc={m['concept_acc']:.1%}[/]  "
            f"base={m['base_acc']:.1%}  teacher(RAG)={m['teacher_acc']:.1%}  (n={m['n_acc']})"
        )

    rprint("[dim]preprocessing (featurize + tokenize once) + static eval brackets…[/dim]")
    prepared = trainer.prepare(train_tuples)  # hoists CPU work out of the training loop
    trainer.setup_eval(eval_val, sg_by_qid, eval_system=eval_system)  # brackets computed once
    report("init")
    for s in range(1, steps + 1):
        trainer.step_prepared(rng.sample(prepared, min(batch, len(prepared))))
        if s % eval_every == 0 or s == steps:
            report(f"step {s}")

    # Held-IN accuracy (a sample of TRAINED questions) disambiguates overfitting from underfitting.
    from conceptformer.train.harness import is_answerable

    held_in = [r for r in train_rows if is_answerable(r) and r.subject_qid in sg_by_qid]
    train_sample = rng.sample(held_in, min(len(eval_val), len(held_in)))
    trainer.setup_eval(train_sample, sg_by_qid, eval_system=eval_system)  # held-in sample
    tm = trainer.evaluate()
    rprint(
        f"  [held-IN sample] concept_acc={tm['concept_acc']:.1%}  KL={tm['val_kl']:.3f}  "
        f"(n={tm['n_acc']}) — high held-in + low held-out = overfit; both low = undertrained"
    )

    if checkpoint:
        ckpt_path = settings.data_root / "checkpoints" / f"{checkpoint}.pt"
        trainer.save_checkpoint(ckpt_path)
        rprint(f"[green]saved checkpoint[/] → {ckpt_path}")

    if popqa_eval:
        from conceptformer.data.benchmarks import load_popqa

        popqa_sgs = {
            sg.center.qid: sg
            for sg in iter_subgraphs(settings.snapshots_dir / popqa_snapshot)
        }
        examples = [e for e in load_popqa() if e.subject_qid in popqa_sgs and e.answer_labels]
        rng.shuffle(examples)
        items = []
        for e in examples[: popqa_eval * 2]:  # over-sample; some neighborhoods may be empty
            sg = popqa_sgs[e.subject_qid]
            if sg.edges:
                items.append((sg, e.question, e.answer_labels))
            if len(items) >= popqa_eval:
                break
        rprint(f"[bold]PopQA (UNSEEN entities, n={len(items)})[/] — external generalization:")
        pm = trainer.evaluate_popqa(items, eval_system=eval_system)
        rprint(
            f"  [bold]concept_acc={pm['concept_acc']:.1%}[/]  "
            f"base={pm['base_acc']:.1%}  teacher(RAG)={pm['teacher_acc']:.1%}"
        )

    rprint("[green]done[/] — concept_acc on HELD-OUT questions is the generalization signal.")


if __name__ == "__main__":
    app()
