# Pre-submission adversarial review (10 reviewers, code + data/analysis + W&B)

Date: 2026-07-28. Ten independent reviewers, one per claim/subsystem, each instructed to
falsify by re-deriving numbers from source (not just read). Deduplicated below. Severity:
BLOCKER (factual error / internal contradiction / figure bug) > MAJOR (framing or rigor a
reviewer will attack) > MINOR > TRIVIA. Tag: [reword] fixable in text, [figure] needs figure
regen, [rerun] needs an experiment, [decide] needs an author call.

## What survived falsification (the core is sound)
- **Headline PopQA result is clean.** PopQA subjects fully excluded from both training corpora
  (0 QID overlap); coverage 14,266/14,267; base/RAG/concept scored on identical items; alias
  metric symmetric and honest ("within 0.5pt" exact at 0.48%). Checkpoint selection is on
  held-out, not PopQA test, so no test-set selection bias on the headline.
- **No fabrication, no data leakage.** Strict held-out has 0 (entity,fact) leakage; the 32.8%
  paraphrase-leak fix is genuine. All six main tables reproduce cell-for-cell from data/analysis.
- **KL objective is correct.** Full-vocab, correctly position-aligned (bit-exact), teacher==student
  frozen model, cache_teacher identical to live, RoPE span contiguous. The adaptation KL is the
  genuine cf-train objective, not a re-implementation.
- **Cross-lingual translation confound REFUTED.** German labels are authentic Wikidata surface
  forms (not Gemma); RAG on the German snapshot = 0.897, so labels are usable as text and only the
  encoder cannot read German label embeddings. System-prompt ablation holds.
- **Qwen 1.7B valley is real** on accuracy/margin level (no seed-range overlap; 4B worst seed >
  1.7B best seed). Gap-closure 44/28/16% recomputes correctly.
- **OOM incident poisoned nothing.** 17 crashed/failed ghosts in the 1.7B group correctly
  excluded; every survivor is cache_teacher=False, _step=100000, distinct seed. Variance is real
  (~10x binomial noise), not eval noise.

## BLOCKERS (fix before submission)

### B1. Internal contradiction: "0.210 -> 0.477" and "ratio 2.05" cannot both hold [decide/figure]
0.477/0.210 = 2.27, not 2.05. Root cause: tab:grid + "Data axis" prose use the **before_entity**
6-seed 10k k=8 cell (.210); fig_kcurves + the "2.05" substitution ratio use the **prefix** 3-seed
cell (.232 -> 2.06). Same nominal cell, two placements, adjacent paragraphs. (R4 HIGH, R9, R10.)
main.tex:373,380,403; scripts/cf2_figures.py:41-48. Fix: pick ONE 10k k=8 model throughout §5.2.

### B2. fig_kcurves caption "identical protocol" is false [figure]
The 10k curve is placement=prefix (p32/p25/v15), the 100k curve is before_entity (pc_k*), plus
different step budgets (72k vs 100k) and caching. The annotated ratios (2.05@k8, 1.48@k32) mix a
data-scaling effect with a placement change. main.tex:360-362. (R4, R9, R10.)

### B3. fig_transfer PNG title says "~32%" but the data is 20-23% [figure]
scripts/cf2_figures.py:258-259 hard-codes the title "two foreign graphs converge to ~32% gap
closure at k=32". Actual 3-seed closures: MetaQA 20.3%, WorldCup 22.9% (home 50.7%). The LaTeX
caption is now correct; the rendered PNG title contradicts it. Fix the title string + regen. (R10.)

### B4. Probes "drops accuracy to near zero" contradicts the figure [reword]
main.tex:473-474. Residual accuracy after ablating the questioned edge is 56.3% (k1) / 34.9% (k8)
/ 29.8% (k32), plotted mid-range in fig_probes. "Near zero" is unsupported. (This phrasing was
introduced in the 2026-07-28 prose pass; the prior "collapses" was also loose.) Fix: state the
residuals; the answer-vs-other-edge asymmetry (30% vs 94%) is the real, defensible result.
(R1 HIGH, R10.)

## MAJOR (a reviewer will attack; reword/re-scope, some optionally rerun)

### M1. "no annotated QA anywhere in the pipeline" overstates label-free [reword]
main.tex:97. The gold answer edge drives verbalize_with_answer (answer guaranteed present) AND the
tiering filter, so the answer signal enters upstream even though it never enters the loss. Keep the
accurate line 265 ("no QA labels enter the loss"); soften line 97. Same issue in §5.4 adaptation
("needs no labels", main.tex:594-602). (R2 MEDIUM, R7.)

### M2. The "RAG" ceiling is a near-oracle, not a budgeted retriever [reword/decide]
RAG uses verbalize_with_answer for eval-final (answer guaranteed) giving PopQA RAG 0.960; the
no-guarantee RAG is 0.937. For transfer, "without an answer guarantee" is vacuous: 1-hop
neighborhoods fit the 1024-tok budget >99% of the time, so RAG renders the full neighborhood. The
guarantee does NOT leak into the text baselines (clean). Relabel as "full-neighborhood text upper
bound" / "oracle-retrieval ceiling"; note Gemma-270m ceiling is 0.825, so "RAG ceiling ~0.97"
(main.tex:421) is Qwen-only. (R3, R5, R6.)

### M3. Transfer "disjoint relation vocabularies" is false for MetaQA [reword/decide]
"cast member" and "genre" are literally Wikidata property labels present in training; the
hand-written reverse-label map (data/metaqa.py) is an undisclosed human-tuned knob. WorldCup is
genuinely clean. Fix: drop/limit "disjoint vocabularies" to WorldCup, disclose the label map,
ideally show robustness to a naive underscore->space mapping [rerun-optional]. main.tex:538-541. (R5 HIGH.)

### M4. Transfer "share nothing with Wikidata / genuinely unseen entities" overstated [reword]
~29% (MetaQA) and ~34% (WorldCup) of entity labels appear verbatim in the training corpus (the
famous ones). Zero-shot on the *task* survives (no QA-pair contamination); "share nothing" does
not. Soften to "disjoint QA supervision". main.tex:538-539. (R5.)

### M5. Efficiency "64 to 128 fact tokens ... five to six times" is self-contradictory [reword/figure]
64-128 / 8 = 8-16x, not 5-6x. The 5-6x reconciles only with *median* tokens (~36-49); no baseline
needs 128. Restate to one accounting. Also fig:efficiency overlays concept (full PopQA, base .103)
against text baselines (n=300 subset, base .073) while the code claims a shared set; honest
same-items head-to-head is 0.466 vs 0.147 (~1pt lower, still favorable). main.tex:327-329. (R3, R10.)

### M6. Probes "injection leaves the model undisturbed elsewhere" cherry-picks the median [reword/decide]
Median control KL 0.05-0.08 is real, but mean is 0.18-0.22, max 3.7-5.6 nats, and greedy
agreement is only ~30% (i.e. injection changes ~2/3 of off-topic continuations). Also two "control"
tasks are name-surface tasks. Report greedy agreement alongside the median, or soften. main.tex:484-487. (R1 HIGH.)

### M7. "Every 10k cell reproducible to within +/-0.8 points" falsified by the paper's own table [reword]
Qwen-0.6B 10k = .210+/-.012 (1.2pt), and two same-config campaigns differ 1.25pt in mean. +/-0.8
is a within-triplet spread. main.tex:443-444. (R9.)

### M8. Held-out "grouped by fact" description does not match what was run [reword/rerun]
Runs used a question-level split + post-hoc strict leak removal. Result is leakage-free, but the
strict set is ~97% singleton facts (vs ~71% under true fact-grouping), so held-out is measured on a
non-representative subsample. Fix the description, or re-run held-out on a true fact-grouped split.
main.tex:299-302. (R8 MAJOR.)

### M9. "scored on ... disjoint sets" is inverted [reword]
Selection samples are NESTED prefixes of the scored sets, not disjoint. Negligible numeric impact
(selection on held-out, 0.75% overlap), but stated as a guarantee. Drop "disjoint". main.tex:304. (R8.)

### M10. Cross-lingual section is single-seed throughout [rerun/decide]
No error bars anywhere in §5.6, contrary to the paper's own >=3-seed rule (F21 marked done at n=1).
Headline contrast (40% vs 8%) is safe; sub-cells (e.g. 1.7B k8 13% vs 1%) are not defensible at
that standard. Add seeds or caveat explicitly. (R7.)

### M11. Gemma "both patterns replicate" overstates [reword]
Gemma replicates the small->mid margin *decline* only. The dip+recovery valley is within-noise at
100k and monotone (inverted) at 10k. Soften abstract/intro. main.tex:59-60,106. (R6 MEDIUM-HIGH.)

### M12. "Training stability is the binding constraint" is an interpretation overreach [reword/decide]
The variance is PopQA-generalization-specific (held-out is tight across the same seeds) and is
confounded by a fixed 100k-step budget (the most stable cell, 0.6B, is undertrained). Frame as an
open observation, not the established bottleneck. main.tex:714-716,753. (R9.)

### M13. "5-6x transfer clears base" and "approaching RAG at k=32" oversell [reword]
Transfer: "every budget clears the base by three to four times" is true only for MetaQA k>=4
(MetaQA k1 = 2.8x; all WorldCup ~1.9-2.4x). main.tex:545. Adaptation: "approaching the RAG
reference at k=32" oversells 0.658 vs 0.921 (26pt gap). main.tex:601. (R5, R7.)

## MINOR

- N1 [reword] "bit-identical at step 0" is literally false (KL~3e-4, max logit diff 3.7 from the k
  placeholder tokens); say "the concept block has no effect at init". main.tex:246. (R2.)
- N2 [reword] "had not saturated" unsupported for PopQA (trajectory flat last 10k; only held-out
  climbs); a 2-point curve can't establish saturation. main.tex:374. (R4.)
- N3 [reword/decide] Held-in "0.82 -> 0.64" uses off-protocol pre-M7 n=200 training-time numbers,
  a cherry-picked placement (prefix 0.82 vs before_entity 0.79) and best seed (0.64 vs mean 0.618);
  not in eval_final. Move to strict protocol or downgrade "memorizes less". main.tex:375. (R4, R10.)
- N4 [reword] Data-gain "recovery" leg (1.7B->4B) overlaps at ~1sigma in both families; state
  recovery on level, not data-gain slope. main.tex:419-420. (R6.)
- N5 [reword] tab:kfamily caption official base "0.107" not reproducible (released dumps give
  0.124); likely stale. main.tex:337. (R10.)
- N6 [reword] "1.48 ratio" rests on an outlier-driven 10k k=32 cell (std 4.4pt; drop outlier ->
  1.59). Add a caveat. main.tex:380. (R4.)
- N7 [reword] ">+/-4 points" is a fragile n=3 threshold (4.16/4.37); report seed ranges (8pt)
  instead. main.tex:713,725. (R9.)
- N8 [reword] Adaptation "3 seeds" are 3 pretrained inits, not 3 reseeded pipelines (deterministic
  adaptation; no --seed). Discloses tight std. Note it. (R7.)
- N9 [reword/footnote] Batched loss is a token-weighted mean, not the eq.'s per-example 1/m;
  gathered-head is fp-approximate, not bit-exact. Add a footnote or fix the equation. losses.py:38. (R2.)
- N10 [reword] Base bracket not bit-identical across runs (~7-item greedy drift); soften "identical
  item sets". (R8.)

## RESOLVED BY RERUNS (2026-07-28)
- M3 naive-label transfer: re-ran MetaQA transfer with naive underscore->space labels (snapshot
  metaqa_naive, 3 seeds). Transfer holds: k=8 0.219+/-.031 vs 0.250 hand-written, all k well above
  base 0.058. Not an artifact of the label map. Stated in S5.4.
- M10 cross-lingual single-seed: re-ran the 0.6B EN- and DE-label conditions at s1/s2 (3 seeds).
  tab:multilingual 0.6B rows + fig_multilingual now carry error bars. Headline holds; the
  single-seed k=32 DE-label closure (17%) was a high outlier, 3-seed is 9+/-7% (near floor, cleaner).
  (Caught and corrected a mis-specified EN context in the first rerun; final uses EN-labels-DE-context.)
  1.7B/4B multilingual remain single-seed (only s0 checkpoints exist), noted in the caption.
- B1/B2 placement: before_entity 10k k-curve retrain in progress (bm0ja2zrw), ~2 days.

## TRIVIA
- T1 "14,230 questions" -> 14,229. main.tex:641,650. (R10.)
- T2 1.7B held-out "0.397 -> 0.418" -> 0.414. main.tex:426. (R10.)
- T3 substitution ratio 2.056 rounds to 2.06, paper says 2.05. main.tex:380. (R10.)
- T4 "German entity mention" applies to only ~33% of questions (67% keep the English surface form);
  does not confound. main.tex:636,659. (R7.)
