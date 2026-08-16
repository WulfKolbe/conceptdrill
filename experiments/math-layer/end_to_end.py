#!/usr/bin/env python3
r"""Step 22: the vector-space fixture through every stage, end to end.

    PYTHONPATH=src python3 experiments/math-layer/end_to_end.py

    section -> CES -> MathConcept -> Lean anchors -> Loogle
            -> semantic neighbourhood -> predicted next mathematics

Seven stages. Each is attempted independently and reports `ok`, `degraded` or
`failed` with a reason, and **the failures are printed in their own section at
the end** rather than mixed in with what worked. A run in which four stages
succeed and three fail is a useful result; a run that prints one number and
hides which stages produced it is not.

`degraded` is its own outcome on purpose. A stage that fell back to a fixture
did not succeed, and calling it `ok` is exactly the substitution that voided a
416-document corpus build.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

# See ces_neighbourhood.py: the installed torchvision is broken against the
# installed torch, and transformers touches it resolving ModernBertModel.
sys.modules.setdefault("torchvision", None)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import blasfix                                                    # noqa: E402

blasfix.apply_env_mitigations()

HERE = Path(__file__).resolve().parent
BAR = "=" * 74

SECTION_TEXT = ("V is an n-dimensional vector space over K. "
                "W is an m-dimensional vector space over K. "
                "Let T colon V right arrow W be linear.")
SECTION_LATEX = [r"V", r"W", r"K", r"T : V \to W", r"T \colon V \rightarrow W"]

results: list[dict] = []


def stage(number: int, name: str):
    """Run one stage, catching everything. The decorator records the outcome."""
    def wrap(fn):
        print(f"\n{BAR}\n{number}. {name}\n{BAR}")
        record = {"stage": number, "name": name, "status": "failed",
                  "reason": "", "detail": None}
        try:
            status, reason, detail = fn()
            record.update(status=status, reason=reason, detail=detail)
        except Exception as exc:
            record["reason"] = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED  {record['reason']}")
            traceback.print_exc(limit=2)
        else:
            mark = {"ok": "OK", "degraded": "DEGRADED", "failed": "FAILED"}
            print(f"\n  {mark[status]}  {reason}")
        results.append(record)
        return fn
    return wrap


# --------------------------------------------------------------------------

@stage(1, "SECTION  (text, spoken mathematics, exact LaTeX)")
def _section():
    print(f"  text:  {SECTION_TEXT}")
    for form in SECTION_LATEX:
        print(f"  latex: {form}")
    try:
        from conceptdrill.hierarchy.speech import load_speaker
        speaker = load_speaker()
        said = speaker.speak_math(r"T : V \to W")
        print(f"  SRE:   {r'T : V \to W'!r} -> {said!r}")
        other = speaker.speak_math(r"T \colon V \rightarrow W")
        if said == other:
            print("  note:  \\to and \\rightarrow speak IDENTICALLY, so the "
                  "spoken form cannot recover the spelling")
        return "ok", f"SRE available; spoken form generated", {"spoken": said}
    except Exception as exc:
        return ("degraded", f"SRE unavailable ({type(exc).__name__}); the "
                f"stored spoken form is used", None)


@stage(2, "CES  (project the section into the corpus concept space)")
def _ces():
    import numpy as np
    from conceptdrill.embeddings import get_embedder

    path = HERE / "ces-neighbourhood.json"
    if not path.exists():
        return "failed", "no ces-neighbourhood.json; run ces_neighbourhood.py", None

    full = Path.home() / "conceptdrill-corpus-llm" / "full"
    vectors, labels = [], []
    for chunk in sorted(full.iterdir()):
        if not ((chunk / "basis.npz").exists() and (chunk / "basis.json").exists()):
            continue
        with np.load(chunk / "basis.npz", allow_pickle=True) as z:
            matrix = np.asarray(z["matrix"], dtype=np.float64)
            row_ids = [str(x) for x in z["row_ids"]]
        by_id = {r["row_id"]: r for r in
                 json.loads((chunk / "basis.json").read_text())["rows"]}
        for i, rid in enumerate(row_ids):
            rec = by_id.get(rid)
            if rec and not rec.get("structural"):
                vectors.append(matrix[i])
                labels.append(rec["label"])
    pool = np.vstack(vectors)

    embedder = get_embedder("modernbert", cache=True)
    query = np.asarray(embedder.encode([SECTION_TEXT]), dtype=np.float64)[0]
    query = query / (np.linalg.norm(query) or 1.0)
    sims = pool @ query
    top = np.argsort(-sims)[:5]
    print(f"  pool: {pool.shape[0]} concept rows (16 chunk bases, not merged)")
    for i in top:
        print(f"  {sims[i]:.3f}  {labels[i][:88]}")

    # The stage runs. Whether it HELPS is a separate question, already measured.
    return ("ok", f"projected; top-1 {sims[top[0]]:.3f} -- and see stage 6 for "
            f"why that number must not be used to rank anything",
            {"top1": round(float(sims[top[0]]), 4),
             "top1_label": labels[top[0]]})


@stage(3, "MathConcept  (extract concepts from the section)")
def _extract():
    from conceptdrill.mathlayer import (LLMExtractor, check_record,
                                        require_measurement_safe)
    from conceptdrill.hierarchy.novita import (DEFAULT_MODEL, load_dotenv,
                                               make_openai_chat, resolve_api_key)
    load_dotenv()
    if not resolve_api_key():
        return "failed", "no NOVITA_API_KEY; the real extractor cannot run", None

    extractor = LLMExtractor(
        make_openai_chat(model=DEFAULT_MODEL, temperature=0.2),
        model=DEFAULT_MODEL)
    require_measurement_safe(extractor)

    records = extractor.extract("s1", "Preliminaries", SECTION_TEXT,
                                latex=SECTION_LATEX, doc_id="fixture:vector-space")
    problems = [(r["concept"], check_record(r)) for r in records
                if check_record(r)]
    for r in records:
        observed = [s["latex"] for s in r["symbols"]
                    if s["status"] == "observed"]
        print(f"  {r['concept']:<22} conf {r['confidence']:<5} "
              f"observed symbols: {observed}")
        for rel in r["related"]:
            print(f"      {rel['relation']:<14} {rel['target']:<24} "
                  f"[{rel['status']}]")
    (HERE / "end-to-end-extracted.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    if problems:
        return "failed", f"records violate the schema: {problems}", None
    return ("ok", f"{len(records)} concepts extracted by {DEFAULT_MODEL}, "
            f"all schema-valid", records)


@stage(4, "LEAN ANCHORS  (propose Mathlib names for the extracted concepts)")
def _anchors():
    from conceptdrill.mathlayer import anchor, confidence

    extracted = next((r["detail"] for r in results
                      if r["stage"] == 3 and r["status"] == "ok"), None)
    if not extracted:
        return "failed", "stage 3 produced nothing to anchor", None

    # Proposed from the step 9/10 table, by concept name. Deliberately NOT
    # asked of the model: step 11 showed a model's memory of Mathlib is a dated
    # snapshot, so proposing and verifying must be separate.
    table = {
        "vector space": ["Module"], "field": ["Field", "DivisionRing"],
        "dimension": ["Module.rank", "Module.finrank",
                      "FiniteDimensional.finrank"],
        "linear map": ["LinearMap"], "basis": ["Basis", "Module.Basis"],
        "linear transformation": ["LinearMap"],
    }
    anchored, proposed = [], 0
    for record in extracted:
        names = table.get(record["concept"], [])
        record = dict(record)
        record["lean"] = [anchor(n) for n in names]
        record["confidence"] = confidence(record)
        proposed += len(names)
        anchored.append(record)
        print(f"  {record['concept']:<22} {names or '(none expected)'}")
    return ("ok", f"{proposed} names proposed, all `unverified`", anchored)


@stage(5, "LOOGLE  (verify every anchor against the live service)")
def _loogle():
    from conceptdrill.mathlayer import (HttpLoogleClient, LoogleUnavailable,
                                        verify)

    anchored = next((r["detail"] for r in results
                     if r["stage"] == 4 and r["status"] == "ok"), None)
    if not anchored:
        return "failed", "stage 4 produced no anchors to verify", None
    try:
        verified = verify(anchored, HttpLoogleClient(),
                          progress=lambda n, r: print(
                              f"  {n:<30} {r['status']:<12}"
                              f"{r.get('hits') or ''}"
                              f"  {r.get('error') or ''}"))
    except LoogleUnavailable as exc:
        return "failed", f"service unreachable: {exc}", None

    supported = sum(1 for r in verified for a in r["lean"]
                    if a["status"] == "supported")
    rejected = [a["name"] for r in verified for a in r["lean"]
                if a["status"] == "unresolved"]
    (HERE / "end-to-end-verified.json").write_text(
        json.dumps(verified, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return ("ok", f"{supported} supported, {len(rejected)} rejected "
            f"{rejected}", verified)


@stage(6, "SEMANTIC NEIGHBOURHOOD  (what CES puts near this mathematics)")
def _neighbourhood():
    path = HERE / "ces-neighbourhood.json"
    if not path.exists():
        return "failed", "run ces_neighbourhood.py first", None
    data = json.loads(path.read_text(encoding="utf-8"))["results"]

    tops = {k: v["hits"][0]["label"] for k, v in data.items() if "hits" in v}
    from collections import Counter
    hubs = Counter(tops.values()).most_common(2)
    spread = [v["top1"] for v in data.values() if "top1" in v]
    print(f"  probes: {len(spread)}   top-1 spread "
          f"{min(spread):.3f} to {max(spread):.3f}")
    for label, n in hubs:
        print(f"  {n}x top-1: {label[:76]}")
    delta = data.get("delta:s3-fixture->s3-variant", {})
    print(f"  'be linear' vs 'be a linear transformation': "
          f"top1 {delta.get('top1_change'):+.4f}, "
          f"{delta.get('shared_of_top_k')}/{delta.get('top_k')} of top-k retained")

    return ("ok", f"measured; {hubs[0][1]} of {len(tops)} probes share one "
            f"top-1 row, so this is reported and NOT used as a signal",
            {"hub_wins": hubs[0][1], "spread": [min(spread), max(spread)]})


@stage(7, "PREDICTED NEXT MATHEMATICS  (from semantic context only)")
def _predict():
    verified = next((r["detail"] for r in results
                     if r["stage"] == 5 and r["status"] == "ok"), None)
    if not verified:
        return "failed", "stage 5 produced no verified concepts", None

    predicted = [(s["latex"], r["concept"]) for r in verified
                 for s in r["symbols"] if s["status"] == "predicted"]
    print("  symbols the concepts imply but the text does not contain:")
    for latex, concept in predicted or [("(none proposed)", "-")]:
        print(f"    {latex:<28} ({concept})")

    path = HERE / "ces-neighbourhood.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))["results"]
        ranked = sorted(((k[5:], v["top1"]) for k, v in data.items()
                         if k.startswith("next:")), key=lambda kv: -kv[1])
        print("\n  continuations ranked by CES (measured, and unusable):")
        for name, score in ranked:
            print(f"    {score:.3f}  {name}")
        print("    ^ the top two are the SAME corpus row; see stage 6")

    if not predicted:
        # The stage ran and produced nothing, which is not success. Rule 2 of
        # the extraction prompt -- INVENT NOTHING -- appears to suppress the
        # predicted channel along with the invented one, so the model emits
        # only symbols it can see. Conservative and safe, and it means this
        # pipeline currently cannot predict.
        return ("degraded", "no predicted symbols at all: the extractor emits "
                "only observed ones, so the prediction channel is empty", [])
    return ("ok", f"{len(predicted)} predicted symbols, each marked "
            f"`predicted` and never `observed`", predicted)


# --------------------------------------------------------------------------

def main() -> int:
    ok = [r for r in results if r["status"] == "ok"]
    degraded = [r for r in results if r["status"] == "degraded"]
    failed = [r for r in results if r["status"] == "failed"]

    print(f"\n{BAR}\nSUCCESSFUL PATHS\n{BAR}")
    for r in ok:
        print(f"  {r['stage']}. {r['name'].split('  ')[0]:<26} {r['reason']}")

    print(f"\n{BAR}\nFAILURES AND DEGRADED PATHS  (reported separately)\n{BAR}")
    if not degraded and not failed:
        print("  none")
    for r in degraded:
        print(f"  DEGRADED  {r['stage']}. {r['name'].split('  ')[0]:<24} "
              f"{r['reason']}")
    for r in failed:
        print(f"  FAILED    {r['stage']}. {r['name'].split('  ')[0]:<24} "
              f"{r['reason']}")

    (HERE / "end-to-end.json").write_text(json.dumps(
        {"stages": [{k: v for k, v in r.items() if k != "detail"}
                    for r in results],
         "ok": len(ok), "degraded": len(degraded), "failed": len(failed)},
        indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n{len(ok)} ok, {len(degraded)} degraded, {len(failed)} failed")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
