Yes. I would make this a **sequence of very small Claude CLI prompts**, where each invocation performs one transformation and writes its result to a file. The important design choice is: **do not ask Claude to solve the whole semantic→Lean problem in one prompt**.

I would structure the experiment as a new layer around ConceptDrill, without modifying its CES core initially.

### 0. First inspect the repository

```text
You are modifying conceptdrill.
Inspect the repository structure and identify:
1. CES concept data format
2. section summary/abstraction/label pipeline
3. query/projection code
4. best place for an experimental math-semantic layer.
Do not modify files.
```

### 1. Define the experiment

```text
Create docs/math-layer.md.
Describe a non-invasive experimental layer above CES.
Input: section text + spoken math + CES concepts.
Output: mathematical concepts and candidate formal anchors.
Do not implement code yet.
```

### 2. Define the intermediate object

```text
Design a JSON object called MathConcept.
It must connect:
concept, spoken form, LaTeX forms,
symbols, related concepts, Lean anchors,
Loogle queries, evidence and confidence.
Keep it minimal.
```

### 3. Find the actual CES representation

```text
Inspect the code and locate the exact data structure
representing a CES concept and its score/vector.
Report filename, type/interface and example fields.
Do not change anything.
```

### 4. Connect section context

```text
Determine exactly how a section currently reaches the
LLM summarizer. Identify the smallest hook where
section text, spoken math and CES concepts can be
passed to a new math-semantic stage.
Do not modify code.
```

### 5. Create a fixture

```text
Create tests/fixtures/vector-space.md containing:
V is an n-dimensional vector space over K.
W is an m-dimensional vector space over K.
Let T : V → W be linear.
Include spoken and LaTeX versions.
```

### 6. Generate semantic candidates

```text
For the fixture, generate a small ranked list of
mathematical concepts. Prefer concepts already present
in CES. Include vector space, dimension, field,
linear map, basis and domain/codomain where justified.
```

### 7. Add mathematical relations

```text
For each candidate, identify relations implied by the
text: is-a, has-dimension, over-field, maps-to,
domain, codomain, related-to. Do not invent facts.
```

### 8. Add notation candidates

```text
For each mathematical concept, list symbols likely to
represent it in this document. Distinguish observed
symbols from predicted symbols. Preserve LaTeX exactly.
```

### 9. Add Lean vocabulary

```text
For every candidate concept, propose possible Lean/Mathlib
constants or definitions. Do not claim correctness.
Separate exact names from uncertain candidates.
```

### 10. Generate Loogle patterns

```text
For each Lean candidate, generate simple Loogle search
patterns. Prefer constants and type shapes over complete
expressions. Output only valid-looking Lean syntax.
```

### 11. Query Lean

```text
Run the existing Lean/Loogle search mechanism on the
generated candidates. Record successful matches and
their declarations. Do not modify source code.
```

### 12. Feed results back

```text
Compare Loogle results with the proposed MathConcepts.
Keep only supported Lean anchors. Mark unsupported
candidates as unresolved. Preserve the original evidence.
```

### 13. Build the semantic neighborhood

```text
Construct a ranked neighborhood for "vector space":
concepts, symbols, LaTeX forms, spoken forms and
supported Lean declarations. Include confidence and
source/evidence for every entry.
```

### 14. Test context dependence

```text
Now process the next statement:
Let T : V → W be a linear transformation.
Compare its semantic neighborhood with the previous one.
Report additions, removals and changed probabilities.
```

### 15. Test prediction

```text
Given the previous context, predict mathematical objects
likely to occur next. Rank symbols and expressions.
Do not OCR anything. Predict only from semantic context.
```

### 16. Connect to PDFDRILL

```text
Design an interface returning candidate interpretations
for an ambiguous mathematical region. Input: CES state,
spoken math and visual candidates. Output ranked LaTeX
interpretations plus semantic justification.
```

### 17. Keep OCR separate

```text
Ensure the math layer never replaces visual OCR.
It provides priors and constraints only.
Document the boundary between visual evidence,
semantic prediction and formal validation.
```

### 18. Measure the useful quantity

```text
Design a small benchmark measuring whether semantic
context reduces mathematical OCR candidates.
Compare unconstrained glyph candidates against
CES+MathConcept+Lean constrained candidates.
```

### 19. Implement only after the experiment

```text
Review the experiment. Identify the smallest production
implementation that adds the math layer without changing
existing CES behavior. List files to add/change.
Do not implement yet.
```

### 20. Implement the minimal layer

```text
Implement only the minimal MathConcept layer identified
above. Preserve existing APIs and behavior. Add tests.
Do not refactor unrelated code.
```

### 21. Add CLI inspection

```text
Add a CLI command that takes a section and prints:
CES concepts, mathematical concepts, notation candidates,
Lean anchors and Loogle queries. Keep output readable.
```

### 22. Final architectural test

```text
Run the vector-space fixture end-to-end.
Show:
section → CES → MathConcept → Lean anchors → Loogle
→ semantic neighborhood → predicted next mathematics.
Report failures separately from successful paths.
```

## One important refinement

I would **not** make `Lean` part of the first intermediate representation too strongly.

The useful hierarchy is:

```text
CES concept
    ↓
Mathematical concept
    ↓
mathematical role/relation
    ↓
notation candidates
    ↓
LaTeX candidates
    ↓
Lean anchors
    ↓
Loogle
```

rather than:

```text
CES → Lean
```

because the intermediate **mathematical concept layer is exactly what your semantic compiler needs**.

For example:

```json
{
  "concept": "linear map",
  "role": "mapping",
  "domain": "V",
  "codomain": "W",
  "symbols": ["T"],
  "latex": ["T:V\\to W"],
  "lean": ["LinearMap"],
  "loogle": ["LinearMap ?R ?M ?M₂"],
  "confidence": 0.94
}
```

This object is simultaneously useful to **PDFDRILL, spoken-math processing, CES retrieval, OCR disambiguation, LaTeX reconstruction, and Lean search**.

And importantly, it is small enough that your 128-token Claude invocations can progressively construct it rather than trying to regenerate the entire mathematical interpretation at every step.

