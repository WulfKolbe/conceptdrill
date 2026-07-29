You are generating the semantic PROTOTYPE of a document section — text that will be
EMBEDDED and used as a BASIS VECTOR for similarity comparison, NOT read by humans.

Given the section TITLE and BODY, describe ONLY:
- the central concept this section introduces,
- its purpose within the document,
- its relationship to neighboring concepts,
- the terminology the author uses.

Do NOT include examples, experiments, numerical results, citations, figure/table
details, or implementation minutiae — unless one is essential to DEFINING the concept.
Answer only the question: what semantic concept does this section define?

Produce THREE progressively more abstract descriptions, so the same section yields a
document-faithful vector, a document-independent concept vector, and a reusable label.

Output ONLY a JSON object, all values plain-text strings.
Write NO LaTeX and NO backslashes of any kind: a backslash silently corrupts the
JSON string (\t, \b, \f, \r are legal escapes, so "\tau" becomes a tab). Spell
symbols out as words — write "tau", not the symbol.

{
  "summary":     "<80-150 words: faithful to the author's terminology and scope>",
  "abstraction": "<~70 words: the underlying idea, independent of this document's specifics>",
  "label":       "<30-42 words: a canonical concept definition suitable for reuse across documents>"
}
