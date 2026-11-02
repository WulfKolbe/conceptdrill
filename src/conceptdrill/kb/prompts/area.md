KIND: area

You name one area of a knowledge base and say how its topics relate. Its
reader is a small language model choosing which topic to open.

You receive the area's topics, each with its title and purpose. Use only
those.

Reply with exactly one JSON object and nothing else:

{"title": "a name of at most 6 words for what the topics share",
 "purpose": "one sentence: what a reader finds in this area",
 "differ": "two to five sentences: how the topics differ and when to pick which; the single word None. if there is one topic or they do not compare"}
