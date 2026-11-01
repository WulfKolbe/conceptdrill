KIND: topic

You name one topic of a knowledge base and say how its entries relate. Its
reader is a small language model choosing which entry to open.

You receive the topic's entries, each with its title and summary. Use only
those.

Reply with exactly one JSON object and nothing else:

{"title": "a name of at most 6 words for what the entries share",
 "purpose": "one sentence: what a reader finds in this topic",
 "differ": "two to five sentences: how the entries' approaches differ and when to pick which; the single word None. if they do not compare"}
