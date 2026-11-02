KIND: kb

You name a knowledge base. Its reader is a small language model deciding
whether this knowledge base answers its question.

You receive the knowledge base's areas, each with its title and purpose. Use
only those.

Reply with exactly one JSON object and nothing else:

{"title": "a name of at most 8 words for the whole knowledge base",
 "purpose": "one or two sentences: what it covers and what it can be used for"}
