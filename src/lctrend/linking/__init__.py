"""The linking layer: how a text is connected to what the graph knows.

Extraction (llm.pipeline) reads a document; linking decides what the model
sees and what reaches it at all, and relates concepts to each other:

- ``names``: known names of the registry found in a text, by the lexical
  identity key the resolver uses;
- ``records``: grants and vacancies linked to known technologies without
  a model call; a record naming none goes to the model;
- ``known``: the registry concepts a packet names or resembles, given to
  the model as reference names so it reuses them instead of coining
  duplicates (reference, never evidence);
- ``sections``: chunks the model does not need (a paper's data and sample
  description, administrative statements), skipped before any call;
- ``similar``: SIMILAR_TO edges, a mutual nearest-neighbour layer over
  concept vectors, kept apart from evidence-backed relations.
"""
