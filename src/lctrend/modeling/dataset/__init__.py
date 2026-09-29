"""Level 1: forming the sample.

``deduplication`` comes first: it merges near-duplicate technologies (and
other embedded kinds) in the graph, so each survivor carries all the links
of its duplicates before anything is exported.

``annotations`` exports point-in-time history rows, reviewer forms and
subgraphs from the graph; ``neighbors`` adds aggregates of each sampled
neighbourhood to the rows; ``labels`` joins reviews into labels and
splits; ``llm_labels`` writes provisional labels for human review.
"""
