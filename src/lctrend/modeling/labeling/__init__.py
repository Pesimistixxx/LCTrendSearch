"""Level 3: the final marking of technologies.

``scoring`` gives every technology the calibrated probability of a
trained model; ``graph_labels`` writes that probability and the LLM labels
onto the Technology nodes. Merging duplicates is step one of level 1
(``dataset.deduplication``), not a final touch.
"""
