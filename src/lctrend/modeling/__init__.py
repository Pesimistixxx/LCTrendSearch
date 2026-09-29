"""Auditable weak-signal labels and optional statistical models.

The domain has three levels over one storage layout (``storage``):

1. ``dataset``  forms the sample: first merges near-duplicate
   technologies in the graph (fewer duplicates, more links each), then
   point-in-time history rows, subgraphs, neighbour aggregates, labels and
   splits;
2. ``training`` forms and trains the models: CatBoost on the feature
   matrix, HGT on the subgraphs, and their explanations;
3. ``labeling`` produces the final marking of technologies: the
   calibrated probability of a trained model for each of them.
"""
