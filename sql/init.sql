CREATE TABLE IF NOT EXISTS weak_signal_results (
    signal_id UUID PRIMARY KEY,
    query TEXT NOT NULL,
    technology_concept_id TEXT NOT NULL,
    rank INTEGER NOT NULL CHECK (rank > 0),
    score DOUBLE PRECISION NOT NULL,
    explanation JSONB NOT NULL,
    graph_snapshot_at TIMESTAMPTZ NOT NULL,
    model_version TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (query, graph_snapshot_at, rank)
);
