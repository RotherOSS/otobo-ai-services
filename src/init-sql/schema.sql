CREATE TABLE fulltext (
  collection_name TEXT NOT NULL,
  source_id TEXT NOT NULL,
  text TEXT NOT NULL,

  PRIMARY KEY (collection_name, source_id)
);

CREATE TABLE source_vector_index_map (
    collection_name TEXT NOT NULL,
    source_id    TEXT NOT NULL,
    vector_id    TEXT NOT NULL,

    PRIMARY KEY (collection_name, source_id, vector_id)
);