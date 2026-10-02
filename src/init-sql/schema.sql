CREATE TABLE fulltext_documents (
  collection_name VARCHAR(255) NOT NULL,
  source_id VARCHAR(255) NOT NULL,
  text LONGTEXT NOT NULL,
  labels JSON DEFAULT NULL,

  PRIMARY KEY (collection_name, source_id)
);

CREATE TABLE source_vector_index_map (
    collection_name VARCHAR(255) NOT NULL,
    source_id VARCHAR(255) NOT NULL,
    vector_id VARCHAR(255) NOT NULL,
    labels JSON DEFAULT NULL,

    PRIMARY KEY (collection_name, source_id, vector_id)
);
