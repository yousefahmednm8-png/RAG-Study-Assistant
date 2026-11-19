INSERT INTO documents (title,source_path)
VALUES 
('Intro to RAG','docs/rag_intro.pdf'),
('Transformers notes','docs/transformers.pdf'),
('Empty document','docs/empty.pdf');

INSERT INTO chunks(document_id,chunk_index,content,page)VALUES
(1, 0, 'RAG combines retrieval with generation.', 1),
  (1, 1, 'A retriever finds relevant chunks.', 1),
  (1, 2, 'The generator writes the answer from the chunks.', 2),
  (2, 0, 'Attention lets tokens look at each other.', 1),
  (2, 1, 'Multi-head attention uses several heads.', 2);
