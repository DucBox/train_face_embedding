import numpy as np
import polars as pl
df_normalize = pl.read_parquet("/workspace/FaceNist/raw_data_processing/output_parquet/data_process/face_embedding_normalize/webface42m_embedding_normalize/webface_normalized_webface_embeddings_output_p3/embed_part_0002_norm.parquet")

for i in range(0, 100):
    sample_normalize = np.array(df_normalize["embedding_normalized"][i])
    # sample = np.array(df_normalize["embedding_center"][i])
    # print(np.linalg.norm(sample))
    print(np.linalg.norm(sample_normalize))