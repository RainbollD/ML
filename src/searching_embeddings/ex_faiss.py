import faiss
import numpy as np

d = 128
n_data = 10000
n_query = 3

np.random.seed(0)
data_vectors = np.random.random((n_data, d)).astype('float32')
data_queries = np.random.random((n_query, d)).astype('float32')

index = faiss.IndexFlatL2(d)
index.add(data_vectors)

D, I = index.search(data_queries, n_data)
print(I)





