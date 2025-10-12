from annoy import AnnoyIndex
import numpy as np

d = 128
n_data = 10000
n_query = 3

np.random.seed(0)
data_vectors = np.random.random((n_data, d)).astype('float32')
data_queries = np.random.random((n_query, d)).astype('float32')

annoy = AnnoyIndex(d, metric='euclidean')

for i in range(n_data):
    annoy.add_item(i, data_vectors[i])

annoy.build(69)

for i in range(n_query):
    neighbors = annoy.get_nns_by_vector(data_queries[i], 10, include_distances=True)
    print(neighbors)
