import hnswlib
import numpy as np

dim = 128
num_elements = 10000

data = np.random.random((num_elements, dim)).astype('float32')

index = hnswlib.Index(space='cosine', dim=dim)
index.init_index(max_elements=num_elements, ef_construction=100, M=32)
index.add_items(data)

index.set_ef(50)

labels, distances = index.knn_query(data[:3], k=5)
print(labels)