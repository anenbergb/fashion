import pickle, os
dataset_path      = "/cvgl/u/anenberg/Fashion144k_stylenet_v1/"
pickle3_  = "similar_pairs.pkl"
pickle2_ = "similar_pairs.pkl2"

with open(os.path.join(dataset_path, pickle3_), 'rb') as f:
	similar_pairs = pickle.load(f)
with open(os.path.join(dataset_path, pickle2_), 'wb') as f:
	pickle.dump(similar_pairs, f, protocol=2)