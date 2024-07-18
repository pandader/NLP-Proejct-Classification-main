from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pairwise_cos_sim
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model.similarity()