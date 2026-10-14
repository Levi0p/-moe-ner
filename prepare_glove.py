"""
One-time setup: download GloVe 100d (wiki-gigaword) and export the two files train.py reads.

    python prepare_glove.py        # writes glove_vecs.npy and glove_words.txt
"""

import numpy as np
from gensim.models import KeyedVectors
from huggingface_hub import snapshot_download

path = snapshot_download("fse/glove-wiki-gigaword-100")
kv = KeyedVectors.load(f"{path}/glove-wiki-gigaword-100.model")
np.save("glove_vecs.npy", kv.vectors.astype("float32"))
with open("glove_words.txt", "w") as f:
    f.write("\n".join(kv.index_to_key))
print(f"saved {kv.vectors.shape[0]:,} vectors of dim {kv.vectors.shape[1]}")
