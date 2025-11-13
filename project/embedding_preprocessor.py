import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# Load CSV
df = pd.read_csv("drugs_for_common_treatments.csv")

print("Kolom tersedia:", df.columns)

# Buat deskripsi dan kondisi unik
condition_groups = df.groupby("medical_condition")  # ✅ GANTI di sini
condition_list = []
for condition, group in condition_groups:
    desc = group["medical_condition_description"].iloc[0]  # ✅ Sesuaikan kolom deskripsi
    drugs = group[["drug_name", "rating", "no_of_reviews"]].to_dict(orient="records")
    condition_list.append({
        "condition": condition,
        "description": desc,
        "drugs": drugs
    })

# Encode menggunakan pretrained model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
condition_names = [c["condition"] for c in condition_list]
condition_embeddings = model.encode(condition_names, convert_to_numpy=True)

# Simpan embedding dan data
np.save("condition_embeddings.npy", condition_embeddings)
with open("condition_data.pkl", "wb") as f:
    pickle.dump(condition_list, f)

print("Embedding dan data kondisi berhasil disimpan.")
