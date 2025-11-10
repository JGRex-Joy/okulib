import faiss

index = faiss.read_index("data/syngan_kylych/index.faiss")
print("Number of vectors:", index.ntotal)
print("Dimension:", index.d)

for i in range(min(5, index.ntotal)):
    vec = index.reconstruct(i)
    print(f"Vector {i}:", vec[:10])  
