import streamlit as st
from transformers import BertTokenizer, BertModel
import torch
import random

#model and tokenizer
tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")
model=BertModel.from_pretrained("bert-base-uncased")
embedding_matrix=model.get_input_embeddings().weight.detach()

def get_top_k_similar(token_id, k=10):
    target_embedding=embedding_matrix[token_id]
    similarities =torch.nn.functional.cosine_similarity(target_embedding.unsqueeze(0),embedding_matrix)
    top_k_ids=torch.topk(similarities, k + 1).indices.tolist()
    top_k_ids=[idx for idx in top_k_ids if idx != token_id][:k]
    return top_k_ids

def smart_mutate_token(token_ids,k=10):
    valid_indices=[i for i in range(1,len(token_ids) - 1)]
    if not valid_indices:
        return token_ids, -1, -1

    idx_to_mutate=random.choice(valid_indices)
    original_token_id=token_ids[idx_to_mutate]
    
    similar_ids=get_top_k_similar(original_token_id, k)
    mutated_token_id=random.choice(similar_ids)
    token_ids[idx_to_mutate]=mutated_token_id
    
    return token_ids, idx_to_mutate, original_token_id, mutated_token_id

st.title("wordpiece mutation")

sentence=st.text_input("enter a sentence to mutate", "Transformers are amazing!")

if sentence:
    encoding =tokenizer(sentence, return_tensors='pt')
    token_ids=encoding["input_ids"][0].clone()
    tokens =tokenizer.convert_ids_to_tokens(token_ids)

    st.write("original tokens:", tokens)

    mutated_ids = token_ids.clone()
    mutated_ids, idx, orig_id, new_id=smart_mutate_token(mutated_ids)

    if idx != -1:
        mutated_sentence=tokenizer.decode(mutated_ids)
        st.subheader(" mutated sentence")
        st.write(mutated_sentence)

        if st.button("reveal Mutation"):
            st.success(f"changed token at index {idx}:")
            st.write(f"**original**: `{tokenizer.convert_ids_to_tokens([orig_id])[0]}`")
            st.write(f"**mutated**: `{tokenizer.convert_ids_to_tokens([new_id])[0]}`")
    else:
        st.error("no valid token found")
