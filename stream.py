import streamlit as st
from transformers import BertTokenizer, BertModel
import torch
import random

#model and tokenizer
tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")
model=BertModel.from_pretrained("bert-base-uncased")
embedding_matrix=model.get_input_embeddings().weight.detach()

def get_top_k_similar(token_id,k=10):
    if isinstance(token_id, torch.Tensor):
        token_id =token_id.item()
        
    target_embedding=embedding_matrix[token_id]
    similarities=torch.nn.functional.cosine_similarity(target_embedding.unsqueeze(0),embedding_matrix)
    top_k_indices=torch.topk(similarities, k + 1).indices #indices tensor
    #list of python int and filter
    top_k_ids=[idx.item() for idx in top_k_indices if idx.item() != token_id][:k]
    return top_k_ids

def smart_mutate_token(token_ids_input,k=10):
    token_ids=token_ids_input.clone() 

    if len(token_ids) <= 2:
        return token_ids, -1, -1, -1 

    # Valid indices for mutation
    valid_indices=[i for i in range(1, len(token_ids) - 1)]
    
    if not valid_indices:
        return token_ids, -1, -1, -1

    idx_to_mutate=random.choice(valid_indices)
    original_token_id_val=token_ids[idx_to_mutate].item()
    
    similar_ids =get_top_k_similar(original_token_id_val, k) #similar_ids is a list of ints
    
    if not similar_ids:
        return token_ids, idx_to_mutate, original_token_id_val, original_token_id_val 

    mutated_token_id_val=random.choice(similar_ids) 
    
    #new token
    if mutated_token_id_val == original_token_id_val and len(similar_ids) > 1:
        different_similar_ids=[sid for sid in similar_ids if sid != original_token_id_val]
        if different_similar_ids:
            mutated_token_id_val=random.choice(different_similar_ids)

    token_ids[idx_to_mutate] = mutated_token_id_val #int to tenosrs
    
    return token_ids, idx_to_mutate, original_token_id_val, mutated_token_id_val

st.title("wordpiece mutation ")

if 'mutation_details' not in st.session_state:
    st.session_state.mutation_details = None
if 'last_input_sentence' not in st.session_state:
    st.session_state.last_input_sentence = "" 

input_sentence=st.text_input("enter a sentence to mutate:", "follow primus__11 on x!")

if input_sentence and (input_sentence != st.session_state.last_input_sentence or st.session_state.mutation_details is None):
    st.session_state.last_input_sentence=input_sentence

    encoding=tokenizer(input_sentence, return_tensors='pt', add_special_tokens=True)
    original_token_ids=encoding["input_ids"][0].clone()
    original_tokens_list=tokenizer.convert_ids_to_tokens(original_token_ids)

    mutated_ids_output, idx, orig_id_val, new_id_val = smart_mutate_token(original_token_ids.clone())

    current_mutation_error = None
    mutated_sentence_display = input_sentence #default to original if muation fails

    if idx != -1: 
        if orig_id_val != new_id_val:
            mutated_sentence_display = tokenizer.decode(mutated_ids_output, skip_special_tokens=True)
        else:
            original_token_str = tokenizer.convert_ids_to_tokens([orig_id_val])[0] if orig_id_val != -1 else "N/A"
            current_mutation_error = f"token ('{original_token_str}') was selected, but no token found"
            mutated_sentence_display = input_sentence #
        
        st.session_state.mutation_details ={
            "original_sentence_text": input_sentence,
            "original_tokens_list": original_tokens_list,
            "mutated_sentence_str": mutated_sentence_display,
            "idx_mutated": idx,
            "original_token_id": orig_id_val,
            "new_token_id": new_id_val,
            "error_message": current_mutation_error
        }
    else: # No valid token found for mutation
        current_mutation_error = "sentence too short"
        st.session_state.mutation_details = {
            "original_sentence_text": input_sentence,
            "original_tokens_list": original_tokens_list,
            "mutated_sentence_str": input_sentence, # No change
            "idx_mutated": -1,
            "original_token_id": -1,
            "new_token_id": -1,
            "error_message": current_mutation_error
        }
    #rerun
if st.session_state.mutation_details and st.session_state.mutation_details["original_sentence_text"] == input_sentence:
    details = st.session_state.mutation_details

    if details["error_message"] and details["idx_mutated"] == -1 : 
        st.error(details["error_message"])
    elif details["error_message"]: #token didn't change
        st.warning(details["error_message"])

    st.subheader("mutated sentence") 
    st.write(details["mutated_sentence_str"])

    if details["idx_mutated"] != -1 and details["original_token_id"] != details["new_token_id"] and not details["error_message"]:
        st.write("Guess what changed ...!!")

    if st.button("Reveal Mutation"):
        st.subheader(" say cheese if you are right")
        st.write(f"**Original Sentence**: `{details['original_sentence_text']}`")
        st.write(f"**Original Tokens**: `{' '.join(details['original_tokens_list'])}`")

        if details["idx_mutated"] != -1:
            original_token_str = tokenizer.convert_ids_to_tokens([details['original_token_id']])[0] if details['original_token_id'] != -1 else "N/A"
            mutated_token_str = tokenizer.convert_ids_to_tokens([details['new_token_id']])[0] if details['new_token_id'] != -1 else "N/A"

            st.success(f"changed token at index {details['idx_mutated']} (0-indexed in the token list):")
            st.write(f"**original token value**: `{original_token_str}`")
            if details['original_token_id'] != details['new_token_id']:
                 st.write(f"**mutated token value**: `{mutated_token_str}`")
            else:
                st.write(f"*(Token selected but not changed)*")
        else:
            st.info("no mutation")
elif not input_sentence:
    st.info(" enter a sentence to mutate dude ")
