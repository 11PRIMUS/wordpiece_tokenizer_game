import  random
from transformers import BertTokenizer
import torch
import numpy as np

tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')

def mutate_token(token_ids,vocab_size):
  valid_indices=[i for i in range(1,len(token_ids)-1)]
  if not valid_indices:
    return token_ids, -1,-1
  
  idx_to_mutate=random.choice(valid_indices)
  #choose random token from vocab
  original_token_id=token_ids[idx_to_mutate]
  mutate_token_id=random.randit(0,vocab_size-1)

  token_ids[idx_to_mutate]=mutate_token_id
  return token_ids,idx_to_mutate,original_token_id

def play_game():
    sentence=input("enter a sentence: ")

    #tokenize
    encoding=tokenizer(sentence,return_tensors='pt')
    token_ids=encoding['input_ids'][0].clone()
    mutated_id,orig_id,new_id=mutate_token(token_ids,tokenizer.vocab_size)
    mutated_tokens=tokenizer.convert_ids_to_tokens(mutated_id)
    print(f"\n mutated sentence: {tokenizer.decode(mutated_id)}")
    print(f"\n guess which word changed")
    print(f" press enter t reveal answer")

    print(f"\n original_token:{tokenizer.convert_ids_to_tokens([orig_id])[0]}")  
    print(f"mutated token: {tokenizer.convert_ids_to_tokens([new_id])[0]}")

if __name__ == "__main__":
    while True:
        play_game()
        again =input("\nPlay again! (y/n): ")
        if again.lower() != 'y':
            break



