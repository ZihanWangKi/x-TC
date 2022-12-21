import numpy as np
import transformers
import torch

def cosine_similarity(repr_a: np.ndarray, repr_b: np.ndarray):
    assert 1 <= repr_a.ndim <= 2 and 1 <= repr_b.ndim <= 2
    if repr_a.ndim == 1:
        repr_a = repr_a[np.newaxis, :]
    if repr_b.ndim == 1:
        repr_b = repr_b[np.newaxis, :]
    assert repr_a.shape[1] == repr_b.shape[1]
    cosine_similarities = np.dot(repr_a, np.transpose(repr_b)) / np.outer(np.linalg.norm(repr_a, axis=1),
                                                                          np.linalg.norm(repr_b, axis=1))
    return np.squeeze(cosine_similarities)

def test(model_name_or_path):
    model = transformers.AutoModel.from_pretrained(model_name_or_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    sentence = "Too expensive for such mediocre food! I understand, that to some extent, that buffets generally don't put out 5 star food but there comes a point where the price you are paying is not worth the food. Even if you're going solely for the fresh seafood, in the end, (unless you are a marathon eater) there is no way you could possibly eat the value of food that you were being charged for. As a repeat customer from the past, even then the buffet was pricey but tolerated because of the fresh seafood. With the recent rise in price that is no longer the case."
    #tokenized_text = tokenizer.tokenize(sentence)
    #hidden_states = []
    #embedding =  model.embeddings.word_embeddings.weight
    #with torch.no_grad():
    #    for i in range(len(tokenized_text)):
    #        hidden_states.append(embedding[tokenizer._convert_token_to_id(tokenized_text[i])].numpy())
    #hidden_states = np.array(hidden_states)
    input_ids = tokenizer([sentence], return_tensors='pt')
    with torch.no_grad():
        hidden_states = model(**input_ids)[0][0].numpy()
    print(model_name_or_path)
    print(hidden_states)
    print(cosine_similarity(hidden_states, hidden_states))

    embeddings = hidden_states
    scale = False
    if scale:
        embedding_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        unit_embeddings = embeddings / embedding_norms
        mean_vec = embedding_norms * np.mean(unit_embeddings, axis=0, keepdims=True)
    else:
        mean_vec = np.mean(embeddings, axis=0)
    print(cosine_similarity(embeddings-mean_vec, embeddings-mean_vec))

    ew, ev = np.linalg.eig(np.cov(hidden_states.T))
    pc = ev[:, np.argmax(ew)].reshape(-1, 1)
    pcT = pc.reshape(1, -1)
    print(pc.dot(pcT).shape)
    hidden_states = hidden_states - hidden_states.dot(pc.dot(pcT))
    print(cosine_similarity(hidden_states, hidden_states))

if __name__ == '__main__':
    #test("bert-base-cased")

    test("roberta-large")