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
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
    sentence = "I like to eat apples"
    input_ids = tokenizer([sentence], return_tensors='pt')
    with torch.no_grad():
        hidden_states = model(**input_ids)[0][0].numpy()
    print(model_name_or_path)
    print(cosine_similarity(hidden_states, hidden_states))

if __name__ == '__main__':
    test("bert-base-cased")

    test("roberta-base")