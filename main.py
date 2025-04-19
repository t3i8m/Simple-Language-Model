from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

all_token_ids = []

with open("data/tinyshakespeare.txt", encoding = "utf-8") as f:
    for n in f.readlines():
        tokens = tokenizer(n.strip(), add_special_tokens=True, truncation=True, max_length=512, )["input_ids"]
        all_token_ids.extend(tokens)

vocab = sorted(set(all_token_ids))

word2idx = {token_id: i for i, token_id in enumerate(vocab)}
idx2word = {i: token_id for token_id, i in word2idx.items()}

X = []
Y = []
window = 5
for n in range(window, len(all_token_ids), window):
    X.append(all_token_ids[n-window:n])
    Y.append(all_token_ids[n])

print(X)
# print(Y)

with open("data/train.csv", mode = "w") as f:
    for index in range(0, len(X)-1):
        print([X[index], Y[index]])
        f.write(f"{[X[index], Y[index]]},")
