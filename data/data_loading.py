from datasets import load_dataset


dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
print(dataset["test"]["text"])

with open("data/data.txt", mode = "w", encoding = "utf-8") as f:
    for n in dataset["train"]["text"]:
        f.write(n)