import numpy as np
import pandas as pd

df1 = pd.read_csv("/Users/Bernd/Documents/01_Coding/02_GitHub/PhD-NLP/02_Data/01_Testdata/train_40k.csv")


def count_words(text):
    return len(text.split())

df1["tokens"] = df1.Text.apply(count_words)


df1.tokens
np.mean(df1.tokens)
np.median(df1.tokens)
np.std(df1.tokens)
np.quantile(df1.tokens, 0.95)
np.quantile(df1.tokens, 0.05)

np.max(df1.tokens)
np.min(df1.tokens)



pets = ["dog", "cat", "cats", "dogs", "box", "food", "toy", "pet"]
toys = ["toy", "game", "play", "fun", "son", "daughter", "kids", "playing", "christmas", "toys", "child", "gift"]
beauty = ["skin", "color", "scent", "smell", "dry", "face", "look", "fragrance", "products", "dry", "perfume", "shampoo"]
baby = ["baby", "seat", "son", "daughter", "newborn", "months", "diaper", "diapers", "car", "stroller", "pump", "bag", "child"]
health = ["try", "long", "water", "feel", "shave", "razor", "shaver", "pain", "heart", "tooth", "balm", "pads", "capsules", "taste", "drink","medical", "blade", "oil","stethoscope", "plastic", "shaker", "soap"]
grocery = ["tea", "taste", "flavor", "coffee", "chocolate", "sugar", "sauce", "milk", "delicious", "water", "diet"]



keywords = {"pets": pets, "toys": toys, "beauty": beauty, "baby": baby, "health": health, "grocery": grocery}


for k,v in keywords.items():
    print(f"{k}: LÄNGE {len(v)}")
