
set.seed(2410)

df1 = read.csv("/Users/bernd/Documents/01_Coding/02_GitHub/Tensorflow_Probability/keyATM/paper_code/30k_amazon.csv")

#df1 = read.csv("/Users/Bernd/Documents/01_Coding/02_GitHub/Tensorflow_Probability/5k_amazon_sample.csv")
#df1 = read.csv("/Users/Bernd/Documents/01_Coding/02_GitHub/Tensorflow_Probability/1k_amazon.csv")
library(tidyverse)
library(quanteda)
library(seededlda)
library(stringr)

df1$doc_id = rownames(df1)


# Sample df
df1.1 = sample_n(df1, 1000000, replace = TRUE)
df1.1$doc_id = rownames(df1.1)
#df1.1 = df1[sample(nrow(df1), 5000), ]


# select text
df1.text = df1.1$Text


start_time <- Sys.time()



# Preprocessing -----------------------------------------------------------


# Create corpus
key_corpus = corpus(df1.1, docid_field = "doc_id", text_field = "Text")

# Tokenize each doc
key_token = tokens(key_corpus,
                   remove_numbers = TRUE,
                   remove_punct = TRUE,
                   remove_symbols = TRUE,
                   remove_separators = TRUE,
                   remove_url = TRUE) %>% 
  tokens_tolower() %>% 
  tokens_remove(c(stopwords("english"), "may", "shall", "can"))


# Stemming
#tokens_wordstem(tokens(key_token))

# Lemming
#key_token = tokens_replace(tokens(key_token), pattern = lexicon::hash_lemmas$token, replacement = lexicon::hash_lemmas$lemma)

# Create a dfm (document feature matrix) from a token object
key_dfm = dfm(key_token)

# Subset dfm with docs that have more than 4 tokens!!
key_dfm = dfm_subset(key_dfm, ntoken(key_dfm) > 4)

key_dfm # See! Geniale Darstellung!



pets = c("dog", "cat", "cats", "dogs", "box", "food", "toy", "pet")
toys = c("toy", "game", "play", "fun", "son", "daughter", "kids", "playing", "christmas", "toys", "child", "gift")
beauty = c("skin", "color", "scent", "smell", "dry", "face", "look", "fragrance", "products", "dry", "perfume", "shampoo")
baby = c("baby", "seat", "son", "daughter", "newborn", "months", "diaper", "diapers", "car", "stroller", "pump", "bag", "child")
health = c("try", "long", "water", "feel", "shave", "razor", "shaver", "pain")
grocery = c("tea", "taste", "flavor", "coffee", "chocolate", "sugar", "sauce", "milk", "delicious")


# tf-idf keywords
pets = c("dog","cat", "litter", "cats", "dogs", "food", "box", "collar", "water", "pet")
toys = c("toy", "game", "play", "fun", "old", "son", "year", "loves", "kids", "daughter")
beauty = c("hair", "skin", "product", "color", "scent", "smell", "used", "dry", "using", "products")
baby = c("baby", "seat", "diaper", "diapers", "stroller", "bottles", "son", "pump", "gate", "months")
health = c("product", "like", "razor", "shave", "time", "day", "shaver", "better", "work", "years")
grocery = c("tea", "taste", "flavor", "coffee", "sauce", "chocolate", "sugar", "eat", "sweet", "delicious")



# Specify keywords

keywords = list(
  "Pets" = pets,
  "Toys" = toys,
  "Beauty" = beauty,
  "Baby" = baby,
  "Health" = health,
  "Grocery" = grocery
)

keywords = dictionary(keywords)



# Fit model: https://koheiw.github.io/seededlda/articles/pkgdown/seeded.html
# Early stopping mechanism with argument 'auto_iter = True'
lda_seed = textmodel_seededlda(key_dfm, keywords, verbose = TRUE)

# Measure ex time
end_time <- Sys.time()
end_time - start_time


# Extract topics
preds = lda_seed$theta

# find highest prediction
pred_values = colnames(preds)[max.col(preds, ties.method = "first")]


# filter initial df - remove those short feedbacks which falled short
used_docs = rownames(key_dfm)

df2 = df1.1 %>% 
  filter(doc_id %in% used_docs)

rownames(df2) = df2$doc_id

# combine initial data and predictions
df3 = df2 %>%
  mutate(Prediction = pred_values)


# Recode pred labels

df3$preds_clean = gsub("[0-9.]", "", df3$Prediction)
df3$preds_clean = gsub("_", "", df3$preds_clean)

df3$preds_clean = recode(df3$preds_clean, Health = "health personal care", Toys = "toys games", Grocery = "grocery gourmet food",
                         Baby = "baby products", Pets = "pet supplies", Beauty = "beauty")


df3$Accuracy = ifelse(df3$Cat1 == df3$preds_clean, 1, 0)

sum(df3$Accuracy)




# CONFUSION MATRIX
conf_table = table(df3$Cat1, df3$preds_clean)
conf_table

caret::confusionMatrix(conf_table)

abc = caret::confusionMatrix(conf_table)
colSums(abc$byClass)/6


# save dfs
amazon = df3
save(lda_seed, amazon, file = "/Users/bernd/Documents/01_Coding/02_GitHub/Tensorflow_Probability/keyATM/paper_code/final_model_fits/SeededLDA/1m/1m_seededlda_model.rds")





