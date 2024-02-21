### Seed Topic Model example

# Load and wrangle data ---------------------------------------------------

set.seed(2410)

df1 = read.csv("/Users/bernd/Documents/01_Coding/02_GitHub/Tensorflow_Probability/keyATM/paper_code/10k_amazon.csv")

#df1 = read.csv("/Users/Bernd/Documents/01_Coding/02_GitHub/Tensorflow_Probability/5k_amazon_sample.csv")
#df1 = read.csv("/Users/Bernd/Documents/01_Coding/02_GitHub/Tensorflow_Probability/1k_amazon.csv")
library(tidyverse)
library(quanteda)
library(keyATM)
library(stringr)

df1$doc_id = rownames(df1)


# Sample df
# df1.1 = sample_n(df1, 5000)
#df1.1 = df1[sample(nrow(df1), 5000), ]
df1.1 = df1


# select text
df1.text = df1.1$Text




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


key_dfm2 = dfm_trim(key_dfm, min_termfreq = 3)

# Reformat in required KeyATM format
keyATM_docs = keyATM_read(texts = key_dfm)


summary(keyATM_docs)


# Predefine Keywords ------------------------------------------------------

pets = c("animal", "dog", "cat", "bird", "pet")
toys = c("play", "toy", "kid", "lego", "game", "daughter", "toys")
beauty = c("beauty", "makeup", "hair", "skin", "dryer")
baby = c("baby", "kid", "son", "daughter", "newborn")
health = c("tooth", "lips", "body", "health")
grocery = c("apple", "banana", "food", "grocery", "groceries")


pets = c("dog", "cat", "cats", "dogs", "box", "food", "toy", "pet")
toys = c("toy", "game", "play", "fun", "son", "daughter", "kids", "playing", "christmas", "toys", "child", "gift")
beauty = c("skin", "color", "scent", "smell", "dry", "face", "look", "fragrance", "products", "dry", "perfume", "shampoo")
baby = c("baby", "seat", "son", "daughter", "newborn", "months", "diaper", "diapers", "car", "stroller", "pump", "bag", "child")
health = c("try", "long", "water", "feel", "shave", "razor", "shaver", "pain")
grocery = c("tea", "taste", "flavor", "coffee", "chocolate", "sugar", "sauce", "milk", "delicious")


# lemmatize if necessary
#pets = as.character(tokens_replace(tokens(pets), pattern = lexicon::hash_lemmas$token, replacement = lexicon::hash_lemmas$lemma))
#toys = as.character(tokens_replace(tokens(toys), pattern = lexicon::hash_lemmas$token, replacement = lexicon::hash_lemmas$lemma))
#beauty = as.character(tokens_replace(tokens(beauty), pattern = lexicon::hash_lemmas$token, replacement = lexicon::hash_lemmas$lemma))
#baby = as.character(tokens_replace(tokens(baby), pattern = lexicon::hash_lemmas$token, replacement = lexicon::hash_lemmas$lemma))
#health = as.character(tokens_replace(tokens(health), pattern = lexicon::hash_lemmas$token, replacement = lexicon::hash_lemmas$lemma))
#grocery = as.character(tokens_replace(tokens(grocery), pattern = lexicon::hash_lemmas$token, replacement = lexicon::hash_lemmas$lemma))


keywords = list(
  "Pets" = pets,
  "Toys" = toys,
  "Beauty" = beauty,
  "Baby" = baby,
  "Health" = health,
  "Grocery" = grocery
)


#keywords = list(
#  "Pets" = c("animal", "dog", "cat", "bird", "pet"),
#  "Toys" = c("play", "toy", "kid", "lego", "game", "daughter", "toys"),
#  "Beauty" = c("beauty", "makeup", "hair", "skin", "dryer"),
#  "Baby" = c("baby", "kid", "son", "daughter", "newborn"),
#  "Health" = c("tooth", "lips", "body", "health"),
#  "Grocery" = c("apple", "banana", "food", "grocery", "groceries")
#)





# See if keywords make sense
#key_viz = visualize_keywords(keyATM_docs, keywords)
#key_viz

# Numerical keyword proportions
#values_fig(key_viz)



# Model calculation -------------------------------------------------------

key_model = keyATM(docs = keyATM_docs,
                   no_keyword_topics = 0,
                   keywords = keywords,
                   model = "base", 
                   options = list(seed = 2410, iterations = 1500))



# Model fit -------------------------------------------------------------

# See LL and perplexity
# -> If the model is working as expected, we would observe an increase trend for the log-likelihood 
# and an decrease trend for the perplexity.
plot_modelfit(key_model)

# See top words
top_words(key_model, n = 10)

# Show the expected proportion of the corpus belonging to each estimated topic along with the top three words
# associated with the topic
plot_topicprop(key_model, show_topic = 1:6)


# Document-topic distribution
key_model$theta

# Topic-word distribution
key_model$phi

# Visualize alpha, the prior for the document-topic distribution, and pi, the probability
# that each topic uses keyword topic-word distribution. 
# -> Values of these parameters should also be stabilize over time
plot_pi(key_model)
plot_alpha(key_model)


# top_docs() Zeigt uns pro Thema die wichtigsten Dokumente auf, also diejenigen Dokumente,
# in denen das jeweilige Thema den grössten Anteil hat
top_docs(key_model, n = 5)



# manual exploration ------------------------------------------------------

preds = key_model$theta

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

# accuracy ----------------------------------------------------------------

### ANALYZE CERTAIN SPECIIFC KEYWORDS
## Schaut z.B. welche wörter in einem Thema oft vorgekommen sind!
test = df3 %>% 
  filter(Cat1 == "health personal care")

test_corp = corpus(test, docid_field = "doc_id", text_field = "Text")
test_tokens = tokens(test_corp,
                     remove_numbers = TRUE,
                     remove_punct = TRUE,
                     remove_symbols = TRUE,
                     remove_separators = TRUE,
                     remove_url = TRUE) %>% 
  tokens_tolower() %>% 
  tokens_remove(c(stopwords("english"), "may", "shall", "can"))

test_dfm = dfm(test_tokens)

unigrams = as.data.frame(colSums(test_dfm))
### ENDE



for (label in unique(df3$Cat1)){
  # Iterate through each label
  res = df3 %>%
    filter(Cat1 == label)
  
  # calculate binary accuracy (1 and 0)
  accuracy = sum(res$Accuracy)/nrow(res)
  
  # Print results
  ab = sprintf("Accuracy of label %s: %s", label, accuracy)
  print(ab)
}


# CONFUSION MATRIX
conf_table = table(df3$Cat1, df3$preds_clean)
conf_table

caret::confusionMatrix(conf_table)

abc = caret::confusionMatrix(conf_table)
colSums(abc$byClass)/6


# save dfs
amazon = df3
save(key_model, amazon, file = "/Users/bernd/Documents/01_Coding/02_GitHub/Tensorflow_Probability/keyATM/paper_code/final_model_fits/keyATM/10k/10k_key_model.rds")


# automated keyword selection ---------------------------------------------

# fit LDA and see top words per category
out = weightedLDA(docs = keyATM_docs,
                  number_of_topics = 6,
                  model = "base",
                  options = list(seed = 250))
top_words(out)
top_words(out, n = 40)
