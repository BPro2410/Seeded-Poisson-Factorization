# ---- Compute model stats --------


df1 = read.csv("/Users/bernd/Documents/01_Coding/02_GitHub/Tensorflow_Probability/keyATM/paper_code/final_model_fits/VSTM/20k/20k_VSTM_df.csv")


# CONFUSION MATRIX
conf_table = table(df1$Cat1, df1$keyatm_label)
conf_table

caret::confusionMatrix(conf_table)

abc = caret::confusionMatrix(conf_table)
colSums(abc$byClass)/6


load("/Users/bernd/Documents/01_Coding/02_GitHub/Tensorflow_Probability/keyATM/paper_code/final_model_fits/SeededLDA/30k/30k_seededlda_model.rds")
write.csv(amazon, "/Users/bernd/Documents/01_Coding/02_GitHub/Tensorflow_Probability/keyATM/paper_code/final_model_fits/SeededLDA/30k/30k_amazon_fit.csv")
