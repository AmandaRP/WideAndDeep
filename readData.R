library(tidyverse)
library(magrittr)


# Wine Ratings ------------------------------------------------------------

wine_ratings <- readr::read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2019/2019-05-28/winemag-data-130k-v2.csv")

wine_ratings %<>% 
  drop_na(price, description, variety) %>%  
  mutate_at(c("country", "province", "variety", "winery", "region_1", "region_2"), as_factor)

#Keep only the most common varieties with atleast "threshold" occurrences
threshold <- 500
most_common_vareties <- wine_ratings %>% count(variety) %>% filter(n > threshold) %>% select(variety)
wine_ratings %<>% right_join(most_common_vareties)

#Split data into train and test
train <- wine_ratings %>% slice_sample(prop = 0.8)
test <- wine_ratings %>% anti_join(train, by = "X1")

# Vectorize data


#Look at distribution of length of reviews (max length is 135)
desc_summary <- wine_ratings$description %>% 
  strsplit(" ") %>% 
  sapply(length) %>% 
  summary()

vocab_size <- 12000 #Only consider the top x words (by freq)
max_length <- desc_summary["Max."]

#---
#TODO: understand this step better (used in blog post). why not use text_tokenizer %>% fit_text_tokenizer? 
text_vectorization <- layer_text_vectorization(
  max_tokens = vocab_size, 
  output_sequence_length = max_length, 
)
#Learn vocabulary:
text_vectorization %>% 
  adapt(train$description) 
#look at vocab:
get_vocabulary(text_vectorization) #TODO: Stop words weren't removed. Need to worry?
#---

#Following book:
#TODO: Should I use hashing trick? See page 168.
tokenizer <- text_tokenizer(num_words = vocab_size) %>% fit_text_tokenizer(train$description)
train_text_sequence_matrix <- texts_to_sequences(tokenizer, train$description) %>%
  pad_sequences(maxlen = max_length, padding = "post") #Returns a matrix. numcols is equal to max seq length (shorter seqns padded with 0).
test_text_sequence_matrix <- texts_to_sequences(tokenizer, test$description) %>%
  pad_sequences(maxlen = max_length, padding = "post") 
train_text_binary_matrix <- texts_to_matrix(tokenizer, train$description, mode = "binary") #binary best here b/c need to concat with other binary vectors
test_text_binary_matrix <- texts_to_matrix(tokenizer, test$description, mode = "binary")

# Convert wine variety to one-hot vectors
num_varieties <- length(levels(train$variety))
variety_tokenizer <- text_tokenizer(num_words = num_varieties) %>% fit_text_tokenizer(train$variety)
train_variety_binary_matrix <- texts_to_matrix(variety_tokenizer, train$variety, mode = "binary") 
test_variety_binary_matrix <- texts_to_matrix(variety_tokenizer, test$variety, mode = "binary") 
#library(onehot)

# wide:

wide_text_input <- layer_input(shape = vocab_size, name = "wide_text_input") 
wide_variety_input  <- layer_input(shape = num_varieties, name = "wide_variety_input") 
wide_merged_layer <- layer_concatenate(list(wide_text_input, wide_variety_input)) %>% 
  layer_dense(units = 256, activation = "relu", name = "wide_merged_layer")

# deep: 

deep_input <- layer_input(shape = max_length, name = "deep_input")
#deep_variety_input <- layer_input(shape = max_length, name = "deep_variety_input") #following blog example. They didn't use this.

embedding <- layer_embedding(input_dim = vocab_size,            # "dictionary" size
                             output_dim = 8, 
                             input_length = max_seq_length) %>% # the length of the sequence that is being fed in
  layer_flatten(name = "embedding")

deep_network <- embedding %>% 
  layer_dense(units = 1024, activation = "relu", kernel_regularizer = regularizer_l2(lambda), name = "layer1") %>%
  layer_dense(units =  512, activation = "relu", kernel_regularizer = regularizer_l2(lambda), name = "layer2") %>%
  layer_dense(units =  256, activation = "relu", kernel_regularizer = regularizer_l2(lambda), name = "layer3") %>%
  

# combine wide & deep:

label <- layer_add(list(wide_merged_layer, deep_network), name = "wide_deep_sum") %>%
  layer_dense(units = 1, 
              activation = "sigmoid", 
              kernel_initializer = "lecun_uniform",
              name = "prediction") 

model <- keras_model(list(user_input, item_input), label)