# This script predicts wine price given wine reviews and the wine variety 
# using a wide & deep model implemented in R Keras.
# Code roughly follows python example in this blog post: 
# https://blog.tensorflow.org/2018/04/predicting-price-of-wine-with-keras-api-tensorflow.html


# Load libraries ----------------------------------------------------------

library(tidyverse)
library(magrittr)
library(keras)

# Read & wrangle data ------------------------------------------------------

wine_ratings <- readr::read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2019/2019-05-28/winemag-data-130k-v2.csv")

#Data cleaning
wine_ratings %<>% 
  drop_na(price, description, variety) %>%  
  mutate_at(c("country", "province", "variety", "winery", "region_1", "region_2"), as_factor)

#Keep only the most common varieties with atleast "threshold" occurrences
threshold <- 500
most_common_vareties <- wine_ratings %>% count(variety) %>% filter(n > threshold) %>% select(variety)
wine_ratings %<>% right_join(most_common_vareties)


# Split data into train and test ------------------------------------------

train <- wine_ratings %>% slice_sample(prop = 0.8)
test <- wine_ratings %>% anti_join(train, by = "X1")


# Pre-process (vectorize) data --------------------------------------------

#Look at distribution of length of reviews 
desc_summary <- 
  wine_ratings$description %>% 
  strsplit(" ") %>% 
  sapply(length) %>% 
  summary()

vocab_size <- 12000 #Only consider the top words (by freq)
max_length <- desc_summary["Max."] #record length of longest description

#---
#TODO: delete this section if I don't use it.
#TODO: understand this step better (used in blog post). why not use text_tokenizer %>% fit_text_tokenizer? 
#text_vectorization <- layer_text_vectorization(
#  max_tokens = vocab_size, 
#  output_sequence_length = max_length, 
#)
#Learn vocabulary:
#text_vectorization %>% 
#  adapt(train$description) 
#look at vocab:
#get_vocabulary(text_vectorization) #TODO: Stop words weren't removed. Need to worry?
#---

# Tokenize description text
tokenizer <- 
  text_tokenizer(num_words = vocab_size) %>% 
  fit_text_tokenizer(train$description)

# Binary description matrix for wide network:
train_text_binary_matrix <- texts_to_matrix(tokenizer, train$description, mode = "binary") 
test_text_binary_matrix <- texts_to_matrix(tokenizer, test$description, mode = "binary")

# Sequence description matrix for deep network:
train_text_sequence_matrix <- 
  texts_to_sequences(tokenizer, train$description) %>%
  pad_sequences(maxlen = max_length, padding = "post") #Returns a matrix. numcols is equal to max seq length (shorter seqns padded with 0).
test_text_sequence_matrix <- 
  texts_to_sequences(tokenizer, test$description) %>%
  pad_sequences(maxlen = max_length, padding = "post") 

# Convert wine variety to one-hot vectors for wide network (matrix dim is number of samples by number of varieties):
num_varieties <- length(levels(train$variety))
train_variety_binary_matrix <- to_categorical(as.integer(train$variety), num_varieties)
test_variety_binary_matrix  <- to_categorical(as.integer(test$variety), num_varieties) 


# Wide network ------------------------------------------------------------

wide_text_input <- layer_input(shape = vocab_size, name = "wide_text_input") 
wide_variety_input  <- layer_input(shape = num_varieties, name = "wide_variety_input") 
wide_network <- 
  layer_concatenate(list(wide_text_input, wide_variety_input), name = "wide_merged_layer") %>% 
  layer_dense(units = 256, activation = "relu", name = "wide_layer_dense1") %>%
  layer_dense(units = 1, name = "wide_layer_dense2")


# Deep Network ------------------------------------------------------------

deep_text_input <- layer_input(shape = max_length, name = "deep_input")

deep_network <- 
  deep_text_input %>%
  layer_embedding(input_dim = vocab_size,    # "dictionary" size
                  output_dim = 8, #1024, 
                  input_length = max_length, # the length of the sequence that is being fed in
                  name = "embedding") %>%    # output shape will be batch size, input_length, output_dim
  layer_flatten(name = "flattened_embedding") %>% 
  layer_dense(units = 1, name = "layer1")
  #layer_dense(units = 1024, activation = "relu", name = "layer1") %>%
  #layer_dense(units =  512, activation = "relu", name = "layer2") %>%
  #layer_dense(units =  256, activation = "relu", name = "layer3") 

#TODO: 
# - Try a pre-trained word embedding.  
# - Author of blog post uses embedding dim of 8 and doesn't use stacked layers after embedding.
# - Author added a 1-dim layer to each of wide and deep networks and then combined those. 

# Combine: Wide & Deep ----------------------------------------------------

response <- 
  layer_concatenate(list(wide_network, deep_network), name = "wide_deep_concat") %>%
  layer_dense(units = 1, name = "prediction") 

model <- keras_model(list(wide_text_input, wide_variety_input, deep_text_input), response)

# Compile model ---------------------------------------------------

model %>% compile(
  optimizer = "adam",
  loss = "mse", 
  metrics = c("accuracy")
)

summary(model)

# Train model -------------------------------------------------------------

# First define callbacks to stop model early when validation loss increases and to save best model
#callback_list <- list(
#  callback_early_stopping(patience = 2),
#  callback_model_checkpoint(filepath = "model.h5", monitor = "val_loss", save_best_only = TRUE)
#)

# Train model
history <- 
  model %>% 
  fit(
    x = list(wide_text_input = train_text_binary_matrix, 
             wide_variety_input = train_variety_binary_matrix, 
             deep_text_input = train_text_sequence_matrix),
    y = as.array(train$price),
    epochs = 2,
    batch_size = 128, 
#    validation_data = list(list(user_input = as.array(validation$user), 
#                                item_input = as.array(validation$item)), 
#                           as.array(validation$label)),
    shuffle = TRUE, 
#    callbacks = callback_list
  ) 

#TODO: There is some work to be done here to create a validation set to ensure
#      avoid over-fitting & callbacks for choosing best model. Stopping now
#      because my goal was simply to follow blog post.
