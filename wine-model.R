# This script predicts wine price given wine reviews and the wine variety 
# using a wide & deep model implemented in R Keras.
# Code roughly follows python example in this blog post: 
# https://blog.tensorflow.org/2018/04/predicting-price-of-wine-with-keras-api-tensorflow.html


# Load libraries ----------------------------------------------------------

library(tidyverse)
library(magrittr)
library(keras)

# Read & wrangle data ------------------------------------------------------

# Dataset originally from Kaggle: https://www.kaggle.com/zynicide/wine-reviews/data
# Also available for download from TidyTuesday GitHub repo
wine_ratings <- readr::read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2019/2019-05-28/winemag-data-130k-v2.csv")

#Data cleaning
wine_ratings %<>% 
  drop_na(country, price, description, variety) %>%  
  mutate_at(c("country", "province", "variety", "winery", "region_1", "region_2"), as_factor)

#Keep only the most common varieties with atleast "threshold" occurrences
threshold <- 500
most_common_vareties <- wine_ratings %>% count(variety) %>% filter(n > threshold) %>% select(variety)
wine_ratings %<>% 
  right_join(most_common_vareties) %>%
  mutate(variety = fct_drop(variety)) # drop factors that are no longer present

# Split data into train and test ------------------------------------------

train <- wine_ratings %>% slice_sample(prop = 0.8)
test <- wine_ratings %>% anti_join(train, by = "X1")


# Pre-process (vectorize) data --------------------------------------------

#Look at distribution of length of descriptions 
desc_summary <- 
  wine_ratings$description %>% 
  strsplit(" ") %>% 
  sapply(length) %>% 
  summary()

vocab_size <- 12000 #Only consider the top words (by freq)
max_length <- desc_summary["Max."] # length of longest description


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

# Convert wine variety to one-hot vectors for wide network (resulting matrix dim is number of samples by number of varieties):
num_varieties <- length(levels(train$variety)) +1
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

deep_text_input <- layer_input(shape = max_length, name = "deep_text_input")

deep_network <- 
  deep_text_input %>%
  layer_embedding(input_dim = vocab_size,    # "dictionary" size
                  output_dim = 1024, 
                  input_length = max_length, # the length of the sequence that is being fed in
                  name = "embedding") %>%    # output shape will be batch size, input_length, output_dim
  layer_flatten(name = "flattened_embedding") %>% 
  layer_dense(units = 540, activation = "relu", name = "layer1") %>%
  layer_dense(units = 256, activation = "relu", name = "layer2") %>%
  layer_dense(units = 128, activation = "relu", name = "layer3") %>%
  layer_dense(units = 64, activation = "relu", name = "layer4") %>%
  layer_dense(units = 32, activation = "relu", name = "layer5") %>%
  layer_dense(units = 16, activation = "relu", name = "layer6") %>%
  layer_dense(units = 8, activation = "relu", name = "layer7") %>%
  layer_dense(units = 1, name = "layer_last")

#Note:
# - The deep layers above were not included in the example tensorflow blog post.

#TODO: 
# - Try a pre-trained word embedding.  

# Combine: Wide & Deep ----------------------------------------------------

output <- 
  layer_concatenate(list(wide_network, deep_network), name = "wide_deep_concat") %>%
  layer_dense(units = 1, name = "prediction") 

model <- keras_model(list(wide_text_input, wide_variety_input, deep_text_input), output)

# Compile model ---------------------------------------------------

model %>% compile(
  optimizer = "adam",
  loss = "mse", 
  metrics = c("accuracy")
)

summary(model)

# Train model -------------------------------------------------------------

history <- 
  model %>% 
  fit(
    x = list(wide_text_input = train_text_binary_matrix, 
             wide_variety_input = train_variety_binary_matrix, 
             deep_text_input = train_text_sequence_matrix),
    y = as.array(train$price),
    epochs = 5,
    batch_size = 128, 
    shuffle = TRUE
  ) 

#TODO: There is some work to be done here to create a validation set to 
#      avoid over-fitting. Stopping here because my goal was simply 
#      to follow blog post.


# Evaluate model ----------------------------------------------------------

model %>% evaluate(list(test_text_binary_matrix, 
                        test_variety_binary_matrix, 
                        test_text_sequence_matrix), 
                   as.array(test$price))

# Generate predictions for test data --------------------------------------

predictions <- 
  model %>% 
  predict(list(test_text_binary_matrix, 
                         test_variety_binary_matrix, 
                         test_text_sequence_matrix)) %>%
  bind_cols(test %>% select(price, description, variety)) %>%
  rename(pred = 1) %>% #rename 1st column to "pred"
  mutate(diff = abs(pred - price))
  

#Plot a sample of the prices and their associated prediction:
sample4inspection <- slice_sample(predictions, n = 100)
ggplot(sample4inspection, aes(x = 1:nrow(sample4inspection))) +
  geom_line(aes(y = pred), color = "darkred", linetype="twodash") +
  geom_line(aes(y = price), color = "steelblue") +
  xlab("Sample number") +
  ylab("Dollars") +
  ggtitle("Price (blue line) vs Prediction (red dashed line)")
  
# Print some predictions and their descriptions:
for(i in 1:nrow(sample4inspection)){
  print(sprintf("Sample: %d | Price: %.0f | Prediction: %.0f | Description: %s", 
                i,
                sample4inspection[i,"price"], 
                round(sample4inspection[i,"pred"]), 
                sample4inspection[i,"description"]))
}

# Look at difference between price and predicted price:
sprintf('Average prediction difference: %f', mean(predictions$diff) )
sprintf('Median prediction difference: %f', median(predictions$diff) )

# Look at best and worst predictions: 
slice_min(predictions, order_by = diff, n = 5)
slice_max(predictions, order_by = diff, n = 5)

