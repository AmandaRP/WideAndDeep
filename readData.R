library(tidyverse)
library(magrittr)


# Wine Ratings ------------------------------------------------------------

wine_ratings <- readr::read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2019/2019-05-28/winemag-data-130k-v2.csv")

wine_ratings %<>% 
  drop_na(price, description, variety) %>%  
  mutate_at(c("country", "province", "variety", "winery", "region_1", "region_2"), as_factor)

#Keep only the most common varieties
threshold <- 500
most_common_vareties <- wine_ratings %>% count(variety) %>% filter(n > threshold) %>% select(variety)
wine_ratings %<>% right_join(most_common_vareties)

#Split data into train and test
train <- wine_ratings %>% slice_sample(prop = 0.8)
test <- wine_ratings %>% anti_join(train, by = "X1")

#Look at distribution of length of reviews (max length is 135)
wine_ratings$description %>% 
  strsplit(" ") %>% 
  sapply(length) %>% 
  summary()

vocab_size <- 12000
max_length <- 135
text_vectorization <- layer_text_vectorization(
  max_tokens = vocab_size, 
  output_sequence_length = max_length, 
)

#Learn vocabulary:
text_vectorization %>% 
  adapt(wine_ratings$description)

#look at vocab:
get_vocabulary(text_vectorization) #TODO: Stop words weren't removed. Need to worry?

