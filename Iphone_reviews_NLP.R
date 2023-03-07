### Analysis of Iphone reviews from Amazon.com ###
# Script by: Banabas Ejiofor
# Data source: https://www.kaggle.com/datasets/thedevastator/apple-iphone-11-reviews-from-amazon-com?resource=download
# This script contains the cleaning process and the NLP model
# NLP script: Kirill Eremenko and Hadelin de Ponteves


# importing important packages
# install all the packages using install.packages() if you don't have them already
library(data.table)
library(tidyverse)
library(ggplot2)
library("stringr")
library(writexl)
library(qdapRegex)


# importing the dataset
df <- fread("apple_iphone_11_reviews.csv")
str(df)
summary(df)

# data cleaning
unique(df$product)
unique(df$helpful_count)
df$helpful_count <- gsub(",", "",df$helpful_count)
df$helpful_count[df$helpful_count == c("One person found this helpful")] <- 1
df$helpful_count <- as.numeric(gsub("([0-9]+).*$", "\\1", df$helpful_count))
summary(df$helpful_count)

unique(df$url)
# removing special characters
df$url <- str_replace_all(df$url, "[^[:alnum:]]", "")
df$url <- gsub("httpswwwamazoninAppleiPhoneXR64GBBlackproductreviewsB07JWV47JWrefcmcrarpdpagingbtmnext2pageNumber", "", df$url)
df$url <- paste("url",df$url, sep=" ", collapse=NULL)

unique(df$review_rating)
df[, review := substr(df$review_rating, 1,1)]
df$review <- as.numeric(df$review)
summary(df$review)

# removing emotions (emojis)
df$review_text <- gsub("[^\x01-\x7F]", "", df$review_text)

# dropping index column
df = subset(df, select = -c(index))

df <- df %>% distinct()

df$review_rating[df$review_rating == "1.0 out of 5 stars"] <- "1 star"
df$review_rating[df$review_rating == "2.0 out of 5 stars"] <- "2 stars"
df$review_rating[df$review_rating == "3.0 out of 5 stars"] <- "3 stars"
df$review_rating[df$review_rating == "4.0 out of 5 stars"] <- "4 stars"
df$review_rating[df$review_rating == "5.0 out of 5 stars"] <- "5 stars"

unique(df$review_rating)

# exporting data for use in Tableau
write.csv(df, "iphone_reviews_tableau.csv")

# cleaning review text column for NLP
df <- df[review_text != c("NOTE:")]
df <- df[review_text != ""]
df[review_text == ""]
df[, .N, review_title][order(-N)]
df[review_title == c("Happy with the purchase")]

# selecting relevant columns for NLP
review_data <- subset(df, select = c(review_text, review))

# Because I am somewhat lazy today and want to save myself the stress
# of editing a lot of lines of code, I will simply change the column names to
# match that used by Kirill Eremenko and Hadelin de Ponteves in their 
# Machine learning A- Z tutorial on Udermy.
colnames(review_data) <- c("Review", "Liked")

# Natural Language Processing

dataset_original <- review_data
str(dataset_original)
dataset_original$Liked <- as.factor(dataset_original$Liked)

# Cleaning the texts
# install.packages('tm')
# install.packages('SnowballC')
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(dataset_original$Review))
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, stopwords())
corpus = tm_map(corpus, stemDocument)
corpus = tm_map(corpus, stripWhitespace)

# Creating the Bag of Words model
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.999)
dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = dataset_original$Liked

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Random Forest Classification to the Training set
# install.packages('randomForest')
library(randomForest)
classifier = randomForest(x = training_set[-743],
                          y = training_set$Liked,
                          ntree = 10)

# Predicting the Test set results
confusion_matrix = predict(classifier, newdata = test_set[-743])

# Making the Confusion Matrix
cm = table(test_set[, 743], confusion_matrix)
cm

# The model has an accuracy of approximately 75.8%
