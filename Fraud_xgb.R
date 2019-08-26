################################################################################################
#### This code is for the Fraud Detection Kaggle Competition
## Created: August 21, 2019
## Edited:
################################################################################################

rm(list = ls())

# set working directory
setwd("/Users/m/Desktop/Kaggle/Fraud")

library(readr)
library(tidyverse)
library(MLmetrics)
library(ggplot2)
library(e1071)
library(pls)
library(dplyr)
library(xgboost)
library(caret)
library(fastDummies)
library(tictoc)
#library(lightgbm)




################################################################################################
# load the data
################################################################################################
train_id <- read.csv('train_identity.csv', header=T)
train_tran <- read.csv('train_transaction.csv', header=T)
test_id <- read.csv('test_identity.csv', header=T)
test_tran <- read.csv('test_transaction.csv', header=T)
sub <- read.csv('sample_submission.csv', header=T)






################################################################################################
# prep the data
################################################################################################
train_data <- left_join(train_tran, train_id)
test_data <- left_join(test_tran, test_id)


################################################################################################
# create x and y train + x test sets
################################################################################################
x_train <- train_data %>% select(-isFraud, -TransactionID)
y_train <- train_data$isFraud

x_test <- test_data %>% select(-TransactionID)

# number of missing values per column
colSums(is.na(x_train))
colSums(is.na(x_test))

# remove volumns with missing values
x_train <- x_train[, colSums(is.na(x_train))==0]
x_test <- x_test[, colSums(is.na(x_test))==0]

# prep columns to keep
keep_cols <- intersect(colnames(x_train), colnames(x_test))

# keep only those columns that are in both sets
x_train <- x_train %>% select(keep_cols)
x_test <- x_test %>% select(keep_cols)


# define categorical variables
cat_vars <- names(x_train)[sapply(x_train, is.character)]

# remove useless variables
del_vars = c('P_emaildomain', 'R_emaildomain')
cat_vars = setdiff(cat_vars, del_vars)
x_train <- x_train %>% select(-del_vars)
x_test <- x_test %>% select(-del_vars)

x_train[, cat_vars] <- lapply(x_train[, cat_vars], as.factor)
x_test[, cat_vars] <- lapply(x_test[, cat_vars], as.factor)

# dummy coding
tic('Dummy coding for x_train')
x_train <- dummy_cols(x_train) %>% select(-cat_vars)
toc()

tic('Dummy coding for x_test')
x_test <- dummy_cols(x_test) %>% select(-cat_vars)
toc()

dim(x_train) %>% print
dim(x_test) %>% print

# get same columns for test and train
cols <- intersect(names(x_train), names(x_test))
x_train <- x_train %>% select(cols)
x_test <- x_test %>% select(cols)

dim(x_train) %>% print
dim(x_test) %>% print

x_train$ProductCD <- as.numeric(x_train$ProductCD)
x_train$card4 <- as.numeric(x_train$card4)
x_train$card6 <- as.numeric(x_train$card6)
x_train$M4 <- as.numeric(x_train$M4)

x_test$ProductCD <- as.numeric(x_test$ProductCD)
x_test$card4 <- as.numeric(x_test$card4)
x_test$card6 <- as.numeric(x_test$card6)
x_test$M4 <- as.numeric(x_test$M4)

################################################################################################
# build xgb model
################################################################################################
set.seed(11)
idx <- createDataPartition(y=y_train, p=0.7, list=FALSE)

dtrain <- xgb.DMatrix(data = as.matrix(x_train[idx,]), label = y_train[idx])
dtest <- xgb.DMatrix(data = as.matrix(x_train[-idx,]), label = y_train[-idx])

watchlist <- list(train = dtrain, test = dtest)

tic("Start training with xgb.train")
model2 <- xgb.train(data = dtrain,                     
                    eval.metric = "auc",
                    max.depth = 9, 
                    eta = 0.05, 
                    subsample = 0.9,
                    colsample_bytree = 0.9,
                    nthread = 2, 
                    nrounds = 500,
                    early_stopping_rounds = 20,
                    verbose = 1,
                    watchlist = watchlist,
                    objective = "binary:logistic")
toc()






################################################################################################
# make predictions
################################################################################################
preds_two <- predict(model2, as.matrix(x_test))





################################################################################################
# run the model on the competition data
################################################################################################
sub$isFraud <- preds_two
write.csv(sub, file='sub2.csv', row.names=F)
