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
# select important feature vars and rename for convenience
################################################################################################
train <- train_tran %>%
  tbl_df() %>%
  select(TransactionID:ProductCD, card4, card6, addr1, P_emaildomain, M1:M9) %>%
  rename(tID = TransactionID,
         tdate = TransactionDT,
         amt = TransactionAmt,
         product = ProductCD,
         cardcompany = card4,
         cardtype=card6) %>%
  mutate(fraudfactor = as.factor(isFraud))

str(train)






################################################################################################
# visualize data
################################################################################################
plot_one <- train %>% # transaction date/time hist
  ggplot(aes(x=tdate, fill=fraudfactor)) +
  geom_histogram(bins=30, color='blue') + 
  annotate('text', label=min(train$tdate), x=0, y=35000, size=4) + 
  annotate('text', label=max(train$tdate), x=max(train$tdate), y=35000, size=4) + 
  annotate('text', label=median(train$tdate), x=median(train$tdate), y=35000, size=4) +
  annotate('text', label="590K Transactions with 3.499% Fraud Rate", x = median(train$tdate), y = -1200, size = 4, color = "#d1517a") + 
  theme_minimal() + 
  scale_fill_manual(values=c("#4698cb", "#d1517a")) + 
  labs(title="Transaction Histogram by Timestamp", 
       subtitle="It is unknown what the actual timecode means") + 
  xlab("Timestamp Bins") + 
  ylab("Transaction Frequency")

plot_one
skewness(train$tdate) # 0.1311541

# concentration of transactions near the beginning of the time stamp
# no obvious visible clues as to when fraud appears
# relatively low skew


mean(train$amt)
median(train$amt)

plot_two <- train %>% # transaction amount histogram
  filter(amt <= 500) %>%
  ggplot(aes(x=amt, fill=fraudfactor)) +
  geom_histogram(bins=100, color='blue') +
  geom_vline(aes(xintercept=mean(train$amt)),
             linetype=2, color='red', size=1.25) +
  geom_vline(aes(xintercept=median(train$amt)),
             linetype=2, size=1.25) +
  annotate('text', label='22K (3.8%) Total Transactions > $500 \n 1K (4.8%) Fraud Transactions > $500',
           x=350, y=25000, size=4) +
  theme_minimal() +
  scale_fill_manual(values=c("#4698cb", "#d1517a")) +
  labs(title='Transaction Amount', 
       subtitle='Black Line = Median Amt ($68), Red Line = Mean Amt ($135)') +
  xlab('Transaction Amount Bins') +
  ylab('Transaction Frequency')

plot_two
skewness(train$amt) # 14.37442

# small number of transactions over $500
# heavily positively skewed
# slightly higher percentage of fraud transactions over $500 than under


plot_three <- train %>% # ECDF graph for transaction amount
  ggplot(aes(amt, color=fraudfactor)) +
  stat_ecdf(geom='step', pad=F, size=2) +
  geom_hline(aes(yintercept=0.25), linetype=2, size=2, color="red") +
  geom_hline(aes(yintercept=0.5), linetype=2, size=2, color="red") +
  geom_hline(aes(yintercept=0.75), linetype=2, size=2, color="red") +
  geom_hline(aes(yintercept=0.95), linetype=2, size=2, color="red") +
  geom_hline(aes(yintercept=0.99), linetype=2, size=2, color="red") +
  annotate('text', label = "25%", x = 0, y = 0.28) + 
  annotate('text', label = "50%", x = 0, y = 0.53) + 
  annotate('text', label = "75%", x = 0, y = 0.78) + 
  annotate('text', label = "95%", x = 0, y = 0.92) + 
  annotate('text', label = "99%", x = 0, y = 1.03) + 
  xlim(0, 500) + 
  theme_bw() + 
  scale_color_manual(values = c("#4698cb", "#d1517a")) + 
  labs(title = "Cumulative Distribution of Transaction Amounts", 
       subtitle = "22,879 rows not shown") + 
  xlab("Transaction Amount Up to $500") + 
  ylab("Cumulative Percentage of Transaction Amounts")

plot_three

# ECDF slope of fraud is above until ~$50







################################################################################################
# visualize data pt. II
################################################################################################
prop.table(table(train$cardcompany))
table(train$cardcompany, train$fraudfactor)

prop.table(table(train$cardtype))
table(train$cardtype, train$fraudfactor)

comp_and_card <- train %>%
  group_by(cardcompany, cardtype) %>%
  summarise(records = n(),
            sumAMT = sum(amt),
            meanAMT = mean(amt),
            medianAMT = median(amt),
            fraud = sum(isFraud)) %>%
  arrange(desc(records)) %>%
  mutate(proprtion = records / nrow(train),
         fraud_proportion = fraud / sum(train$isFraud),
         fraud_rate = fraud / records)

comp_and_card

# visa makes up ~65% of all transactions, mastercard ~32%
# ~74% of transactions are deb, ~25% credit
# credit's fraud rate is higher than that of debit's
# visa and mastercard have similar fraud rates ~6.8/6.9%






################################################################################################
# model prep
################################################################################################
train_model_data <- train %>%
  mutate(cardcompany = recode_factor(train$cardcompany,
                                     'american express' = 'OTHER',
                                     'discover'='OTHER',
                                     'visa'='visa',
                                     'mastercard'='mastercard',
                                     .default='OTHER'),
         cardtype = recode_factor(train$cardtype,
                                  'charge card' = 'OTHER',
                                  'debit or credit' = 'OTHER',
                                  'credit' = 'credit',
                                  'debit' = 'debit',
                                  .default = 'OTHER')) %>%
  select(amt, cardcompany, cardtype, isFraud)


# create test/training split for model based on training data
inTrain <- sample(1:nrow(train_model_data), nrow(train_model_data)*0.8)
model_train <- train_model_data[inTrain,]
model_test <- train_model_data[-inTrain,]

prop.table(table(model_train$isFraud))
prop.table(table(model_test$isFraud))
# comparable proportion of fraud in each set







################################################################################################
# build the model
################################################################################################
log_model_1 <- glm(isFraud ~., family=binomial, data=model_train)
summary(log_model_1)

mod1_preds <- predict(log_model_1, newdata=model_test, type='response')
model1 <- model_test %>% 
  mutate(m1_p1 = mod1_preds, 
         m1_result = mod1_preds >= 0.065, 
         correct_ind = ifelse((isFraud == 1 & m1_result == TRUE) | (isFraud == 0 & m1_result == FALSE), 1, 0))

sum(model1$correct_ind) / nrow(model_test) # 0.773724







################################################################################################
# run the model on the competition data
################################################################################################
test <- test_tran %>%
  tbl_df() %>%
  select(TransactionID:ProductCD, card4, card6, addr1, P_emaildomain, M1:M9) %>%
  rename(tID = TransactionID,
         tdate = TransactionDT,
         amt = TransactionAmt,
         product = ProductCD,
         cardcompany = card4,
         cardtype=card6)

str(test)


test_model_data <- test %>%
  mutate(cardcompany = recode_factor(test$cardcompany,
                                     'american express' = 'OTHER',
                                     'discover'='OTHER',
                                     'visa'='visa',
                                     'mastercard'='mastercard',
                                     .default='OTHER'),
         cardtype = recode_factor(test$cardtype,
                                  'charge card' = 'OTHER',
                                  'debit or credit' = 'OTHER',
                                  'credit' = 'credit',
                                  'debit' = 'debit',
                                  .default = 'OTHER')) %>%
  select(amt, cardcompany, cardtype)


test_preds <- predict(log_model_1, newdata=test_model_data, type='response')

test_result <- test_model_data %>%
  mutate(predicted_prob = test_preds,
         TransactionID = test$tID,
         isFraud = ifelse(predicted_prob >= 0.065, 1, 0))

submission <- select(test_result, TransactionID, isFraud)
write.csv(submission, file='sub1.csv', row.names=F)




