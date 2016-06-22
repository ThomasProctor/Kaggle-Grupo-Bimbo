library(readr)
library(Metrics)
library(ggplot2)
library(caret)
library(dplyr)
library(tidyr)

sdata <- read_csv('data/train.csv', n_max = 1000000)
str(sdata)

towndata <- read_csv('data/town_state.csv')
str(towndata)
colnames(towndata)[1] <- 'Depot.ID'

clientdata <- read_csv('data/cliente_tabla.csv')
str(clientdata)
colnames(clientdata) <- c('Client.ID','Client.name')

productdata <- read_csv('data/producto_tabla.csv')
str(productdata)
colnames(productdata) <- c('Product.ID','Product.name')

English.Col.Names <- c('Week.number', 'Depot.ID', 'Channel.ID', 'Route.ID', 
                       'Client.ID', 'Product.ID', 'Sales.unit', 'Sales', 'Unit.returns', 
                       'Returns', 'Adjusted.demand')
colnames(sdata) <- English.Col.Names
sum(is.na(sdata))

sdata5 <- sdata %>% dplyr::group_by(factor(Client.ID), factor(Depot.ID), factor(Product.ID)) %>%
  dplyr::summarise(median = median(Adjusted.demand))

colnames(sdata5) <- c('Client.ID', 'Depot.ID', 'Product.ID', 'Adjusted.demand')
towndata$Depot.ID <- factor(towndata$Depot.ID)
clientdata$Client.ID <- factor(clientdata$Client.ID)
productdata$Product.ID <- factor(productdata$Product.ID)
sdata2 <- sdata5  %>% 
  dplyr::left_join(towndata, by = 'Depot.ID') %>%
  left_join(clientdata, by = 'Client.ID') %>%
  left_join(productdata, by = 'Product.ID')



# Exploratory data analysis
# May be worth showing that sales was a very good predictor of demand - highly correlated
  
# Rank demand by client
rankClient <- aggregate(sdata2$Adjusted.demand, list(sdata2$Client.name), sum, na.rm = T)
## Grab top 100 most best and worst
top_bottom_client <- sapply(c(T,F), function(TF) rankClient[order(rankClient[,2], 
                                                               decreasing = TF)[1:100],1])
colnames(top_bottom_client) <- c('Best','Worst')
head(top_bottom_client, 50)

ggplot()
# walmarts, places beginning with 'LA' rank high

# Rank demand by product
rankProduct <- aggregate(sdata2$Adjusted.demand, list(sdata2$Product.name), sum, na.rm = T)
## Grab top 100 most best and worst
top_bottom_product <- sapply(c(T,F), function(TF) rankProduct[order(rankProduct[,2], 
                                                                  decreasing = TF)[1:100],1])
colnames(top_bottom_product) <- c('Best','Worst')
head(top_bottom_product, 50)
# no clear trend

# separate words in client and count
corpus <- unlist(strsplit(unlist(sdata2$Client.name),' '))
corpusTable <- table(corpus)
rankName <- aggregate(as.data.frame(corpusTable)$Freq, list(rownames(corpusTable)), sum, na.rm = T)
top_bottom_names <- sapply(c(T,F), function(TF) rankName[order(rankName[,2], 
                                                                    decreasing = TF)[1:100],1])

colnames(top_bottom_names) <- c('Most','Least')
head(top_bottom_names, 50)

corpusNames <- rownames(corpusTable)
dumVars1 <- sdata2
dumVars1$newcolumn <- 0

# library(parallel)
# library(foreach)
# library(doParallel)
# cl <- makeCluster(detectCores() - 2)
# registerDoParallel(cl, cores = detectCores() - 2)
# 
# ptm <- proc.time()
# foreach (i = 2:length(corpusNames), .combine = rbind) %dopar% {
#   try({
#     n <-grepl(corpusNames[i], dumVars1$Client.name) 
#     dumVars1$newcolumn <- as.integer(n)
#     colnames(dumVars1)[ncol(dumVars1)] <- corpusNames[i]
#   })
# }
# proc.time() - ptm
# stopCluster(cl)
# 
# head(dumVars1)

# Remove columns not in the test dataset or redundant
sdata3 <- sdata2 %>% select(-Sales.unit,-Sales,-Unit.returns,
                            -Returns,-Client.ID, -Product.ID, -State)

# Add product weight
getWeight <- function(x){
  weightString <-regmatches(x, gregexpr("[[:digit:]]+g", x))[[1]]
  result <- ifelse(is.null(weightString), 0, 
                   as.numeric(substr(weightString,1,nchar(weightString)-1)))
  return(result)
}
sdata2$weight <- unlist(lapply(sdata2$Product.name, getWeight))
head(sdata2$weight)

# Aggregate by weight
rankweight <- aggregate(sdata2$Adjusted.demand, list(sdata2$weight), sum, na.rm = T)


# split dataset
set.seed(666)
indices <- createDataPartition(sdata3$Adjusted.demand, p= 0.8, list=F, times =1)
train_data <- sdata3[indices,]
test_data <- sdata3[-indices,]

# train_data$Product.ID <- factor(train_data$Product.ID)
# test_data$Product.ID <- factor(test_data$Product.ID)
# mydv <- dummyVars(~ factor(Channel.ID), data = train_data)
# mydv
# newTrain_data <- cbind(train_data, predict(mydv, train_data))
# newTest_data <- cbind(test_data, predict(mydv, test_data))



enetGrid <- expand.grid(.alpha = seq(0, 0.1, 0.05), #Aplha between 0 (ridge) to 1 (lasso).
                        .lambda = seq(0, 0.2, by = 0.04))

ctrl <- trainControl(method = "cv", number = 10,
                     verboseIter = T)
set.seed(1)
enetTune <- train(train_data$Adjusted.demand ~ ., data = train_data,   
                  method = "glmnet", 
                  tuneGrid = enetGrid,
                  trControl = ctrl)
enetTune
enetTune$bestTune
summary(enetTune$finalModel)
plot(enetTune)
plot(varImp(enetTune))
## RMSE worse when included product/client names etc

prediction <- round(predict(enetTune, test_data))
prediction[prediction<0] = 0
rmsle(test_data$Adjusted.demand, prediction)


stest <- read_csv('data/test.csv')
colnames(stest) <- English.Col.Names
prediction_test <- round(predict(enetTune, stest))
prediction[prediction<0] = 0
rmsle(test_data$Adjusted.demand, prediction_test)

# TODO
# Multiple cross validations to get statistics on errors 
# 
# clients that have the large sales - do they have any trend
# yes - walmarts and places with 'LA' have high demand
# separate clients to dummy variables - is.walmart 1/0
# 
# Same with product - see if any correlation with the 3 
# letter part at end of the product name, also the weight
# of the product
