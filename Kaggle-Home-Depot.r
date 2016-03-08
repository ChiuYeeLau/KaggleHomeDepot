
library(tm)

train = read.csv("train.csv")
test = read.csv("test.csv")

train = data.frame(lapply(train, as.character), stringsAsFactors = FALSE)
test = data.frame(lapply(test, as.character), stringsAsFactors = FALSE)

set.seed(6)

head(train)

head(test)

attributes = read.csv("attributes.csv")
attributes = data.frame(lapply(attributes, as.character), stringsAsFactors = FALSE)

library(data.table)
attributes = data.table(attributes)

attributes = attributes[,.(value = paste0(value, collapse = " ")), by = .(product_uid)]

head(attributes, 1)

sampleSub = read.csv("sample_submission.csv")

productDescript = read.csv("product_descriptions.csv")

productDescript = data.frame(lapply(productDescript, as.character), stringsAsFactors = FALSE)
head(productDescript, 1)

train = merge(train, productDescript, by.x = "product_uid", by.y = "product_uid", all.x = TRUE, all.y = FALSE)
test = merge(test, productDescript, by.x = "product_uid", by.y = "product_uid", all.x = TRUE, all.y = FALSE)

train = merge(train, attributes, by.x = "product_uid", by.y = "product_uid", all.x = TRUE, all.y = FALSE)
test = merge(test, attributes, by.x = "product_uid", by.y = "product_uid", all.x = TRUE, all.y = FALSE)

stopWords = stopwords("en")

wordMatch = function(searchTerm, productTitle, desc, value){
  numProductTitle = 0
  ratioProductTitle = 0
    
  numDesc = 0
  ratioDesc = 0
    
  numValue = 0
  ratioValue = 0
    
  searchTerm = MC_tokenizer(searchTerm)
  searchTerm = searchTerm[searchTerm != ""]
  productTitle = MC_tokenizer(productTitle)
  productTitle = productTitle[productTitle != ""]
  desc = MC_tokenizer(desc)
  desc = desc[desc != ""]
  value = MC_tokenizer(value)
  value = value[value != ""]
    
  searchTerm = unlist(searchTerm)[!(unlist(searchTerm) %in% stopWords)]
  productTitle = unlist(productTitle)[!(unlist(productTitle) %in% stopWords)]
  desc = unlist(desc)[!(unlist(desc) %in% stopWords)]
  value = unlist(value)[!(unlist(value) %in% stopWords)]
    
  lenSearchTerm = length(searchTerm)
  lenProductTitle = length(productTitle)
  lenDesc = length(desc)
  lenValue = length(value)

  for(i in 1 : lenSearchTerm){
      pattern = paste("(^| )", searchTerm[i], "($| )", sep = "")
      numProductTitle = numProductTitle + grepl(pattern, paste(productTitle, collapse = " "), perl = TRUE, ignore.case = TRUE)
      numDesc = numDesc + grepl(pattern, paste(desc, collapse = " "), perl = TRUE, ignore.case = TRUE)
      numValue = numValue + grepl(pattern, paste(value, collapse = " "), perl = TRUE, ignore.case = TRUE)
  }
  ratioProductTitle = numProductTitle/lenSearchTerm
  ratioDesc = numDesc/lenSearchTerm
  ratioValue = numValue/lenSearchTerm
  return(c(lenSearchTerm, numProductTitle, ratioProductTitle, numDesc, ratioDesc, numValue, ratioValue, lenProductTitle, lenDesc, lenValue))
}

trainWords = as.data.frame(t(mapply(wordMatch, train$search_term, train$product_title, train$product_description, train$value)))
train$lenSearchTerm = trainWords[,1]
train$numProductTitle = trainWords[,2]
train$ratioProductTitle = trainWords[,3]
train$numDesc = trainWords[,4]
train$ratioDesc = trainWords[,5]
train$numValue = trainWords[,6]
train$ratioValue = trainWords[,7]
train$lenProductTitle = trainWords[,8]
train$lenDesc = trainWords[,9]
train$lenValue = trainWords[,10]

testWords = as.data.frame(t(mapply(wordMatch, test$search_term, test$product_title, test$product_description, test$value)))
test$lenSearchTerm = testWords[,1]
test$numProductTitle = testWords[,2]
test$ratioProductTitle = testWords[,3]
test$numDesc = testWords[,4]
test$ratioDesc = testWords[,5]
test$numValue = testWords[,6]
test$ratioValue = testWords[,7]
test$lenProductTitle = testWords[,8]
test$lenDesc = testWords[,9]
test$lenValue = testWords[,10]

library(xgboost)

dim(train)

dim(test)

colnames(train)

colnames(test)

head(test[,7:16], 5)

h = sample(NROW(train), 5000)

h[1:10]

dval = xgb.DMatrix(data = data.matrix(train[h, 8:17]), label = train[h, 5], missing = NaN)
dtrain = xgb.DMatrix(data = data.matrix(train[-h, 8:17]), label = train[-h, 5], missing = NaN)
watchlist = list(val = dval, train = dtrain)
dwhole = xgb.DMatrix(data = data.matrix(train[, 8:17]), label = train[,5], missing = NaN)

param <- list(  
    objective = "reg:linear",
    eval_metric = "rmse",
    eta = 0.1, 
    max_depth = 6,
    subsample = 0.7,
    colsample_bytree = 0.7
)

xgbModel = xgb.train(   
    params = param, 
    data = dtrain, 
    nrounds = 1000,
    verbose = 1,
    early.stop.round = 20,
    watchlist = watchlist,
    maximize = FALSE
)

xgbModel = xgb.train(   
    params = param, 
    data = dwhole, 
    nrounds = 86,
    verbose = 1,
    early.stop.round = 20,
    watchlist = watchlist,
    maximize = FALSE
)

colnames(test)

testRelevance = predict(xgbModel, data.matrix(test[,7:16]), missing = NaN)

testRelevance = ifelse(testRelevance > 3, 3, testRelevance)
testRelevance = ifelse(testRelevance < 1, 1, testRelevance)

testRelevance[1:20]

sampleSub$relevance = testRelevance

head(sampleSub)

write.csv(sampleSub, "xgbSubmission.csv", quote = FALSE, row.names = FALSE)

min(sampleSub$relevance)

for(i in 1:10){
    xgbModel = xgb.train(   
        params = param, 
        data = dwhole, 
        nrounds = 86,
        verbose = 1,
        early.stop.round = 20,
        watchlist = watchlist,
        maximize = FALSE
    )
    testRelevance = testRelevance + predict(xgbModel, data.matrix(test[,7:16]), missing = NaN)   
}

testRelevance = testRelevance/11.0
testRelevance = ifelse(testRelevance > 3, 3, testRelevance)
testRelevance = ifelse(testRelevance < 1, 1, testRelevance)

testRelevance[1:20]

sampleSub$relevance = testRelevance
write.csv(sampleSub, "xgbSubmission.csv", quote = FALSE, row.names = FALSE)

min(sampleSub$relevance)

max(sampleSub$relevance)

MC_tokenizer(MC_tokenizer[1])


