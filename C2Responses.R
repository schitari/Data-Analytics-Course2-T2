install.packages("e1071")
library("caret")
library("corrplot")
library(e1071)
library(gbm)

library("readr")
#Read Data csv file into Data frame
CompleteResponses <- read.csv("CompleteResponses.csv")


#Find correlation in the columns
source("http://www.sthda.com/upload/rquery_cormat.r")
rquery.cormat(CompleteResponses, type="full")

##Pre processing and feature engineering
#make the categorical fields into factors
CompleteResponses$elevel = as.factor(CompleteResponses$elevel)
CompleteResponses$brand = as.factor(CompleteResponses$brand)
CompleteResponses$car = as.factor(CompleteResponses$car)
CompleteResponses$zipcode = as.factor(CompleteResponses$zipcode)
str(CompleteResponses)
#discretize age by frequency. Check histogram distribution to check if has normal distribution
hist(CompleteResponses$age)
#cut the age column into bins
CompleteResponses$ageBins = cut(CompleteResponses$age, c(19,30,40,50,60,70,80))
##Split data for training and testing
respIndices = createDataPartition(CompleteResponses$brand, p=0.75,list = FALSE)
resp_75_train = CompleteResponses[respIndices, c("salary", "elevel","car", "zipcode", "credit","brand","ageBins")]
resp_25_test =  CompleteResponses[-respIndices, c("salary", "elevel","car", "zipcode", "credit","brand","ageBins")]

#apply 10 fold cross validation
fitcontrol = trainControl(method ="repeatedcv", number = 10, repeats = 1)

#train c5.0
c50train1 = train(brand~., data = resp_75_train, method ="C5.0", trControl = fitcontrol )
c50train1
plot(varImp(c50train1))
#train GBM
system.time(gbmTrain1 <- train(brand~., data = resp_75_train, method ="gbm", trControl = fitcontrol, verbose = FALSE ))
gbmTrain1
plot(gbmTrain1)

#train Random forest
rfGrid = expand.grid(mtry = 4)
system.time(rfTrain1 <- train(brand~., data = resp_75_train, method ="rf", trControl = fitcontrol))
rfTrain1
plot(rfTrain1)
plot(varImp(rfTrain1))

#train LogitBoost
system.time(logTrain1 <- train(brand~., data = resp_75_train, method ="LogitBoost", trControl = fitcontrol))
logTrain1
plot(logTrain1)

#predict using C5.0
brandPredictions = predict(c50train1,resp_25_test, type = "raw")
brandPredictions
confusionMatrix(resp_25_test$brand,brandPredictions)

#predict using random forest
brandPredictionsRF = predict(rfTrain1,resp_25_test, type = "raw")
brandPredictionsRF
confusionMatrix(resp_25_test$brand,brandPredictionsRF)
qplot(brandPredictionsRF,resp_25_test$brand, geom = "jitter")
postResample(brandPredictionsRF,resp_25_test$brand)

#Step 3 of plan of attack to test the random forest model for 5 different mtry values
#tunelength = 5 10 fold trainControl
rfGrid = expand.grid(mtry = 5)
rfTrain5 = train(brand~.,data = resp_75_train, method ='rf', trControl= fitcontrol, tuneGrid = rfGrid)
rfTrain5
plot(varImp(rfTrain5))

#tunelength = 10 10 fold trainControl
rfGrid = expand.grid(mtry = 10)
rfTrain10 = train(brand~.,data = resp_75_train, method ='rf', trControl= fitcontrol, tuneGrid = rfGrid)
plot(varImp(rfTrain10))

#tunelength = 20 10 fold trainControl
rfGrid = expand.grid(mtry = 20)
rfTrain20 = train(brand~.,data = resp_75_train, method ='rf', trControl= fitcontrol, tuneGrid = rfGrid)
plot(varImp(rfTrain20))

#tunelength = 25 10 fold trainControl
rfGrid = expand.grid(mtry = 25)
rfTrain25 = train(brand~.,data = resp_75_train, method ='rf', trControl= fitcontrol, tuneGrid = rfGrid)
plot(varImp(rfTrain25))

#tunelength = 30 10 fold trainControl
rfGrid = expand.grid(mtry = 30)
rfTrain30 = train(brand~.,data = resp_75_train, method ='rf', trControl= fitcontrol, tuneGrid = rfGrid)
plot(varImp(rfTrain30))

#predict using random forest and mtry = 10
brandPredictionsRF = predict(rfTrain10,resp_25_test, type = "raw")
brandPredictionsRF
confusionMatrix(resp_25_test$brand,brandPredictionsRF)
qplot(brandPredictionsRF,resp_25_test$brand, geom = "jitter")
postResample(brandPredictionsRF,resp_25_test$brand)

#predict using random forest and mtry = 25
brandPredictionsRF = predict(rfTrain25,resp_25_test, type = "raw")
brandPredictionsRF
confusionMatrix(resp_25_test$brand,brandPredictionsRF)
qplot(brandPredictionsRF,resp_25_test$brand, geom = "jitter")
postResample(brandPredictionsRF,resp_25_test$brand)

# read SurveyIncomplete csv file
SurveyIncomplete <- read_csv("SurveyIncomplete.csv")
##Pre processing and feature engineering
#make the categorical fields into factors
SurveyIncomplete$elevel = as.factor(SurveyIncomplete$elevel)
SurveyIncomplete$brand = as.factor(SurveyIncomplete$brand)
SurveyIncomplete$car = as.factor(SurveyIncomplete$car)
SurveyIncomplete$zipcode = as.factor(SurveyIncomplete$zipcode)
str(SurveyIncomplete)
#discretize age by frequency. Check histogram distribution to check if has normal distribution
hist(SurveyIncomplete$age)
#cut the age column into bins
SurveyIncomplete$ageBins = cut(SurveyIncomplete$age, c(19,30,40,50,60,70,80))

#predict using random forest and mtry = 10
brandPredictionsRF = predict(rfTrain10,SurveyIncomplete, type = "raw")
brandPredictionsRF
confusionMatrix(SurveyIncomplete$brand,brandPredictionsRF)
qplot(brandPredictionsRF,SurveyIncomplete$brand, geom = "jitter")
postResample(brandPredictionsRF,SurveyIncomplete$brand)

#export to CSV
output = cbind(SurveyIncomplete, brandPredictionsRF, deparse.level = 1)
write.csv(output,"SurveyIncompletePredictions.csv")

#VarImpPlot will show the importance of attributes used
library(randomForest)
varImpPlot(rfTrain10)
class(rfTrain10)

#Visualizations
library(ggplot2)
ggplot(subset(SurveyIncompletePredictions, ageBins %in% c("(60,70]","(70,80]")), aes(x=salary, y=brandPredictionsRF, color=ageBins)) + geom_point(position=position_jitter(height=0.05, width=0.05))+ geom_smooth(method="auto")
ggplot(subset(SurveyIncompletePredictions, ageBins %in% c("(19,30]","(30,40]","(40,50]","(50,60]")), aes(x=salary, y=brandPredictionsRF, color=ageBins)) + geom_point(position=position_jitter(height=0.05, width=0.05))+ geom_smooth(method="auto")
ggplot(varImp(rfTrain10))

