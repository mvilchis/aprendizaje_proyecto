rm(list=ls())
set.seed(12345)

if(!require(caret)) install.packages("caret")
if(!require(plyr)) install.packages("plyr")
if(!require(dplyr)) install.packages("dplyr")
if(!require(AppliedPredictiveModel)) install.packages("AppliedPredictiveModeling")
if(!require(doSNOW)) install.packages("doSNOW")
if(!require(rpart)) install.packages("rpart")
if(!require(rpart.plot)) install.packages("rpart.plot")
if(!require(C50)) install.packages("C50")
if(!require(caretEnsemble)) install.packages("caretEnsemble")
if(!require(randomForest)) install.packages("randomForest")
if(!require(xgboost)) install.packages("xgboost")
if(!require(kernlab)) install.packages("kernlab")
if(!require(nnet)) install.packages("nnet")
if(!require(ipred)) install.packages("ipred")
if(!require(e1071)) install.packages("e1071")
if(!require(foreach)) install.packages("foreach")
if(!require(snn)) install.packages("snn")

library(foreach)
library(caret)
library(plyr)
library(dplyr)
library(AppliedPredictiveModeling)
library(rpart)
library(rpart.plot)
library(C50)
library(caretEnsemble)
library(randomForest)
library(xgboost)
library(kernlab)
library(nnet)
library(ipred)
library(e1071)
library(doSNOW)
library(snn)

############################################################
###--Reading data
############################################################
setwd("/home/farid/Documents/aprendizaje maquina/proyecto final")
data <- data.frame(read.csv("train.csv"))
#data <- sample_n(data, size=1000)

data$TripType = make.names(factor(data$TripType), unique=F)
data$Upc = factor(data$Upc)
data$FinelineNumber = factor(data$FinelineNumber)
names <- names(data)

############################################################
#############---Selecting "important" variables---############
############################################################
#names
# [1] "TripType"              "VisitNumber"           "Weekday"              
# [4] "Upc"                   "ScanCount"             "DepartmentDescription"
# [7] "FinelineNumber" 
data <- select(data, TripType,  Weekday, ScanCount, DepartmentDescription)


############################################################
#############---Visualizing labels---############
############################################################
pie(sort(round(table(data$TripType)*100)/length(data$TripType)), labels = names(sort(table(data$TripType))), main="Frequencies of TripType labels")

############################################################
#############---Preprocessing data---############
############################################################
pp_data <- preProcess(data, method=c("scale", "center"),  na.remove=TRUE)
pp_data

transformed <- data.frame(predict(pp_data, newdata = data))
transformed <- na.omit(transformed)
summary(transformed)

############################################################
#############---Training & Testing data---############
############################################################
trainIndex <- createDataPartition(transformed$TripType, 
                                  p=.8, 
                                  list = FALSE,
                                  time=1)

ttrain <- transformed[ trainIndex, ]
ttest  <- transformed[-trainIndex, ]

train2 <- ttrain
ttrain <- ttrain

############################################################
#############---Cross Validation---############
############################################################
validacionCruzada <- createMultiFolds(data$TripType, k=10, times=1)

############################################################
#############---Control train Function---############
############################################################
control <- trainControl(method="repeatedcv", number=10, repeats=1, 
                        index = validacionCruzada, savePredictions = T, classProbs = T) 

############################################################
#############---Initializing clusters---############
############################################################
cl <- makeCluster(4, type="SOCK")
registerDoSNOW(cl)

############################################################
#############---Make model---############
############################################################
system.time(rfcv <- train(x = select(ttrain, Weekday, ScanCount, DepartmentDescription),
                          y = ttrain$TripType, 
                          method = "C5.0",
                          importance = TRUE,
                          trControl = control))

#parRF, C5.0

############################################################
#############---Close clusters---############
############################################################
stopCluster(cl)
registerDoSEQ()
rfcv


############################################################
#############---Test values---############
############################################################
prueba <- data.frame(read.csv("test.csv"))
prueba <- select(prueba,  Weekday, ScanCount, DepartmentDescription)
prediccion <- predict(rfcv, prueba)
prediccion2 <- predict(rfcv, prueba, type="prob")
############################################################
#############---Save values---############
############################################################
prueba <- data.frame(read.csv("test.csv"))
resultado <- data.frame(TripType=prediccion, prueba)
write.csv(resultado, "pronosticosWalmart.csv", col.names=TRUE, row.names = FALSE)

############################################################
#############---Prediction of score in Kaggle---############
############################################################
testPred <- predict(rfcv, ttest, type="prob")
write.csv(testPred, "PrediccionesWalmart.csv", col.names=TRUE, row.names = FALSE)

one_hot_y <- dummies::dummy.data.frame(ttest, names = "TripType") %>% 
  select(-Weekday, -ScanCount, -DepartmentDescription)


loss_function <- function(df_real, df_pred){
  #(-1/N)(sum(sum(prob(etiqueta))))
  mult <- df_real * log(df_pred)
  suma <- mult %>% sum()
  return(-suma/nrow(df_real))
}

loss_function(one_hot_y, testPred)
