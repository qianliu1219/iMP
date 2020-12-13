rm(list = ls())

library(scales)
library(ggplot2)

setwd("~/Desktop/projects/iMP/riskscore/ictcf")
#setwd("~/Desktop/projects/iMP/riskscore/UCSD")
pCT <- read.csv("risk_score_pCT.txt",col.names = F)
pCT_scale <- as.data.frame(rescale(as.matrix(pCT), to = c(0, 100)))

nCT <- read.csv("risk_score_nCT.txt",col.names = F)
nCT_scale <- as.data.frame(rescale(as.matrix(nCT), to = c(0, 100)))

p<-cbind(pCT_scale,rep(1, nrow(pCT_scale)))
colnames(p)<-c("rs","label")
n<-cbind(nCT_scale,rep(0, nrow(nCT_scale)))
colnames(n)<-c("rs","label")
  
mydata<-rbind(p,n)
mydata$label <- factor(mydata$label)

set.seed(101) 
sample <- sample.int(n = nrow(mydata), size = floor(.85*nrow(mydata)), replace = F)
train <- mydata[sample, ]
test  <- mydata[-sample, ]


mylogit <- glm(label ~ rs, data = train, family = "binomial")
test$rankP<-predict(mylogit, newdata = test, type = "response")

library(pROC)
library(ROCR)
g <- roc(label ~ rankP, data = test)
plot(g)  


library(e1071) ## this is a standard R package for svm
library(ROCR)
library(SDMTools)
library(iterators)
library(glmnet)
library(pact) #cross-validation
library(ROSE)
library(rpart)


mycost<-c(6,1,10,5,5,1)
mygamma<-c(0.002,7,0.5,6,0.001)
for (i in 1:6){
  mymodel<-svm(rs~label,data=train,probability=T,kernel="radial",scale=T,shrinking = T,cost=mycost[i],gamma=mygamma[i],na.action = na.omit) # use the optimal cost
  predRes<-predict(mymodel,test,probability=T,decision.values = T,na.action = na.omit)
  mypred<-attr(predRes,"decision.values") #the column 2 is the prediction probability of label 1 (another is 0)
  pred <- prediction(mypred,test$label)
  perf <- performance(pred,"tpr","fpr")
  auc <- performance(pred,"auc")
  auc <- unlist(slot(auc, "y.values"))
  par(new=T)
  if (auc< 0.5){
    x<-1-unlist(slot(perf,"x.values"))
    y<-1-unlist(slot(perf,"y.values"))
    plot(x,y,type="l", main="ROC curve", col=i,xlab="",ylab="")
    auc2<-1-auc
  }
  
  if (auc >= 0.5){
    plot(perf, main="ROC curve", col=i) 
    auc2<-auc
  }
}


svmfit = svm(rs ~ label, data = train, kernel = "radial", cost = 5,  scale = FALSE)
print(svmfit)
plot(svmfit, train)

predRes<-predict(svmfit,train,probability=T,decision.values = T,na.action = na.omit)
mypred<-attr(predRes,"decision.values") #the column 2 is the prediction probability of label 1 (another is 0)
pred <- prediction(mypred,test$label)
perf <- performance(pred,"tpr","fpr")
auc <- performance(pred,"auc")
auc <- unlist(slot(auc, "y.values"))
par(new=T)
if (auc< 0.5){
  x<-1-unlist(slot(perf,"x.values"))
  y<-1-unlist(slot(perf,"y.values"))
  plot(x,y,type="l", main="ROC curve",xlab="",ylab="")
  auc2<-1-auc
}

if (auc >= 0.5){
  plot(perf, main="ROC curve", col=i) 
  auc2<-auc
}
#plotting the roc curve

svmmodel.confusion<-confusion.matrix(test$label,mypred)
svmmodel.accuracy<-prop.correct(svmmodel.confusion)
message("auc: ",auc2)
message("accuracy ",(svmmodel.accuracy*100))








library("ROCR")    
pred <- prediction(test$rankP, test$label)    
perf <- performance(pred, measure = "tpr", x.measure = "fpr")     
plot(perf, col=rainbow(7), main="ROC curve Admissions", xlab="Specificity", 
     ylab="Sensitivity")    
abline(0, 1) #add a 45 degree line
auc = performance(pred, "auc")



perf <- performance(pred, measure="acc", x.measure="cutoff")
bestAccInd <- which.max(perf@"y.values"[[1]])
bestMsg <- paste("best accuracy=", perf@"y.values"[[1]][bestAccInd], 
                 " at cutoff=", round(perf@"x.values"[[1]][bestAccInd], 4))

plot(perf, sub=bestMsg)

perf <- performance(pred, measure="f")
bestAccInd <- which.max(perf@"y.values"[[1]])
bestMsg <- paste("best F1=", perf@"y.values"[[1]][bestAccInd], 
                 " at cutoff=", round(perf@"x.values"[[1]][bestAccInd], 4))
plot(perf, sub=bestMsg)

library(caret)
precision <- posPredValue(as.factor(ifelse(test$rankP>= 0.3,1,0)), as.factor(test$label), positive="1")
recall <- sensitivity(as.factor(ifelse(test$rankP>= 0.339,1,0)), as.factor(test$label), positive="1")

F1 <- (2 * precision * recall) / (precision + recall)

t.test(pCT_scale,nCT_scale)




ggplot(nCT_scale) + geom_density(aes(x = FALSE.,fill="red"), alpha = 0.5) + 
  geom_density(data = pCT_scale,aes(x = FALSE.,fill="blue"), alpha = 0.5)
