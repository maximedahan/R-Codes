# DEFINE INPUT AND OUTPUT FOLDERS  (train + test)
setwd("C:/.")
output="C:/..."

library(xgboost)
library(caret)
library(caretEnsemble)

sub<-read.table("sample_submission.csv",sep=',',header=T)
test<-read.table("test.csv",sep=',',header=T)
train<-read.table("train.csv",sep=',',header=T)

train$id<-NULL

# XGBOOST PARAMETERS
xgb_grid_1 = expand.grid(
  nrounds = 100,
  eta = c(0.01,0.05,0.1),
  max_depth = c(8,10,14,20),
  gamma = 0,
  colsample_bytree=c(0.8,0.9,1),
  min_child_weight = 1,
  subsample=c(0.8,1)
)

# pack the training control parameters
xgb_trcontrol_1 = trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  returnData = FALSE,
  returnResamp = "all",                                  # save losses across all models
  classProbs = TRUE,                                     # set to TRUE for AUC to be computed
  summaryFunction = twoClassSummary,
  allowParallel = TRUE
)

# DEFINE MODEL
train$target<-ifelse(train$target==1,'event','no event')
formula="..."

xgb_train_1 = train(x = formula,
                    y = train$target,
                    trControl = xgb_trcontrol_1,
                    tuneGrid = xgb_grid_1,
                    nthread=4,
                    method = "xgbTree",
                    metric='ROC'
)



#-------------------

xgb <- xgboost(data = data.matrix(train[,-1]), 
               label = train$target, 
               eta = 0.1,
               max_depth = 15, 
               nround=25, 
               subsample = 0.5,
               colsample_bytree = 0.5,
               seed = 1,
               eval_metric = "error",
               objective = "binary:logistic",
               num_class = 12,
               nthread = 3,
               missing = ''
)

#-------------------

sub$target<-predict(xgb,test)
min(sub$target)
max(sub$target)

write.table(sub,file= output,quote=F,sep=",",row.names = F)
