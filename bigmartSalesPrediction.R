setwd("D:/ML/BigMart Sales/")

library(dplyr)

trainRaw <- read.csv("Train_UWu5bXk.csv", stringsAsFactors = F,na.strings = c(" ",""))

sapply(trainRaw, FUN = function(x){return(length(unique(x)))})

# Item_Identifier               Item_Weight          Item_Fat_Content           Item_Visibility 
# 1559                       416                         5                      7880 
# Item_Type                  Item_MRP         Outlet_Identifier Outlet_Establishment_Year 
# 16                      5938                        10                         9 
# Outlet_Size      Outlet_Location_Type               Outlet_Type         Item_Outlet_Sales 
# 4                         3                         4                      3493 


sapply(trainRaw, FUN = function(x){return(sum(is.na(x)))})

# Item_Identifier               Item_Weight          Item_Fat_Content           Item_Visibility 
# 0                      1463                         0                         0 
# Item_Type                  Item_MRP         Outlet_Identifier Outlet_Establishment_Year 
# 0                         0                         0                         0 
# Outlet_Size      Outlet_Location_Type               Outlet_Type         Item_Outlet_Sales 
# 2410                         0                         0                         0 

itemWeights <- trainRaw %>% select(Item_Identifier, Item_Weight) %>% distinct()
sum(is.na(itemWeights$Item_Weight))
# 1142

trainNoNA <- itemWeights[which(!is.na(itemWeights$Item_Weight)),]
itemWeightsMissing <- merge(itemWeights,trainNoNA,by = "Item_Identifier",all.x = T)

# Find out where both values are missing. Such values can be imputed as mean
missingCounts <- itemWeightsMissing %>% 
  select(Item_Weight.x,Item_Weight.y) %>% 
  transform(present_x = is.na(Item_Weight.x)) %>%
  transform(present_y = is.na(Item_Weight.y)) %>%
  group_by(present_x,present_y) %>%
  summarize(counts = n())

itemWeightsMissing$ItemWeightImputed <- ifelse(is.na(itemWeightsMissing$Item_Weight.x),
                                               yes = itemWeightsMissing$Item_Weight.y,
                                               no = itemWeightsMissing$Item_Weight.x)

itemWeightsMissing$ItemWeightImputed[is.na(itemWeightsMissing$ItemWeightImputed)] <- 
  mean(itemWeightsMissing$ItemWeightImputed,na.rm = T)

itemWeightsMissing <- unique(itemWeightsMissing[,c("Item_Identifier", "ItemWeightImputed")])

trainWeightsImputed <- merge(trainRaw,y = itemWeightsMissing,by = "Item_Identifier")

# Treating outlet size column
# How do outlet sizes compare to outlet type
table(trainWeightsImputed$Outlet_Type,
      trainWeightsImputed$Outlet_Size,useNA = "ifany")
table(trainWeightsImputed$Outlet_Identifier,
      trainWeightsImputed$Outlet_Size,useNA = "ifany")
table(trainWeightsImputed$Outlet_Identifier,
      trainWeightsImputed$Outlet_Type,useNA = "ifany")
table(trainWeightsImputed$Outlet_Identifier,
      trainWeightsImputed$Outlet_Location_Type,useNA = "ifany")
table(trainWeightsImputed$Outlet_Location_Type,
      trainWeightsImputed$Outlet_Size,useNA = "ifany")

# OUT10 = SMALL
# OUT17 & OUT45 = SMALL
# All missing outlet sizes replaced by "Small"
trainSizesimputed <- trainWeightsImputed
trainSizesimputed$OutletSizeImputed <- ifelse(is.na(trainSizesimputed$Outlet_Size), 
                                              "Small",
                                              trainSizesimputed$Outlet_Size)

table(trainSizesimputed$Item_Fat_Content)
trainFatClean <- trainSizesimputed
trainFatClean$ItemFatCleaned <- trainFatClean$Item_Fat_Content
trainFatClean$ItemFatCleaned[trainFatClean$ItemFatCleaned == "reg"] = "Regular"
trainFatClean$ItemFatCleaned[trainFatClean$ItemFatCleaned == "low fat"] = "Low Fat"
trainFatClean$ItemFatCleaned[trainFatClean$ItemFatCleaned == "LF"] = "Low Fat"
table(trainFatClean$ItemFatCleaned)

trainFinal <- trainFatClean

# Creating the Training and Validation datasets
allRow <- 1:nrow(trainFinal)
trn <- sample(x = allRow,size = 0.8 * nrow(trainFinal))
tst <- allRow[!(allRow %in% trn)]
colnames(trainFinal)
dataTrain <- trainFinal[trn,c("ItemWeightImputed",
                              "ItemFatCleaned",
                              "Item_Visibility",
                              "Item_Type",
                              "Item_MRP",
                              "OutletSizeImputed",
                              "Outlet_Establishment_Year",
                              "Outlet_Location_Type",
                              "Item_Outlet_Sales")]

dataTrain$ItemWeightImputed <- (dataTrain$ItemWeightImputed - mean(dataTrain$ItemWeightImputed))/sd(dataTrain$ItemWeightImputed)
dataTrain$Item_Visibility <- (dataTrain$Item_Visibility - mean(dataTrain$Item_Visibility))/sd(dataTrain$Item_Visibility)
dataTrain$Item_MRP <- (dataTrain$Item_MRP - mean(dataTrain$Item_MRP))/sd(dataTrain$Item_MRP)
# dataTrain$Outlet_Establishment_Year <- (dataTrain$Outlet_Establishment_Year - mean(dataTrain$Outlet_Establishment_Year))/sd(dataTrain$Outlet_Establishment_Year)
maxSales <- max(dataTrain$Item_Outlet_Sales)
minSales <- min(dataTrain$Item_Outlet_Sales)
dataTrain$Item_Outlet_Sales <- (dataTrain$Item_Outlet_Sales - minSales)/(maxSales - minSales)


dataTrain$ItemFatCleaned <- as.factor(dataTrain$ItemFatCleaned)
dataTrain$Item_Type <- as.factor(dataTrain$Item_Type)
dataTrain$OutletSizeImputed <- as.factor(dataTrain$OutletSizeImputed)
dataTrain$Outlet_Location_Type <- as.factor(dataTrain$Outlet_Location_Type)


dataTest <- trainFinal[tst,c("ItemWeightImputed",
                             "ItemFatCleaned",
                             "Item_Visibility",
                             "Item_Type",
                             "Item_MRP",
                             "OutletSizeImputed",
                             "Outlet_Establishment_Year",
                             "Outlet_Location_Type",
                             "Item_Outlet_Sales")]

dataTest$ItemWeightImputed <- (dataTest$ItemWeightImputed - mean(dataTest$ItemWeightImputed))/sd(dataTest$ItemWeightImputed)
dataTest$Item_Visibility <- (dataTest$Item_Visibility - mean(dataTest$Item_Visibility))/sd(dataTest$Item_Visibility)
dataTest$Item_MRP <- (dataTest$Item_MRP - mean(dataTest$Item_MRP))/sd(dataTest$Item_MRP)
dataTest$Outlet_Establishment_Year <- (dataTest$Outlet_Establishment_Year - mean(dataTest$Outlet_Establishment_Year))/sd(dataTest$Outlet_Establishment_Year)


dataTest$ItemFatCleaned <- as.factor(dataTest$ItemFatCleaned)
dataTest$Item_Type <- as.factor(dataTest$Item_Type)
dataTest$OutletSizeImputed <- as.factor(dataTest$OutletSizeImputed)
dataTest$Outlet_Location_Type <- as.factor(dataTest$Outlet_Location_Type)

# Model using randomForests algorithm to predict sales of an item
library(randomForest)

randomModel <- randomForest(formula = Item_Outlet_Sales ~ ., data = dataTrain)

dataValidation <- dataTest
dataValidation$predictedSales <- predict(object = randomModel,newdata = dataTest)

# Computing metrics of Fit
dataValidation$deviation <- abs(dataValidation$Item_Outlet_Sales - dataValidation$predictedSales)
dataValidation$percentageDeviation <- dataValidation$deviation/dataValidation$Item_Outlet_Sales

# Mean Absolute Deviation
mean(dataValidation$deviation)
# %MAPE
mean(dataValidation$percentageDeviation)


library(neuralnet)

modelTrain <- model.matrix(  ~ Item_Outlet_Sales + ItemWeightImputed + ItemFatCleaned + Item_Visibility + 
    Item_Type + Item_MRP + OutletSizeImputed + Outlet_Establishment_Year + 
    Outlet_Location_Type, data = dataTrain)

dataNames <- colnames(modelTrain)
dataNames <- gsub(dataNames,pattern = " ",replacement = "")
nnFormula <- as.formula(paste("Item_Outlet_Sales ~ ", 
                              paste(dataNames[!dataNames %in% c("Item_Outlet_Sales","(Intercept)")],
                                    collapse = "+")))
colnames(modelTrain) <- dataNames

nnModel <- neuralnet(data = modelTrain,
                     formula = nnFormula, 
                     hidden = 3,
                     learningrate = 0.00001,
                     algorithm = "backprop",
                     act.fct = "logistic")

plot(nnModel)

dataValidation <- dataTest
modelTest <- model.matrix(  ~ ItemWeightImputed + ItemFatCleaned + Item_Visibility + 
                              Item_Type + Item_MRP + OutletSizeImputed + Outlet_Establishment_Year + 
                              Outlet_Location_Type, data = dataTest)

modelTest <- modelTest[,-1]

modelComputation <- compute(x = nnModel,covariate = modelTest)

# Computing metrics of Fit
dataValidation$deviation <- abs(dataValidation$Item_Outlet_Sales - dataValidation$predictedSales)
dataValidation$percentageDeviation <- dataValidation$deviation/dataValidation$Item_Outlet_Sales
