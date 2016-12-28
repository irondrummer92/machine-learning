# Code to solve the Bigmart sales problem by analytics vidhya
setwd("D:/ML/BigMart Sales/")

# Set libraries
library(dplyr)
library(randomForest)

# Reading and checking input dataset
trainRaw <- read.csv("Train_UWu5bXk.csv", stringsAsFactors = F,na.strings = c(" ",""))
sapply(trainRaw, FUN = function(x){return(length(unique(x)))})

##---------------------------------------------------------------------------------------------
# Item_Identifier               Item_Weight          Item_Fat_Content           Item_Visibility 
# 1559                       416                         5                      7880 
# Item_Type                  Item_MRP         Outlet_Identifier Outlet_Establishment_Year 
# 16                      5938                        10                         9 
# Outlet_Size      Outlet_Location_Type               Outlet_Type         Item_Outlet_Sales 
# 4                         3                         4                      3493 
##---------------------------------------------------------------------------------------------

sapply(trainRaw, FUN = function(x){return(sum(is.na(x)))})

##---------------------------------------------------------------------------------------------
# Item_Identifier               Item_Weight          Item_Fat_Content           Item_Visibility 
# 0                      1463                         0                         0 
# Item_Type                  Item_MRP         Outlet_Identifier Outlet_Establishment_Year 
# 0                         0                         0                         0 
# Outlet_Size      Outlet_Location_Type               Outlet_Type         Item_Outlet_Sales 
# 2410                         0                         0                         0 
##---------------------------------------------------------------------------------------------

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

# Impute weights for products
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

# Check for item fat content
table(trainSizesimputed$Item_Fat_Content)
trainFatClean <- trainSizesimputed
trainFatClean$ItemFatCleaned <- trainFatClean$Item_Fat_Content
trainFatClean$ItemFatCleaned[trainFatClean$ItemFatCleaned == "reg"] = "Regular"
trainFatClean$ItemFatCleaned[trainFatClean$ItemFatCleaned == "low fat"] = "Low Fat"
trainFatClean$ItemFatCleaned[trainFatClean$ItemFatCleaned == "LF"] = "Low Fat"
table(trainFatClean$ItemFatCleaned)

# Final training set
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

trainBackup <- dataTrain

# Conversion of categorical variables to factors
dataTrain$ItemFatCleaned <- as.factor(dataTrain$ItemFatCleaned)
dataTrain$Item_Type <- as.factor(dataTrain$Item_Type)
dataTrain$OutletSizeImputed <- as.factor(dataTrain$OutletSizeImputed)
dataTrain$Outlet_Location_Type <- as.factor(dataTrain$Outlet_Location_Type)
dataTrain$Outlet_Establishment_Year <- as.factor(dataTrain$Outlet_Establishment_Year)

dataTest <- trainFinal[tst,c("ItemWeightImputed",
                             "ItemFatCleaned",
                             "Item_Visibility",
                             "Item_Type",
                             "Item_MRP",
                             "OutletSizeImputed",
                             "Outlet_Establishment_Year",
                             "Outlet_Location_Type",
                             "Item_Outlet_Sales")]

testBackup <- dataTest

# Conversion of categorical variables to factors
dataTest$ItemFatCleaned <- as.factor(dataTest$ItemFatCleaned)
dataTest$Item_Type <- as.factor(dataTest$Item_Type)
dataTest$OutletSizeImputed <- as.factor(dataTest$OutletSizeImputed)
dataTest$Outlet_Location_Type <- as.factor(dataTest$Outlet_Location_Type)
dataTest$Outlet_Establishment_Year <- as.factor(dataTest$Outlet_Establishment_Year)

# Model using randomForests algorithm to predict sales of an item
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
# RMSE
sqrt(mean(dataValidation$deviation ^ 2))

# RMSE - 1141 on test dataset. Might be good if modelled on entire dataset
