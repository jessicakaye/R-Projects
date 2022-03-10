# First we load the packages
library(tidyverse)
library(caret)
library(glmnet)
library(dplyr)
library(car)
library(ggplot2)
library(GGally)
library(car)
library(MASS)
library(olsrr)
library(forecast)
library(h2o)
library(summarytools)
library(broom)

# We import in a function that we can utilize to get robust standard errors
library(RCurl)
# import the function
url_robust <- "https://raw.githubusercontent.com/IsidoreBeautrelet/economictheoryblog/master/robust_summary.R"
eval(parse(text = getURL(url_robust, ssl.verifypeer = FALSE)),
     envir=.GlobalEnv)

setwd("~/Documents/FALL 2020/ECON 511A/Final Project")

# Now we load the dataset!
cancer <- read.csv("cancer_reg.csv", fileEncoding = "latin1")
# cancer
# attributes(cancer)

# DEATH_RATE Histogram overlaid with kernel density curve
# ggplot(cancer, aes(x=TARGET_deathRate)) + 
#   ggtitle("Distribution of TARGET_deathRate variable") +
#   geom_histogram(aes(y=..density..),      # Histogram with density instead of count on y-axis
#                  binwidth=.5,
#                  colour="black", fill="white") +
#   geom_density(alpha=.2, fill="#FF6666") +  # Overlay with transparent density plot
#   geom_vline(aes(xintercept=mean(TARGET_deathRate, na.rm=T)),   # Ignore NA values for mean
#            color="red", linetype="dashed", size=1)

# Determine the structure of each column
str(cancer)

#Let's divide out the geography column into two columns:
cancer <- separate(cancer,"Geography",into = c("County","State"),sep = ",") # split column on comma into two

# In this case, we need to change this binned income into a factor
# cancer$binnedInc <- factor(cancer$binnedInc)
cancer$State <- factor(cancer$State)
ordered(cancer$binnedInc, levels =  c('[22640, 34218.1]', '(34218.1, 37413.8]', '(37413.8, 40362.7]', '(40362.7, 42724.4]', 
'(42724.4, 45201]', '(45201, 48021.6]', '(48021.6, 51046.4]',  '(51046.4, 54545.6]',  '(54545.6, 61494.5]', '(61494.5, 125635]')
)

# Descriptive Statistics
summary(cancer)
descr(cancer, stats = "common")

# How many NULL/incomplete values exist?
colSums(is.na(cancer))

# Remove the column with more than 30% NA
complete.cancer <- cancer[,-which(colMeans(is.na(cancer)) > 0.3)]

# Remove the regressors with null values
complete.cancer <- na.omit(complete.cancer)
summary(complete.cancer)
str(complete.cancer)

# Remove the regressors with obvious outliers (MedianAge)
complete.cancer <- subset(complete.cancer, select = -c(MedianAge))

# We want to standardize the dataset prior to conducting the regression analysis (if not already standardized)
std_cancer <- subset(complete.cancer, select = -c(TARGET_deathRate)) %>% mutate_if(is.numeric, scale)
TARGET_deathRate <- complete.cancer[,'TARGET_deathRate']
std_cancer <- cbind(TARGET_deathRate, std_cancer)
summary(std_cancer)
str(std_cancer)

# We are going to remove the Geography and binnedInc columns for the correlation matrix
# crows <- c("County", "City", "binnedInc","avgAnnCount", "avgDeathsPerYear","incidenceRate","studyPerCap")

# What does our correlation matrix look like for this df?
# cor(std_cancer[, !names(std_cancer) %in% crows], use = "complete.obs")
# library(corrplot)
# corrplot(cor(std_cancer[, !names(std_cancer) %in% crows], use = "complete.obs"), na.label = "?")

# Scatterplots to check distribution and correlation
# library(faraway)
# pairs(std_cancer[TARGET_deathRate, !names(std_cancer) %in% crows], col = "dodgerblue")
# graphics.off()
# par("mar")
# par(mar=c(1,1,1,1))

# It seems there are some regressors that are looking correlated here.

# We want to remove the predictors that obviously will not lead to causal inference.
# Will not use the following:
# avgAnnCount: Mean number of reported cases of cancer diagnosed annually
# avgDeathsPerYear: Mean number of reported mortalities due to cancer
# incidenceRate: Mean per capita (100,000) cancer diagoses
# studyPerCap: Per capita number of cancer-related clinical trials per county
# TARGET_deathRate: Mean per capita (100,000) cancer mortalities
# medIncome is the same as binnedInc, decided to use the bins instead
notpred <- c("County", "medIncome","avgAnnCount", "avgDeathsPerYear","incidenceRate","studyPerCap")
#for now we are not using State, but I would like to use it later!

fin_cancer <- std_cancer[,!names(std_cancer) %in% notpred]

summary(fin_cancer)
str(fin_cancer)

# FIRST MODEL: OLS MULTIPLE LINEAR REGRESSION FOR PREDICTION

# Split the data into training and test set
set.seed(123)
training.samples <- std_cancer$TARGET_deathRate %>%
  createDataPartition(p = 0.7, list = FALSE)
train.data  <- fin_cancer[training.samples, ]
test.data <- fin_cancer[-training.samples,]

# Build the model
OLSmodel <- lm(TARGET_deathRate ~., data = train.data)

# Make predictions
OLS.test.pred <- OLSmodel %>% predict(test.data)

# What does the linear model look like?
# We will use robust standard errors. From what I have seen in the later areas (when checking asssumptions),
# I know that some heteroscedasticity exists.
summary(OLSmodel, robust=T)
sqrt(mean(residuals(OLSmodel)^2))
# confint(OLSmodel, level=0.95)

# Model performance
eval_metrics = function(model, df, predictions, target){
  resids = df[,target] - predictions
  resids2 = resids**2
  N = length(predictions)
  k = length(OLSmodel$coefficients) + 1
  r2 = R2(predictions, test.data$TARGET_deathRate, na.rm=TRUE)
  adj_r2 = r2*(N-1)/(N-k)
  print(paste0("R^2: ", as.numeric(r2,2)))
  print(paste0("Adj. R^2: ", as.numeric(adj_r2,2))) #Adjusted R-squared
  #print(as.character(round(sqrt(sum(resids2)/N), 2))) #RMSE
  print(paste0("RMSE: ", RMSE(predictions, test.data$TARGET_deathRate, na.rm=TRUE)))
}
# Here are the metrics for our test data
ans <- eval_metrics(OLSmodel,test.data, OLS.test.pred, target = 'TARGET_deathRate')


k <- length(OLSmodel$coefficients) + 1
N <- nrow(train.data)

# Let's check some of the assumptions of our model!

# # QUICK CHECK
# # 1: Are the relationships between your X predictors and Y roughly linear? Rejection of the null (p < .05) indicates a non-linear relationship between one or more of your X’s and Y.
# # 2: Is your distribution skewed positively or negatively, necessitating a transformation to meet the assumption of normality? Rejection of the null (p < .05) indicates that you should likely transform your data.
# # 3: Is your distribution kurtotic (highly peaked or very shallowly peaked), necessitating a transformation to meet the assumption of normality? Rejection of the null (p < .05) indicates that you should likely transform your data.
# # 4: Is your dependent variable truly continuous, or categorical? Rejection of the null (p < .05) indicates that you should use an alternative form of the generalized linear model (e.g. logistic or binomial regression).
# # 5: Is the variance of your model residuals constant across the range of X (assumption of homoscedastiity)? Rejection of the null (p < .05) indicates that your residuals are heteroscedastic, and thus non-constant across the range of X. Your model is better/worse at predicting for certain ranges of your X scales.
# library(gvlma)
# gvlma(OLSmodel)

# DETAILED CHECK

#A1: E(epsilon|X) = 0 OR E(epsilon) = 0 AND correlation between residuals & X = 0
mean(OLSmodel$residuals)

# There exists a linear relationship between all X's and Y
# Want a straight line and random looking scatter plot. 
# This tells us how the mean of the residuals change for the predictors (Correlation between residuals and X)
plot(OLSmodel, 1)

      # Homoscedasticity - It's okay if it is not homoescedastic. versus the expectation of epsilon, we look at the variance of epsilon
      # In this case, you want to see a horizontal line. This tells us we want to see robust standard errors.
      plot(OLSmodel, 3)
      spreadLevelPlot(OLSmodel)

      # Normality of residuals - not too necessary either
      qqPlot(OLSmodel, main="QQ Plot")
      # Some points seem to be falling off outside the 95% confidence interval... (dotted line)
      checkresiduals(OLSmodel)

# A2: error terms are uncorrelated, IID
# It's okay to just make this assumption

# A3: There is no perfect multicollinearity
# Let's use the Variance Inflation Factor (VIF). The VIF of a predictor is a measure for how easily it is predicted from a linear regression using the other predictors.
# A general guideline is that a VIF larger than 5 or 10 is large, indicating that the model has problems estimating the coefficient.
# Overall, if the VIF is larger than 1/(1-R2), where R2 is the Multiple R-squared of the regression, then that predictor is more related to the other predictors than it is to the response.
1/(1-summary(OLSmodel)$adj.r.squared)
vif(OLSmodel)
ols_vif_tol(OLSmodel)
# There are some very high values here... This tells me ridge regression or PCR might be better. You can also just drop the variables that have a high VIF

# A4: Large outliers are unlikely (see upper right or lower right corners)
# Outliers are possibly values with standardized residuals > |3| 
# High leverage means greater than 2(k + 1)/n
# Leverage: How unusual is the observation in terms of its values on the independent predictors?
2*(k + 1)/N
plot(OLSmodel, 5)
# Influence: Inclusion or exclusion can alter the results of the regression analysis
# If value is > 4/(n - k - 1), then that value is associated with a lare residual and can thus be influential
cutoff = 4/(N - k - 1)
cutoff
plot(OLSmodel, 4)
# influence.measures(OLSmodel)
# It seems there is possiblity for outliers here


# SECOND MODEL: LASSO (L1) REGRESSION FOR PREDICTION
# this is used to select the best lambda value
lambda_seq <- 10^seq(2, -3, by = -.1)

# define our input parameters
x.train <- model.matrix(TARGET_deathRate ~ ., train.data)
y.train <- train.data$TARGET_deathRate
x.test <- model.matrix(TARGET_deathRate ~ ., test.data)
y.test <- test.data$TARGET_deathRate

# initial run-through to utilize cross-validation
output <- cv.glmnet(x.train, y.train, alpha = 1, lambda = lambda_seq, nfolds = 5)
plot(output)

best.lambda <- output$lambda.min
best.lambda

# using the best.lambda, we will train the lasso model again!
lasso_best <- glmnet(x.train, y.train, alpha = 1, lambda = best.lambda)
lasso.train.pred <- predict(lasso_best, s=best.lambda, newx=x.train)
lasso.test.pred <- predict(lasso_best, s=best.lambda, newx=x.test)

# let's evaluate the model
# Here we define a function that will calculate the values we are looking for!
# Compute R^2 from true and predicted values
eval_results <- function(true, predicted, df) {
  SSE <- sum((predicted - true)^2)
  SST <- sum((true - mean(true))^2)
  R_square <- 1 - SSE / SST
  N = length(predicted)
  k = length(coef(lasso_best)) + 1
  adjr_square <- R_square * (N-1)/(N-k)
  RMSE = sqrt(SSE/nrow(df))
  
  
  # Model performance metrics
  data.frame(
    RMSE = RMSE,
    Rsquare = R_square,
    Adj_Rsquare = adjr_square
  )
  
}

eval_results(y.train, lasso.train.pred, train.data)
eval_results(y.test, lasso.test.pred, test.data)

# Let's take a look at the lasso estimated coefficients
coef(lasso_best)
# 
# # We find that ~ 6 variables have been set completely to 0. There also appear to be some values that are quite close to 0.
# lasso_vars <- c('PctMarriedHouseholds',
#                 'PctBachDeg25_Over',
#                 'PctOtherRace',
#                 'PctEmployed16_Over',
#                 'MedianAgeMale',
#                 'BirthRate',
#                 'PctNoHS18_24',
#                 'MedianAgeFemale',
#                 'PctPrivateCoverage',
#                 'PctWhite',
#                 'PctPrivateCoverageAlone',
#                 'PctBachDeg18_24',
#                 'State',
#                 'PctBlack',
#                 'PctUnemployed16_Over',
#                 'PctHS18_24',
#                 'PctPublicCoverageAlone',
#                 'PctHS25_Over',
#                 'PctEmpPrivCoverage',
#                 'PercentMarried')
# 
# lasso_cancer <- std_cancer[,names(std_cancer) %in% lasso_vars]
# 
# # Build the model to check multicollinearity using LASSO variables
# lassomodel <- lm(TARGET_deathRate ~., data = lasso_cancer)
# # Let's see the coefficients
# summary(lassomodel, robust = T)
# 
# #Let's check VIF
# 1/(1-summary(OLSmodel)$adj.r.squared)
# ols_vif_tol(lassomodel)
# # there is still multicollinearity...

#FIRST MODEL: REFINEMENT!!
# We are going to take the first model we had and manually remove all of the regressors with high VIF
highvif_vars <- c('PercentMarried',
                  'povertyPercent',
                  'MedianAgeMale',
                  'MedianAgeFemale',
                  'PctPrivateCoverage',
                  'PctPrivateCoverageAlone',
                  'PctEmpPrivCoverage',
                  'PctPublicCoverage',
                  'PctPublicCoverageAlone',
                  'PctWhite',
                  'PctBlack',
                  'PctMarriedHouseholds')
refined_cancer <- fin_cancer[,!names(fin_cancer) %in% highvif_vars]
str(refined_cancer)
# Build the model
train.data2  <- refined_cancer[training.samples, ]
test.data2 <- refined_cancer[-training.samples,]

ref_OLSmodel <- lm(TARGET_deathRate ~., data = train.data2)

# Make predictions
ref.OLS.test.pred <- ref_OLSmodel %>% predict(test.data2)

# What does the linear model look like?
# We will use robust standard errors. From what I have seen in the later areas (when checking asssumptions),
# I know that some heteroscedasticity exists.
summary(ref_OLSmodel, robust=T)
sqrt(mean(residuals(ref_OLSmodel)^2))
# confint(ref_OLSmodel, level=0.95)

# Model performance
eval_metrics2 = function(model, df, predictions, target){
  resids = df[,target] - predictions
  resids2 = resids**2
  N = length(predictions)
  k = length(ref_OLSmodel$coefficients) + 1
  r2 = R2(predictions, test.data2$TARGET_deathRate, na.rm=TRUE)
  adj_r2 = r2*(N-1)/(N-k)
  print(paste0("R^2: ", as.numeric(r2,2)))
  print(paste0("Adj. R^2: ", as.numeric(adj_r2,2))) #Adjusted R-squared
  #print(as.character(round(sqrt(sum(resids2)/N), 2))) #RMSE
  print(paste0("RMSE: ", RMSE(predictions, test.data2$TARGET_deathRate, na.rm=TRUE)))
}
# Here are the metrics for our test data
ans <- eval_metrics2(ref_OLSmodel,test.data2, ref.OLS.test.pred, target = 'TARGET_deathRate')
ols_vif_tol(ref_OLSmodel)


# SECOND MODEL: REFINEMENT
# this is used to select the best lambda value
lambda_seq <- 10^seq(2, -3, by = -.1)

# define our input parameters
x.train2 <- model.matrix(TARGET_deathRate ~ ., train.data2)
y.train2 <- train.data2$TARGET_deathRate
x.test2 <- model.matrix(TARGET_deathRate ~ ., test.data2)
y.test2 <- test.data2$TARGET_deathRate

# initial run-through to utilize cross-validation
output2 <- cv.glmnet(x.train2, y.train2, alpha = 1, lambda = lambda_seq, nfolds = 5)
plot(output2)

best.lambda2 <- output2$lambda.min
best.lambda2

# using the best.lambda, we will train the lasso model again!
lasso_best2 <- glmnet(x.train2, y.train2, alpha = 1, lambda = best.lambda2)
lasso.train.pred2 <- predict(lasso_best2, s=best.lambda2, newx=x.train2)
lasso.test.pred2 <- predict(lasso_best2, s=best.lambda2, newx=x.test2)


# let's evaluate the model
# Compute R^2 from true and predicted values
eval_results(y.train2, lasso.train.pred2, train.data2)
eval_results(y.test2, lasso.test.pred2, test.data2)

# Let's take a look at the lasso estimated coefficients
coef(lasso_best2)
