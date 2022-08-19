library(ggplot2)
library(glmnet)
library(readr)
library(randomForest)
library(tidyverse)
library(gridExtra)
rf.data <- read_csv("file_path_here")
ailerons <- as.matrix(rf.data)

# use the old sampling algorithm 
RNGkind(sample.kind = "Rounding")

# set seed for reproducibility
set.seed(10)


# For ridge,lasso, el-net, and random forrest we need to iterate 100 times:
#   Randomly Split the Dataset into D.Train & D.Test
#   Use D.Train to fit ridge, lasso, el-net, and random forrest models
#   Tune the lambda's using 10-fold CV
#   For each estimated model calculate R.sq.test and R.sq.train


# For Ridge Regression 
# 100 times we will split the data into train and sample
#   - we need to create 100 different random samples
#   - then split the original data set into train and test
#  For each iteration we will: 
#   - Perform 10 fold CV to find the optimal level of lambda
#   - Do this using the k-fold cross validation procedure used for Homework 3
#   - For each lambda we will fit a ridge model on D.Train
#   - Then, we will calculate and store the prediction errors in a temporary data frame
#   - we will then get the MSE for each lambda and store that in a permananet data frame outside the loop
#   - Will will also store the standard errors for each lambda and each iteration in a separate data frame

n   = nrow(ailerons)
p   = ncol(ailerons[,-1])
X = ailerons[,1:40]
y = ailerons[,41]

# Initialize storage for the Test MSE, Variance, and R^2 for ridge, lasso, elastic net

mse.ridge = matrix(0, nrow = 100, 1)
mse.lasso = matrix(0, nrow = 100, 1)
mse.elnet = matrix(0, nrow = 100, 1)

var.ridge = matrix(0, nrow = 100, 1)
var.lasso = matrix(0, nrow = 100, 1)
var.elnet = matrix(0, nrow = 100, 1)

rsq.ridge = matrix(0, 100, 1)
rsq.lasso = matrix(0, 100, 1)
rsq.elnet = matrix(0, 100, 1)

# Initialize storage for the Training MSE, Variance, and R^2 for ridge, lasso, and elnet 
mse.ridge.train = matrix(0, 100, 1)
mse.lasso.train = matrix(0, 100, 1)
mse.elnet.train = matrix(0, 100, 1)

var.ridge.train = matrix(0, 100, 1)
var.lasso.train = matrix(0, 100, 1)
var.elnet.train = matrix(0, 100, 1)

rsq.ridge.train = matrix(0, 100, 1)
rsq.lasso.train = matrix(0, 100, 1)
rsq.elnet.train = matrix(0, 100, 1)


# capture the time it takes to execute cross validation
time.cv.ridge = matrix(0, 100, 1)
time.cv.lasso = matrix(0, 100, 1)
time.cv.elnet = matrix(0, 100, 1)

cv.times = data.frame(Ridge = mean(time.cv.ridge),
                      Lasso = mean(time.cv.lasso),
                      ElNet = mean(time.cv.elnet))
cv.times
# Capture the residuals for one of the samples for both the test data and training data
res.ridge = matrix(0,2750)
res.lasso = matrix(0,2750)
res.elnet = matrix(0,2750)


res.ridge.train = matrix(0,11000)
res.lasso.train = matrix(0,11000)
res.elnet.train = matrix(0,11000)

# Store the cross validation results for the 5th sample in order to plot. 

for(i in 1:100){

    train          =  sample(1:n, ceiling(.8*n))
    X.train        =  X[train,]
    y.train        =  y[train]
    ybar.train     =  mean(y.train)
    n.train        =  nrow(X.train)

    X.val          =  X[-train,]
    y.val          =  y[-train]
    ybar.val       =  mean(y.val)
    n.test         =  nrow(X.val)

    ridge.start       =  proc.time()
    cv.ridge          =  cv.glmnet(X.train, y.train, alpha = 0)
    ridge.end         =  proc.time()
    time.cv.ridge[i]  =  ridge.end[3] - ridge.start[3]
    
    lasso.start       =  proc.time()
    cv.lasso          =  cv.glmnet(X.train, y.train, alpha = 1)
    lasso.end         =  proc.time()
    time.cv.lasso[i]  =  lasso.end[3] - lasso.start[3]
    
    elnet.start       =  proc.time()
    cv.elnet          =  cv.glmnet(X.train, y.train, alpha =.5)
    elnet.end         =  proc.time()
    time.cv.elnet[i]  =  elnet.end[3] - elnet.start[3]


    # Extract the lambda chosen by CV and fit the model
    lambda.ridge      =  cv.ridge$lambda.min
    lambda.lasso      =  cv.lasso$lambda.min
    lambda.elnet      =  cv.elnet$lambda.min

    ridge.fit         =  glmnet(X.train, y.train, alpha =  0, lambda = lambda.ridge)
    lasso.fit         =  glmnet(X.train, y.train, alpha =  1, lambda = lambda.lasso)
    elnet.fit         =  glmnet(X.train, y.train, alpha = .5, lambda = lambda.elnet)

    y.hat.ridge       =  predict(ridge.fit, newx = X.val)
    y.hat.lasso       =  predict(lasso.fit, newx = X.val)
    y.hat.elnet       =  predict(elnet.fit, newx = X.val)

    # Capture the residuals and plot the cross validation curves
    if(i == 5){
        res.ridge = y.val - y.hat.ridge
        res.lasso = y.val - y.hat.lasso
        res.elnet = y.val - y.hat.elnet

        plot(cv.ridge)
        title("CV Curve - Ridge", line = 3)
        plot(cv.lasso)
        title("CV Curve - Lasso", line = 3)
        plot(cv.elnet) 
        title("CV Curve - Elastic Net", line = 3)
    }

    mse.ridge[i]   =  mean((y.val - y.hat.ridge)^2)
    mse.lasso[i]   =  mean((y.val - y.hat.lasso)^2)
    mse.elnet[i]   =  mean((y.val - y.hat.elnet)^2)

    var.ridge[i]   = mean((y.val - ybar.val)^2)
    var.lasso[i]   = mean((y.val - ybar.val)^2)
    var.elnet[i]   = mean((y.val - ybar.val)^2)

    rsq.ridge[i]   = 1 - mse.ridge[i]/var.ridge[i]
    rsq.lasso[i]   = 1 - mse.lasso[i]/var.lasso[i]
    rsq.elnet[i]   = 1 - mse.elnet[i]/var.elnet[i]


    # Calculate R^2 for the training data set
    yhat.ridge.train     = predict(ridge.fit, newx = X.train)
    yhat.lasso.train     = predict(lasso.fit, newx = X.train)
    yhat.elnet.train     = predict(elnet.fit, newx = X.train)

    #Store the training residuals
    if(i == 5){
        res.ridge.train = y.val - yhat.ridge.train
        res.lasso.train = y.val - yhat.lasso.train
        res.elnet.train = y.val - yhat.elnet.train
    }


    mse.ridge.train[i]   = mean((y.train - yhat.ridge.train)^2)
    mse.lasso.train[i]   = mean((y.train - yhat.ridge.train)^2)
    mse.elnet.train[i]   = mean((y.train - yhat.ridge.train)^2)

    var.ridge.train[i]   = mean((y.train - ybar.train)^2)
    var.lasso.train[i]   = mean((y.train - ybar.train)^2)
    var.elnet.train[i]   = mean((y.train - ybar.train)^2)

    rsq.ridge.train[i]   = 1 - mse.ridge.train[i, ]/var.ridge.train[i,]
    rsq.lasso.train[i]   = 1 - mse.lasso.train[i, ]/var.lasso.train[i,]
    rsq.elnet.train[i]   = 1 - mse.elnet.train[i, ]/var.elnet.train[i,]
    
}


# Random Forest Test MSE and R^2
mse.rf    = matrix(0, 100, 1)
var.rf    = matrix(0, 100, 1)
rsq.rf    = matrix(0, 100, 1)

# Random Forest Training MSE and R^2 
mse.rf.train = matrix(0,100,1)
var.rf.train = matrix(0,100,1)
rsq.rf.train = matrix(0,100,1)

# Test Residuals and Training Residuals
res.rf = matrix(0, 2750)
res.rf.train = matrix(0, 11000)

# Random Forest - put  into its own loop to conserve time
for(i in 1:100){
    # Training Set
    train          =  sample(1:n, ceiling(.8*n))
    d.train        =  rf.data[train,]
    y.train        =  as.matrix(rf.data[train,41])
    ybar.train     =  mean(y.train)
    n.train        =  nrow(d.train)
    
    
    # Test Set
    d.test         =  rf.data[-train, ]
    y.test         =  as.matrix(rf.data[-train,41])
    ybar.test      =  mean(y.test)
    n.test         =  nrow(d.test)
    
    # Train a full tree on the training data
    base.forest    =  randomForest(goal~., d.train, mtry = sqrt(40))
    
    
    # Make predictions using the test data
    pred.test      =  predict(base.forest, newdata = d.test)
    mse.rf[i]      =  mean((y.test - pred.test)^2)
    var.rf[i]      =  mean((y.test - ybar.test)^2)
    rsq.rf[i,1]    =  1 - mse.rf[i]/var.rf[i]
    
    if(i == 5){
        res.rf = pred.test - y.test
        res.rf.train = predict(base.forest, newdata = d.train) - y.train
    }
    
    # Calculate the training MSE and test R^2
    pred.train         = predict(base.forest, newdata = d.train)
    mse.rf.train[i]    = mean((y.train - pred.train)^2)
    var.rf.train[i]    = mean((y.train - ybar.train)^2)
    
    rsq.rf.train[i]    = 1 - mse.rf.train[i]/var.rf.train[i]
}



## Part 5 - model the entire dataset
model         = c("Ridge", "Lasso", "Elastic Net", "Random Forest")
time.full.fit = matrix(0,4)
rsq.5        = matrix(0,4)
rsq.95       = matrix(0,4)
rsq.5[1] = quantile(rsq.ridge, probs=c(.05))
rsq.5[2] = quantile(rsq.lasso, probs=c(.05))
rsq.5[3] = quantile(rsq.elnet, probs=c(.05))
rsq.5[4] = quantile(rsq.rf,    probs=c(.05))

rsq.95[1] = quantile(rsq.ridge, probs=c(.95))
rsq.95[2] = quantile(rsq.lasso, probs=c(.95))
rsq.95[3] = quantile(rsq.elnet, probs=c(.95))
rsq.95[4] = quantile(rsq.rf,    probs=c(.95))
# Record the time it takes to cross validate the entire data set and fit the model

#############
##  Ridge  ##
#############
start.full.ridge    = proc.time()
full.cv.ridge       = cv.glmnet(X, y, alpha = 0)
full.ridge          = glmnet(X, y, alpha = 0, lambda = full.cv.ridge$lambda.min)
end.full.ridge      = proc.time()
time.full.fit[1]    = end.full.ridge[3]-start.full.ridge[3]


#############
##  Lasso  ##
#############

start.full.lasso    = proc.time()
full.cv.lasso       = cv.glmnet(X, y, alpha = 1)
full.lass           = glmnet(X, y, alpha = 1, lambda = full.cv.lasso$lambda.min)
end.full.lasso      = proc.time()
time.full.fit[2]    = end.full.lasso[3]-start.full.lasso[3]


#############
##  Elnet  ##
#############

start.full.elnet    = proc.time()
full.cv.elnet       = cv.glmnet(X, y, alpha = 0.5)
full.elnet          = glmnet(X, y, alpha = 0.5, lambda = full.cv.elnet$lambda.min)
end.full.elnet      = proc.time()
time.full.fit[3]    = end.full.elnet[3]-start.full.elnet[3]


####################
##  Random Forest ##
####################


start.full.rf       = proc.time()
full.rf.model       = randomForest(goal~., rf.data, mtry = sqrt(40))
end.full.rf         = proc.time()
time.full.fit[4]    = end.full.rf[3]-start.full.rf[3]



# Collect the R^2 and full fit timing in a single table
table.df      = data.frame(Model = model, RsqLowerQuantile = rsq.5, RsqUpperQuantile = rsq.95,
                           Time = time.full.fit)

# Create an image file for the table
png("RsqTimes.png", width = 430, height = 200, bg = "white")
grid.table(table.df)
dev.off()

# Put together barplots of the standardized coefficients
s = apply(X, 2, sd)

ridge.coef = as.vector(full.ridge$beta * s)
lasso.coef = as.vector(full.lass$beta  * s)
elnet.coef = as.vector(full.elnet$beta * s)


g = importance(full.rf.model)
rownames(g) <- NULL
f = seq(1:40)
VarName = colnames(X)
imp.df = data.frame(Name = VarName, Variable = f,
                    Ridge = ridge.coef, Lasso = lasso.coef,
                    Elnet = elnet.coef, Importance = g)


imp.df = arrange(imp.df, desc(Elnet))
imp.df$Variable = factor(imp.df$Variable, levels =imp.df$Variable)


ridgePlot = imp.df %>% ggplot(aes(x = as.factor(Variable), y = Ridge)) +
  geom_col() +
  labs(title = "Standardized Ridge Coefficients", x = "Variable", y = "Coefficient") +
  theme_bw()


lassoPlot = imp.df %>% ggplot(aes(x = as.factor(Variable), y = Lasso)) + geom_col()+ labs(title = "Standardized Lasso Coefficients", x = "Variable", y = "Coefficient")+ theme_bw()
elnetPlot = imp.df %>% ggplot(aes(x = as.factor(Variable), y = Elnet)) + geom_col()+ labs(title = "Standardized Elastic Net Coefficients", x = "Variable", y = "Coefficient")+ theme_bw()
rfPlot = imp.df %>% ggplot(aes(x = as.factor(Variable), y = IncNodePurity)) + geom_col()+ labs(title = "Random Forest - Variable Importance", x = "Variable", y = "Coefficient")+ theme_bw()


grid.arrange(ridgePlot, lassoPlot, elnetPlot, rfPlot, nrow = 4 )


# Part 4 - Side by Side box plots of R^2 Test and Train

rsq.test.df = data.frame(Sample = rep("Test", 400), Ridge = rsq.ridge,
                         Lasso = rsq.lasso, ElNet = rsq.elnet,
                         RandomForest = rsq.rf)

rsq.train.df = data.frame(Sample = rep("Train", 400), Ridge = rsq.ridge.train,
                          Lasso = rsq.lasso.train, ElNet = rsq.elnet.train,
                          RandomForest = rsq.rf.train)
rsq.df = rbind(rsq.test.df, rsq.train.df)
dim(rsq.df)

ridge.rsq.plot = rsq.df %>% ggplot(aes(x = Sample, y = Ridge)) + geom_boxplot() + labs(title = "Ridge R^2", y = "R^2") + theme_bw()
lasso.rsq.plot =rsq.df %>% ggplot(aes(x = Sample, y = Lasso)) + geom_boxplot() + labs(title = "Lasso R^2", y = "R^2") + theme_bw()
elnet.rsq.plot =rsq.df %>% ggplot(aes(x = Sample, y = ElNet)) + geom_boxplot() + labs(title = "Elastic Net R^2", y = "R^2") + theme_bw()
rf.rsq.plot =rsq.df %>% ggplot(aes(x = Sample, y = RandomForest)) + geom_boxplot() + labs(title = "Random Forest R^2", y = "R^2") + theme_bw()

grid.arrange(ridge.rsq.plot,lasso.rsq.plot,elnet.rsq.plot, rf.rsq.plot, nrow = 2, ncol =2)


# Plot the barplots of the residuals for Test and Train. 
names.res = c("Sample", "Ridge", "Lasso", "ElNet", "RandomForest")

test = rep("Test", 2750)
train = rep("Train", 11000)
res.df.test = data.frame(Sample = test,
                         Ridge  = res.ridge, 
                         Lasso  = res.lasso, 
                         ElNet  = res.elnet, 
                         RandomForest = res.rf)
colnames(res.df.test) = names.res

res.df.train = data.frame(Sample = train,
                         Ridge = res.ridge.train,
                         Lasso = res.lasso.train,
                         ElNet = res.elnet.train,
                         RandomForest = res.rf.train)
colnames(res.df.train) = names.res

residuals.df = rbind(res.df.test, res.df.train)
colnames(residuals.df) = names.res

# Residual Box Plots

ridge.box = residuals.df %>% ggplot(aes(x = Sample, y = Ridge)) + geom_boxplot() + 
    labs(title="Ridge Residuals", y = "Residuals") + theme_bw()
lasso.box = residuals.df %>% ggplot(aes(x = Sample, y = Lasso)) + geom_boxplot() + 
    labs(title="Lasso Residuals", y = "Residuals")+ theme_bw()
elnet.box = residuals.df %>% ggplot(aes(x = Sample, y = ElNet)) + geom_boxplot() +
    labs(title="Elastic Net Residuals", y = "Residuals")+ theme_bw()
rf.box = residuals.df %>% ggplot(aes(x = Sample, y = RandomForest)) + geom_boxplot() +
    labs(title="Random Forest Residuals", y = "Residuals")+ theme_bw()
grid.arrange(ridge.box, lasso.box, elnet.box, rf.box, nrow = 2, ncol =2)

