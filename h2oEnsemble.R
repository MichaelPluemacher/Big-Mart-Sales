#
# Big Mart Sale Prediction
#
# building an ensemble of models
# from the h2o package
#

# deep learning neural network from h2o
library(h2o)
library(h2oEnsemble)

# parallel computing
library(doParallel)

#working directory
path <- "~/BigMartSales"

#set working directory
setwd(path)

# Load Datasets with the new predictors
# in the order of their importance,
# according to rfe
predictors <- read.csv("ordered_predictors.csv")
sales <- read.csv("sales.csv")$x
ordered_test <- read.csv("ordered_test.csv")

# the original test set
test <- read.csv("Test_u94Q5KV.csv")

# rescale sales to interval [0,1]
maxSales <- max(sales)
sales <- as.vector(sales/maxSales)

set.seed(42)

# do it in parallel
cl <- makeCluster(detectCores()); registerDoParallel(cl)


###############################################################
#
# H2O
#
###############################################################

## Start a local cluster with 8GB RAM
localH2O = h2o.init(ip = "localhost",
                    port = 54321,
                    startH2O = TRUE,
                    nthreads = -1,     # use all CPUs
                    max_mem_size = '8g') # maximum memory

# import dataframe into h2o
train.hex <- as.h2o(cbind(predictors,sales), destination_frame="train.hex")
test.hex <- as.h2o(ordered_test, destination_frame="test.hex")

######################################################
#
# h2o ensemble using the 12 most important predictors
#
######################################################

# # number of predictors we use to buld the model:
predictorCount <- 12

# glm base learners
h2o.glm.1 <- function(..., alpha = 0.0) h2o.glm.wrapper(..., alpha = alpha, solver = "IRLSM")
h2o.glm.2 <- function(..., alpha = 0.5) h2o.glm.wrapper(..., alpha = alpha, solver = "IRLSM")
h2o.glm.3 <- function(..., alpha = 1.0) h2o.glm.wrapper(..., alpha = alpha, solver = "IRLSM")
h2o.glm.4 <- function(..., alpha = 0.0) h2o.glm.wrapper(..., alpha = alpha, solver = "L_BFGS")
h2o.glm.5 <- function(..., alpha = 0.5) h2o.glm.wrapper(..., alpha = alpha, solver = "L_BFGS")
h2o.glm.6 <- function(..., alpha = 1.0) h2o.glm.wrapper(..., alpha = alpha, solver = "L_BFGS")

# random forest base learners
h2o.randomForest.1 <- function(...,
                               ntrees = 500,
                               mtries = predictorCount,
                               sample_rate = 0.8,
                               col_sample_rate = 1,
                               max_depth = 10,
                               min_rows = 2,
                               seed = 1)
  h2o.randomForest.wrapper(...,
                           ntrees = ntrees,
                           mtries = mtries,
                           sample_rate = sample_rate,
                           col_sample_rate = col_sample_rate,
                           max_depth = max_depth,
                           min_rows = min_rows,
                           seed = seed)
h2o.randomForest.2 <- function(...,
                               ntrees = 500,
                               mtries = -1,
                               sample_rate = 0.8,
                               col_sample_rate = 1,
                               max_depth = 10,
                               min_rows = 2,
                               seed = 1)
  h2o.randomForest.wrapper(...,
                           ntrees = ntrees,
                           mtries = mtries,
                           sample_rate = sample_rate,
                           col_sample_rate = col_sample_rate,
                           max_depth = max_depth,
                           min_rows = min_rows,
                           seed = seed)
h2o.randomForest.3 <- function(...,
                               ntrees = 500,
                               mtries = predictorCount,
                               sample_rate = 0.8,
                               col_sample_rate = 0.5,
                               max_depth = 10,
                               min_rows = 2,
                               seed = 1)
  h2o.randomForest.wrapper(...,
                           ntrees = ntrees,
                           mtries = mtries,
                           sample_rate = sample_rate,
                           col_sample_rate = col_sample_rate,
                           max_depth = max_depth,
                           min_rows = min_rows,
                           seed = seed)
h2o.randomForest.4 <- function(...,
                               ntrees = 500,
                               mtries = -1,
                               sample_rate = 0.8,
                               col_sample_rate = 0.5,
                               max_depth = 10,
                               min_rows = 2,
                               seed = 1)
  h2o.randomForest.wrapper(...,
                           ntrees = ntrees,
                           mtries = mtries,
                           sample_rate = sample_rate,
                           col_sample_rate = col_sample_rate,
                           max_depth = max_depth,
                           min_rows = min_rows,
                           seed = seed)
h2o.randomForest.5 <- function(...,
                               ntrees = 500,
                               mtries = 12,
                               sample_rate = 0.8,
                               col_sample_rate = 1,
                               max_depth = 10,
                               min_rows = 2,
                               seed = 1)
  h2o.randomForest.wrapper(...,
                           ntrees = ntrees,
                           mtries = mtries,
                           sample_rate = sample_rate,
                           col_sample_rate = col_sample_rate,
                           max_depth = max_depth,
                           min_rows = min_rows,
                           seed = seed)
h2o.randomForest.6 <- function(...,
                               ntrees = 500,
                               mtries = 12,
                               sample_rate = 0.8,
                               col_sample_rate = 0.5,
                               max_depth = 10,
                               min_rows = 2,
                               seed = 1)
  h2o.randomForest.wrapper(...,
                           ntrees = ntrees,
                           mtries = mtries,
                           sample_rate = sample_rate,
                           col_sample_rate = col_sample_rate,
                           max_depth = max_depth,
                           min_rows = min_rows,
                           seed = seed)

# gbm base learners
h2o.gbm.1 <- function(...,
                      ntrees = 100,
                      sample_rate = 0.5, # row sample rate
                      col_sample_rate = 0.5, # column sample rate
                      #col_sample_rate_per_tree = 0.5, # column sample rate per tree
                      min_rows = 1, # Minimum number of rows to assign to teminal nodes
                      seed = 1)
  h2o.gbm.wrapper(...,
                  ntrees = ntrees,
                  sample_rate = sample_rate,
                  col_sample_rate = col_sample_rate,
                  #col_sample_rate_per_tree = col_sample_rate_per_tree,
                  min_rows = min_rows,
                  seed = seed)
h2o.gbm.2 <- function(...,
                      ntrees = 100,
                      sample_rate = 0.5, # row sample rate
                      col_sample_rate = 0.5, # column sample rate
                      #col_sample_rate_per_tree = 0.5, # column sample rate per tree
                      min_rows = 1, # Minimum number of rows to assign to teminal nodes
                      nbins = 50,
                      seed = 1)
  h2o.gbm.wrapper(...,
                  ntrees = ntrees,
                  sample_rate = sample_rate,
                  col_sample_rate = col_sample_rate,
                  #col_sample_rate_per_tree = col_sample_rate_per_tree,
                  min_rows = min_rows,
                  nbins = nbins,
                  seed = seed)
h2o.gbm.3 <- function(...,
                      ntrees = 100,
                      sample_rate = 0.5, # row sample rate
                      col_sample_rate = 0.5, # column sample rate
                      #col_sample_rate_per_tree = 0.5, # column sample rate per tree
                      min_rows = 1, # Minimum number of rows to assign to teminal nodes
                      max_depth = 5,
                      seed = 1)
  h2o.gbm.wrapper(...,
                  ntrees = ntrees,
                  sample_rate = sample_rate,
                  col_sample_rate = col_sample_rate,
                  #col_sample_rate_per_tree = col_sample_rate_per_tree,
                  min_rows = min_rows,
                  max_depth = max_depth,
                  seed = seed)
h2o.gbm.4 <- function(...,
                      ntrees = 100,
                      sample_rate = 0.5, # row sample rate
                      col_sample_rate = 0.8, # column sample rate
                      #col_sample_rate_per_tree = 0.5, # column sample rate per tree
                      min_rows = 1, # Minimum number of rows to assign to teminal nodes
                      seed = 1)
  h2o.gbm.wrapper(...,
                  ntrees = ntrees,
                  sample_rate = sample_rate,
                  col_sample_rate = col_sample_rate,
                  #col_sample_rate_per_tree = col_sample_rate_per_tree,
                  min_rows = min_rows,
                  seed = seed)
h2o.gbm.5 <- function(...,
                      ntrees = 100,
                      sample_rate = 0.5, # row sample rate
                      col_sample_rate = 0.7, # column sample rate
                      #col_sample_rate_per_tree = 0.5, # column sample rate per tree
                      min_rows = 1, # Minimum number of rows to assign to teminal nodes
                      seed = 1)
  h2o.gbm.wrapper(...,
                  ntrees = ntrees,
                  sample_rate = sample_rate,
                  col_sample_rate = col_sample_rate,
                  #col_sample_rate_per_tree = col_sample_rate_per_tree,
                  min_rows = min_rows,
                  seed = seed)
h2o.gbm.6 <- function(...,
                      ntrees = 100,
                      sample_rate = 0.5, # row sample rate
                      col_sample_rate = 0.6, # column sample rate
                      #col_sample_rate_per_tree = 0.5, # column sample rate per tree
                      min_rows = 1, # Minimum number of rows to assign to teminal nodes
                      seed = 1)
  h2o.gbm.wrapper(...,
                  ntrees = ntrees,
                  sample_rate = sample_rate,
                  col_sample_rate = col_sample_rate,
                  #col_sample_rate_per_tree = col_sample_rate_per_tree,
                  min_rows = min_rows,
                  seed = seed)
h2o.gbm.7 <- function(...,
                      ntrees = 100,
                      sample_rate = 0.5, # row sample rate
                      col_sample_rate = 0.5, # column sample rate
                      #col_sample_rate_per_tree = 0.5, # column sample rate per tree
                      min_rows = 1, # Minimum number of rows to assign to teminal nodes
                      balance_classes = TRUE,
                      seed = 1)
  h2o.gbm.wrapper(...,
                  ntrees = ntrees,
                  sample_rate = sample_rate,
                  col_sample_rate = col_sample_rate,
                  #col_sample_rate_per_tree = col_sample_rate_per_tree,
                  min_rows = min_rows,
                  balance_classes = balance_classes,
                  seed = seed)
h2o.gbm.8 <- function(...,
                      ntrees = 100,
                      sample_rate = 0.5, # row sample rate
                      col_sample_rate = 0.5, # column sample rate
                      #col_sample_rate_per_tree = 0.5, # column sample rate per tree
                      min_rows = 1, # Minimum number of rows to assign to teminal nodes
                      max_depth = 3,
                      seed = 1)
  h2o.gbm.wrapper(...,
                  ntrees = ntrees,
                  sample_rate = sample_rate,
                  col_sample_rate = col_sample_rate,
                  #col_sample_rate_per_tree = col_sample_rate_per_tree,
                  min_rows = min_rows,
                  max_depth = max_depth,
                  seed = seed)
h2o.gbm.9 <- function(...,
                      ntrees = 100,
                      sample_rate = 0.5, # row sample rate
                      col_sample_rate = 0.5, # column sample rate
                      #col_sample_rate_per_tree = 0.5, # column sample rate per tree
                      min_rows = 1, # Minimum number of rows to assign to teminal nodes
                      max_depth = 5,
                      seed = 1)
  h2o.gbm.wrapper(...,
                  ntrees = ntrees,
                  sample_rate = sample_rate,
                  col_sample_rate = col_sample_rate,
                  #col_sample_rate_per_tree = col_sample_rate_per_tree,
                  min_rows = min_rows,
                  max_depth = max_depth,
                  seed = seed)
h2o.gbm.10 <- function(...,
                       ntrees = 100,
                       sample_rate = 0.5, # row sample rate
                       col_sample_rate = 0.5, # column sample rate
                       #col_sample_rate_per_tree = 0.5, # column sample rate per tree
                       min_rows = 2, # Minimum number of rows to assign to teminal nodes
                       max_depth = 5,
                       seed = 1)
  h2o.gbm.wrapper(...,
                  ntrees = ntrees,
                  sample_rate = sample_rate,
                  col_sample_rate = col_sample_rate,
                  #col_sample_rate_per_tree = col_sample_rate_per_tree,
                  min_rows = min_rows,
                  max_depth = max_depth,
                  seed = seed)
h2o.gbm.11 <- function(...,
                       ntrees = 100,
                       sample_rate = 0.5, # row sample rate
                       col_sample_rate = 0.5, # column sample rate
                       #col_sample_rate_per_tree = 0.5, # column sample rate per tree
                       min_rows = 2, # Minimum number of rows to assign to teminal nodes
                       max_depth = 3,
                       seed = 1)
  h2o.gbm.wrapper(...,
                  ntrees = ntrees,
                  sample_rate = sample_rate,
                  col_sample_rate = col_sample_rate,
                  #col_sample_rate_per_tree = col_sample_rate_per_tree,
                  min_rows = min_rows,
                  max_depth = max_depth,
                  seed = seed)
h2o.gbm.12 <- function(...,
                       ntrees = 150,
                       sample_rate = 0.5, # row sample rate
                       col_sample_rate = 0.5, # column sample rate
                       #col_sample_rate_per_tree = 0.5, # column sample rate per tree
                       min_rows = 2, # Minimum number of rows to assign to teminal nodes
                       max_depth = 3,
                       seed = 1)
  h2o.gbm.wrapper(...,
                  ntrees = ntrees,
                  sample_rate = sample_rate,
                  col_sample_rate = col_sample_rate,
                  #col_sample_rate_per_tree = col_sample_rate_per_tree,
                  min_rows = min_rows,
                  max_depth = max_depth,
                  seed = seed)
h2o.gbm.13 <- function(...,
                       ntrees = 80,
                       sample_rate = 0.5, # row sample rate
                       col_sample_rate = 0.5, # column sample rate
                       #col_sample_rate_per_tree = 0.5, # column sample rate per tree
                       min_rows = 2, # Minimum number of rows to assign to teminal nodes
                       max_depth = 3,
                       seed = 1)
  h2o.gbm.wrapper(...,
                  ntrees = ntrees,
                  sample_rate = sample_rate,
                  col_sample_rate = col_sample_rate,
                  #col_sample_rate_per_tree = col_sample_rate_per_tree,
                  min_rows = min_rows,
                  max_depth = max_depth,
                  seed = seed)
h2o.gbm.14 <- function(...,
                       ntrees = 80,
                       sample_rate = 0.5, # row sample rate
                       col_sample_rate = 0.5, # column sample rate
                       #col_sample_rate_per_tree = 0.5, # column sample rate per tree
                       min_rows = 1, # Minimum number of rows to assign to teminal nodes
                       max_depth = 5,
                       seed = 1)
  h2o.gbm.wrapper(...,
                  ntrees = ntrees,
                  sample_rate = sample_rate,
                  col_sample_rate = col_sample_rate,
                  #col_sample_rate_per_tree = col_sample_rate_per_tree,
                  min_rows = min_rows,
                  max_depth = max_depth,
                  seed = seed)

# deep learning neural net base learners
h2o.deeplearning.1 <- function(...,
                               hidden = c(50,50),
                               activation = "TanhWithDropout",
                               epochs = 500,
                               seed = 1)
  h2o.deeplearning.wrapper(...,
                           hidden = hidden,
                           activation = activation,
                           nesterov_accelerated_gradient = TRUE,
                           seed = seed)
h2o.deeplearning.2 <- function(...,
                               hidden = c(30,30,30),
                               activation = "Tanh",
                               epochs = 500,
                               seed = 1)
  h2o.deeplearning.wrapper(...,
                           hidden = hidden,
                           activation = activation,
                           nesterov_accelerated_gradient = TRUE,
                           seed = seed)
h2o.deeplearning.3 <- function(...,
                               hidden = c(13,13,13,13),
                               activation = "Rectifier",
                               epochs = 500,
                               max_w2 = 50,
                               seed = 1)
  h2o.deeplearning.wrapper(...,
                           hidden = hidden,
                           activation = activation,
                           nesterov_accelerated_gradient = TRUE,
                           seed = seed)
h2o.deeplearning.4 <- function(...,
                               hidden = c(30,30,30),
                               activation = "Rectifier",
                               epochs = 500,
                               max_w2 = 50,
                               seed = 1)
  h2o.deeplearning.wrapper(...,
                           hidden = hidden,
                           activation = activation,
                           nesterov_accelerated_gradient = TRUE,
                           seed = seed)
h2o.deeplearning.5 <- function(...,
                               hidden = c(30,30,30),
                               activation = "TanhWithDropout",
                               epochs = 500,
                               max_w2 = 10,
                               seed = 1)
  h2o.deeplearning.wrapper(...,
                           hidden = hidden,
                           activation = activation,
                           nesterov_accelerated_gradient = TRUE,
                           seed = seed)
h2o.deeplearning.6 <- function(...,
                               hidden = c(13,13,13,13),
                               activation = "TanhWithDropout",
                               epochs = 500,
                               max_w2 = 10,
                               seed = 1)
  h2o.deeplearning.wrapper(...,
                           hidden = hidden,
                           activation = activation,
                           nesterov_accelerated_gradient = TRUE,
                           seed = seed)
h2o.deeplearning.7 <- function(...,
                               hidden = c(13,13,13,13),
                               activation = "TanhWithDropout",
                               epochs = 500,
                               max_w2 = 10,
                               rate = 0.5,
                               seed = 1)
  h2o.deeplearning.wrapper(...,
                           hidden = hidden,
                           activation = activation,
                           nesterov_accelerated_gradient = TRUE,
                           seed = seed)
h2o.deeplearning.8 <- function(...,
                               hidden = c(14,14,14,14),
                               activation = "TanhWithDropout",
                               epochs = 500,
                               max_w2 = 10,
                               rate = 0.5,
                               seed = 1)
  h2o.deeplearning.wrapper(...,
                           hidden = hidden,
                           activation = activation,
                           nesterov_accelerated_gradient = TRUE,
                           seed = seed)
h2o.deeplearning.9 <- function(...,
                               hidden = c(20,20),
                               activation = "Tanh",
                               epochs = 500,
                               seed = 1)
  h2o.deeplearning.wrapper(...,
                           hidden = hidden,
                           activation = activation,
                           nesterov_accelerated_gradient = TRUE,
                           seed = seed)
h2o.deeplearning.10 <- function(...,
                                hidden = c(20,20),
                                activation = "Rectifier",
                                epochs = 500,
                                seed = 1)
  h2o.deeplearning.wrapper(...,
                           hidden = hidden,
                           activation = activation,
                           nesterov_accelerated_gradient = TRUE,
                           seed = seed)
h2o.deeplearning.11 <- function(...,
                                hidden = c(13,13,13,13),
                                activation = "TanhWithDropout",
                                epochs = 500,
                                seed = 1)
  h2o.deeplearning.wrapper(...,
                           hidden = hidden,
                           activation = activation,
                           nesterov_accelerated_gradient = TRUE,
                           adaptive_rate = TRUE,
                           rho = 0.9,
                           epsilon = 1e-8,
                           rate = 0.5,
                           input_dropout_ratio = 0.0, # fraction of inputs dropout
                           hidden_dropout_ratios = c(0.2,0.0,0.0,0.0), # fraction for nodes dropout
                           seed = seed)
h2o.deeplearning.12 <- function(...,
                                hidden = c(10,10,10,10),
                                activation = "TanhWithDropout",
                                epochs = 500,
                                seed = 1)
  h2o.deeplearning.wrapper(...,
                           hidden = hidden,
                           activation = activation,
                           nesterov_accelerated_gradient = TRUE,
                           adaptive_rate = TRUE,
                           rho = 0.9,
                           epsilon = 1e-8,
                           rate = 0.5,
                           input_dropout_ratio = 0.0, # fraction of inputs dropout
                           hidden_dropout_ratios = c(0.2,0.0,0.0,0.0), # fraction for nodes dropout
                           seed = seed)
h2o.deeplearning.13 <- function(...,
                                hidden = c(8,8,8,8),
                                activation = "TanhWithDropout",
                                epochs = 500,
                                seed = 1)
  h2o.deeplearning.wrapper(...,
                           hidden = hidden,
                           activation = activation,
                           nesterov_accelerated_gradient = TRUE,
                           adaptive_rate = TRUE,
                           rho = 0.9,
                           epsilon = 1e-8,
                           rate = 0.5,
                           input_dropout_ratio = 0.0, # fraction of inputs dropout
                           hidden_dropout_ratios = c(0.2,0.0,0.0,0.0), # fraction for nodes dropout
                           seed = seed)
h2o.deeplearning.14 <- function(...,
                                hidden = c(8,8,8,8),
                                activation = "TanhWithDropout",
                                epochs = 500,
                                seed = 1)
  h2o.deeplearning.wrapper(...,
                           hidden = hidden,
                           activation = activation,
                           nesterov_accelerated_gradient = TRUE,
                           adaptive_rate = TRUE,
                           rho = 0.9,
                           epsilon = 1e-8,
                           rate = 0.5,
                           input_dropout_ratio = 0.2, # fraction of inputs dropout
                           hidden_dropout_ratios = c(0.2,0.0,0.0,0.0), # fraction for nodes dropout
                           seed = seed)
h2o.deeplearning.15 <- function(...,
                                hidden = c(15,15,15,15),
                                activation = "TanhWithDropout",
                                epochs = 500,
                                seed = 1)
  h2o.deeplearning.wrapper(...,
                           hidden = hidden,
                           activation = activation,
                           nesterov_accelerated_gradient = TRUE,
                           adaptive_rate = TRUE,
                           rho = 0.9,
                           epsilon = 1e-8,
                           rate = 0.5,
                           input_dropout_ratio = 0.0, # fraction of inputs dropout
                           hidden_dropout_ratios = c(0.2,0.0,0.0,0.0), # fraction for nodes dropout
                           seed = seed)
h2o.deeplearning.16 <- function(...,
                                hidden = c(15,15,15,15),
                                activation = "TanhWithDropout",
                                epochs = 500,
                                seed = 1)
  h2o.deeplearning.wrapper(...,
                           hidden = hidden,
                           activation = activation,
                           nesterov_accelerated_gradient = TRUE,
                           adaptive_rate = TRUE,
                           rho = 0.9,
                           epsilon = 1e-8,
                           rate = 0.5,
                           max_w2 = 10,
                           input_dropout_ratio = 0.0, # fraction of inputs dropout
                           hidden_dropout_ratios = c(0.2,0.0,0.0,0.0), # fraction for nodes dropout
                           seed = seed)
h2o.deeplearning.17 <- function(...,
                                hidden = c(10,10,10,10),
                                activation = "TanhWithDropout",
                                epochs = 500,
                                seed = 1)
  h2o.deeplearning.wrapper(...,
                           hidden = hidden,
                           activation = activation,
                           nesterov_accelerated_gradient = TRUE,
                           adaptive_rate = TRUE,
                           rho = 0.9,
                           epsilon = 1e-8,
                           rate = 0.5,
                           max_w2 = 10,
                           input_dropout_ratio = 0.0, # fraction of inputs dropout
                           hidden_dropout_ratios = c(0.2,0.0,0.0,0.0), # fraction for nodes dropout
                           seed = seed)
h2o.deeplearning.18 <- function(...,
                                hidden = c(30,30,30),
                                activation = "TanhWithDropout",
                                epochs = 500,
                                seed = 1)
  h2o.deeplearning.wrapper(...,
                           hidden = hidden,
                           activation = activation,
                           nesterov_accelerated_gradient = TRUE,
                           adaptive_rate = TRUE,
                           rho = 0.9,
                           epsilon = 1e-8,
                           rate = 0.5,
                           max_w2 = 10,
                           input_dropout_ratio = 0.0, # fraction of inputs dropout
                           hidden_dropout_ratios = c(0.2,0.0,0.0), # fraction for nodes dropout
                           seed = seed)
h2o.deeplearning.19 <- function(...,
                                hidden = c(50,50,50),
                                activation = "TanhWithDropout",
                                epochs = 500,
                                seed = 1)
  h2o.deeplearning.wrapper(...,
                           hidden = hidden,
                           activation = activation,
                           nesterov_accelerated_gradient = TRUE,
                           adaptive_rate = TRUE,
                           rho = 0.9,
                           epsilon = 1e-8,
                           rate = 0.5,
                           max_w2 = 10,
                           input_dropout_ratio = 0.0, # fraction of inputs dropout
                           hidden_dropout_ratios = c(0.2,0.0,0.0), # fraction for nodes dropout
                           seed = seed)
h2o.deeplearning.20 <- function(...,
                                hidden = c(100,100),
                                activation = "TanhWithDropout",
                                epochs = 500,
                                seed = 1)
  h2o.deeplearning.wrapper(...,
                           hidden = hidden,
                           activation = activation,
                           nesterov_accelerated_gradient = TRUE,
                           adaptive_rate = TRUE,
                           rho = 0.9,
                           epsilon = 1e-8,
                           rate = 0.5,
                           max_w2 = 10,
                           input_dropout_ratio = 0.0, # fraction of inputs dropout
                           hidden_dropout_ratios = c(0.2,0.0), # fraction for nodes dropout
                           seed = seed)
h2o.deeplearning.21 <- function(...,
                                hidden = c(50,50),
                                activation = "TanhWithDropout",
                                epochs = 500,
                                seed = 1)
  h2o.deeplearning.wrapper(...,
                           hidden = hidden,
                           activation = activation,
                           nesterov_accelerated_gradient = TRUE,
                           adaptive_rate = TRUE,
                           rho = 0.9,
                           epsilon = 1e-8,
                           rate = 0.5,
                           max_w2 = 10,
                           input_dropout_ratio = 0.0, # fraction of inputs dropout
                           hidden_dropout_ratios = c(0.2,0.0), # fraction for nodes dropout
                           seed = seed)
h2o.deeplearning.22 <- function(...,
                                hidden = c(10,10,10,10,10),
                                activation = "TanhWithDropout",
                                epochs = 500,
                                seed = 1)
  h2o.deeplearning.wrapper(...,
                           hidden = hidden,
                           activation = activation,
                           nesterov_accelerated_gradient = TRUE,
                           adaptive_rate = TRUE,
                           rho = 0.9,
                           epsilon = 1e-8,
                           rate = 0.5,
                           max_w2 = 10,
                           input_dropout_ratio = 0.0, # fraction of inputs dropout
                           hidden_dropout_ratios = c(0.2,0.0,0.0,0.0,0.0), # fraction for nodes dropout
                           seed = seed)

# which of the base learners defined above do we want to include 
# in the ensemble?
# Play around with various combinations, you'll see some rather
# important differences

learner <- c(#"h2o.randomForest.2", 
             #"h2o.randomForest.3", 
             #"h2o.randomForest.4", 
             #"h2o.randomForest.6",
             "h2o.gbm.8",
             "h2o.gbm.11",
             "h2o.gbm.12",
             "h2o.gbm.13",
             "h2o.deeplearning.2",
             #"h2o.deeplearning.3",
             #"h2o.deeplearning.4",
             #"h2o.deeplearning.9",
             "h2o.deeplearning.11",
             "h2o.deeplearning.12",
             "h2o.deeplearning.15",
             "h2o.deeplearning.16",
             #"h2o.deeplearning.17",
             "h2o.deeplearning.18",
             "h2o.deeplearning.19",
             "h2o.deeplearning.20"#,
             #"h2o.deeplearning.21"
)

# define the metalearner
metalearner <- "h2o.glm.wrapper"
#metalearner <- "h2o.gbm.wrapper"
#metalearner <- "h2o.deeplearning.wrapper"
#metalearner <- "h2o.randomForest.wrapper"

# train the ensemble
fit <- h2o.ensemble(x = 1:predictorCount,  # column numbers for predictors
                    y = ncol(predictors)+1,   # column number for label
                    training_frame = train.hex, # data in H2O format
                    family = "AUTO", 
                    learner = learner, 
                    metalearner = metalearner,
                    cvControl = list(V = 3)) # 3-fold cross validation

# generate predictions on the test set

pred <- predict(fit, test.hex)
prediction12 <- as.data.frame(pred$pred*maxSales)

min(prediction12)
max(prediction12)

# predicted values smaller than zero are set to zero
prediction12[prediction12<0] <- 0

################################################
#
# END of N=12
#
################################################

# # clean up after h2o
h2o.removeAll()

# stop the parallel processing and register sequential front-end
stopCluster(cl); registerDoSEQ();

################################
#
# preparing the final submissions
#
################################

final12 <- data.frame(test$Item_Identifier, test$Outlet_Identifier, prediction12)

names(final12) <- c("Item_Identifier",
                    "Outlet_Identifier",
                    "Item_Outlet_Sales")

write.csv(final12, file="final12.csv", row.names=FALSE, quote = FALSE)

# free up some memory
gc(verbose = TRUE)

ls(all = TRUE)

rm(list = ls(all = TRUE)) 

ls(all = TRUE)

gc(verbose = TRUE)
