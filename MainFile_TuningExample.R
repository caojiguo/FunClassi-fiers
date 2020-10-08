#################################
# FNNs Classification Paper     #
#                               #
# Example of Tuning Procedure   #
#                               #
# Anonymized                    #
#################################

# Libraries
library(fda)
library(fda.usc)
library(keras)
library(ggplot2)
library(modEvA)
library(caret)
library(tfruns)
source("FNN_FunctionsFile.R")

# Clearing backend
K <- backend()
K$clear_session()
options(warn=-1)

# Setting seeds
set.seed(1919)
use_session_with_seed(
  1919,
  disable_gpu = F,
  disable_parallel_cpu = F,
  quiet = T
)

# Loading data
load("OJ.RData")

# Combining data
full_resp = c(OJ$y.learning, OJ$y.test)
full_df = data.frame(rbind(OJ$x.learning, OJ$x.test))

# Making classification bins
resp = ifelse(full_resp > 40, 1, 0)

# define the time points on which the functional predictor is observed. 
timepts = seq(1, 700, 1)

# define the fourier basis 
nbasis = 65
spline_basis = create.fourier.basis(c(min(timepts), max(timepts)), nbasis)

# convert the functional predictor into a fda object
fd =  Data2fd(timepts, t(full_df), spline_basis)
deriv1 = deriv.fd(fd)
deriv2 = deriv.fd(deriv1)

# Setting up arrays
func_cov_1 = fd$coefs
func_cov_2 = deriv1$coefs
func_cov_3 = deriv2$coefs
final_data = array(dim = c(nbasis, nrow(full_df), 1))
final_data[,,1] = func_cov_1
final_data[,,2] = func_cov_2
final_data[,,3] = func_cov_3

# fData Object
fdata_obj = fdata(full_df, argvals = timepts, rangeval = c(min(timepts), max(timepts)))

# Choosing fold number
num_folds = 2

# Creating folds
fold_ind = createFolds(resp, k = num_folds)

# Setting index
i = 1

################## 
# Splitting data #
##################

# Test and train
train_x = fdata_obj[-fold_ind[[i]],]
test_x = fdata_obj[fold_ind[[i]],]
train_y = resp[-fold_ind[[i]]]
test_y = resp[fold_ind[[i]]]

# Setting up for FNN
if(dim(final_data)[3] == 1){
  data_train = array(dim = c(nbasis, nrow(train_x$data), 1))
  data_test = array(dim = c(nbasis, nrow(test_x$data), 1))
  data_train[,,1] = final_data[, -fold_ind[[i]], ]
  data_test[,,1] = final_data[, fold_ind[[i]], ]
} else {
  data_train = array(dim = c(nbasis, nrow(train_x$data), dim(final_data)[3]))
  data_test = array(dim = c(nbasis, nrow(test_x$data), dim(final_data)[3]))
  data_train = final_data[, -fold_ind[[i]], ]
  data_test = final_data[, fold_ind[[i]], ]
}

############################################
# Running Functional Neural Network Tuning #
############################################

if(dim(data_train)[3] > 1){
  # Now, let's pre-process
  pre_train = FNN_Preprocess(func_cov = data_train,
                             basis_choice = c("fourier", "fourier", "fourier"),
                             num_basis = c(3, 5, 7),
                             domain_range = list(c(min(timepts), max(timepts)), 
                                                 c(min(timepts), max(timepts)), 
                                                 c(min(timepts), max(timepts))),
                             covariate_scaling = T,
                             raw_data = F)
  
  pre_test = FNN_Preprocess(func_cov = data_test,
                            basis_choice = c("fourier", "fourier", "fourier"),
                            num_basis = c(3, 5, 7),
                            domain_range = list(c(min(timepts), max(timepts)), 
                                                c(min(timepts), max(timepts)), 
                                                c(min(timepts), max(timepts))),
                            covariate_scaling = T,
                            raw_data = F)
} else {
  
  # Now, let's pre-process
  pre_train = FNN_Preprocess(func_cov = data_train,
                             basis_choice = c("fourier"),
                             num_basis = c(3),
                             domain_range = list(c(min(timepts), max(timepts))),
                             covariate_scaling = T,
                             raw_data = F)
  
  pre_test = FNN_Preprocess(func_cov = data_test,
                            basis_choice = c("fourier"),
                            num_basis = c(3),
                            domain_range = list(c(min(timepts), max(timepts))),
                            covariate_scaling = T,
                            raw_data = F)
}

# Flags
FLAGS <- flags(
  flag_numeric('dropout1', 0.3),
  flag_integer('neurons1', 128),
  flag_integer('neurons2', 128),
  flag_integer('neurons3', 128),
  flag_numeric('lr', 0.001),
  flag_numeric('l2', 0.01),
  flag_string("activation1", "relu"),
  flag_string("activation2", "relu"),
  flag_string("activation3", "relu")
)

# Grid to search over
par <- list(
  dropout1 = c(0.3,0.4,0.5,0.6),
  neurons1 = c(32,64,128),
  neurons2 = c(32,64,128),
  neurons3 = c(32,64,128),
  lr = c(0.005,0.0005,0.001,0.01, 0.1),
  activation1 = c("relu", "sigmoid"),
  activation2 = c("relu", "sigmoid"),
  activation3 = c("relu", "sigmoid")
)

# Setting up FNN model with FLAGS
model_fnn <- keras_model_sequential()
model_fnn %>% 
  layer_dense(units = FLAGS$neurons1,
              activation = FLAGS$activation1) %>%
  layer_dense(units = FLAGS$neurons2,
              activation = FLAGS$activation2) %>%
  layer_dropout(FLAGS$dropout1) %>%
  layer_dense(units = FLAGS$neurons3,
              activation = FLAGS$activation3) %>%
  layer_dense(units = length(unique(resp)), 
              activation = 'softmax')

# Setting parameters for FNN model
model_fnn %>% compile(
  optimizer = optimizer_adam(lr = FLAGS$lr), 
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)

# Running model tuning
runs = tuning_run('TrainingFile_TuningExample.R', 
                  runs_dir = '_tuning_example', 
                  sample = 0.005, 
                  flags = par)

# Getting best run
all_runs = ls_runs(order = metric_val_loss, decreasing= F, runs_dir = '_tuning_example')
best_run = ls_runs(order = metric_val_loss, decreasing= F, runs_dir = '_tuning')[1,]

##########################################################################################

# Saving runs results
# saveRDS(all_runs, file = "sim1_tuning.RData")

# # Getting best model run
# run <- training_run('nn_ht.R',flags = list(
#   dropout1 = best_run$flag_dropout1,
#   neurons1 = best_run$flag_neurons1,
#   neurons2 = best_run$flag_neurons2,
#   neurons3 = best_run$flag_neurons3,
#   activation1 = best_run$flag_activation1,
#   activation2 = best_run$flag_activation2,
#   activation3 = best_run$flag_activation3,
#   l2 = best_run$flag_l2,
#   lr = best_run$flag_lr))
# 
# # Loading best model
# best_model <- load_model_hdf5('model.h5')
# 
# # Looking at model parameters
# summary(best_model)
# 
# # Predictions
# test_predictions <- best_model %>% predict(pre_test$data)
# preds = apply(test_predictions, 1, function(x){return(which.max(x))}) - 1
# 
# # Plotting
# confusion_fnn = confusionMatrix(as.factor(preds), as.factor(test_y))

