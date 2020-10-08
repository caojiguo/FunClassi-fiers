#################################
# FNNs Classification Paper     #
#                               #
# Example 2 code for paper      #
#                               #
# Anonymized                    #
#################################

# Libraries
library(fda)
library(fda.usc)
library(keras)
library(ggplot2)
library(refund)
library(modEvA)
library(future.apply)
library(caret)
library(randomForest)
library(e1071)
library(gbm)
library(stringr)
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
df_train = read.table("fungi/fungi_TRAIN.txt", as.is = T, header = F)
df_test = read.table("fungi/fungi_TEST.txt", as.is = T, header = F)

# Combining data
full_resp = c(df_train[,1], df_test[,1]) - 1
full_df = rbind(df_train[,-1], df_test[,-1])

# Making classification bins
resp = full_resp

# define the time points on which the functional predictor is observed. 
timepts = seq(1, 201, 1)

# define the fourier basis 
nbasis = 39
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
# final_data[,,2] = func_cov_2
# final_data[,,3] = func_cov_3

# fData Object
fdata_obj = fdata(full_df, argvals = timepts, rangeval = c(min(timepts), max(timepts)))

# Choosing fold number
num_folds = 2

# Creating folds
fold_ind = createFolds(resp, k = num_folds)

# numbr of models
num_models = 9

# number of measures
num_measures = 5

# Initializing matrices for results
error_mat_flm = matrix(nrow = num_folds, ncol = num_measures)
error_mat_pc1 = matrix(nrow = num_folds, ncol = num_measures)
error_mat_pc2 = matrix(nrow = num_folds, ncol = num_measures)
error_mat_pc3 = matrix(nrow = num_folds, ncol = num_measures)
error_mat_pls1 = matrix(nrow = num_folds, ncol = num_measures)
error_mat_pls2 = matrix(nrow = num_folds, ncol = num_measures)
error_mat_np = matrix(nrow = num_folds, ncol = num_measures)
error_mat_fnn = matrix(nrow = num_folds, ncol = num_measures)
error_mat_fglm = matrix(nrow = num_folds, ncol = num_measures)
# error_mat_svm = matrix(nrow = num_folds, ncol = num_measures)
# error_mat_nn = matrix(nrow = num_folds, ncol = num_measures)
# error_mat_glm = matrix(nrow = num_folds, ncol = num_measures)
# error_mat_rf = matrix(nrow = num_folds, ncol = num_measures)
# error_mat_gbm = matrix(nrow = num_folds, ncol = num_measures)

# Doing pre-processing of neural networks
if(dim(final_data)[3] > 1){
  # Now, let's pre-process
  pre_dat = FNN_Preprocess(func_cov = final_data,
                           basis_choice = c("fourier", "fourier", "fourier"),
                           num_basis = c(5, 7, 9),
                           domain_range = list(c(min(timepts), max(timepts)), 
                                               c(min(timepts), max(timepts)), 
                                               c(min(timepts), max(timepts))),
                           covariate_scaling = T,
                           raw_data = F)
  
} else {
  
  # Now, let's pre-process
  pre_dat = FNN_Preprocess(func_cov = final_data,
                           basis_choice = c("fourier"),
                           num_basis = c(13),
                           domain_range = list(c(min(timepts), max(timepts))),
                           covariate_scaling = T,
                           raw_data = F)
}

# Functional weights
func_weights = matrix(nrow = num_folds, ncol = 13)


# Looping to get results
for (i in 1:num_folds) {
  
  ################## 
  # Splitting data #
  ##################
  
  # Test and train
  train_x = fdata_obj[-fold_ind[[i]],]
  test_x = fdata_obj[fold_ind[[i]],]
  train_y = resp[-fold_ind[[i]]]
  test_y = resp[fold_ind[[i]]]
  
  # Setting up for FNN
  pre_train = pre_dat$data[-fold_ind[[i]], ]
  pre_test = pre_dat$data[fold_ind[[i]], ]
  
  # Setting up for GLM
  ldata = list("x" = train_x, "df" = as.data.frame(train_y))
  
  ###################################
  # Running usual functional models #
  ###################################
  
  # Functional GLM
  model_fglm = classif.glm(train_y ~ x, data = ldata)
  pred_fglm = predict(model_fglm, new.fdataobj = test_x)
  confusion_gflm = confusionMatrix(as.factor(pred_fglm), as.factor(test_y))
  
  #  Functional Linear Model (Basis)
  l=2^(-2:8)
  func_basis = fregre.basis.cv(train_x, train_y, type.basis = "fourier",
                               lambda=l, type.CV = GCV.S, par.CV = list(trim=0.15))
  pred_basis = round(predict(func_basis[[1]], test_x))
  final_pred_basis = ifelse(pred_basis < min(test_y), min(test_y), ifelse(pred_basis > max(test_y), max(test_y), pred_basis))
  confusion_flm = confusionMatrix(as.factor(final_pred_basis), as.factor(test_y))
  
  # Functional Principal Component Regression (No Penalty)
  func_pc = fregre.pc.cv(train_x, train_y, 8)
  pred_pc = round(predict(func_pc$fregre.pc, test_x))
  final_pred_pc = ifelse(pred_pc < min(test_y), min(test_y), ifelse(pred_pc > max(test_y), max(test_y), pred_pc))
  confusion_fpc = confusionMatrix(as.factor(final_pred_pc), as.factor(test_y))
  
  # Functional Principal Component Regression (2nd Deriv Penalization)
  func_pc2 = fregre.pc.cv(train_x, train_y, 8, lambda=TRUE, P=c(0,0,1))
  pred_pc2 = round(predict(func_pc2$fregre.pc, test_x))
  final_pred_pc2 = ifelse(pred_pc2 < min(test_y), min(test_y), ifelse(pred_pc2 > max(test_y), max(test_y), pred_pc2))
  confusion_fpc2 = confusionMatrix(as.factor(final_pred_pc2), as.factor(test_y))
  
  # Functional Principal Component Regression (Ridge Regression)
  func_pc3 = fregre.pc.cv(train_x, train_y, 1:8, lambda=TRUE, P=1)
  pred_pc3 = round(predict(func_pc3$fregre.pc, test_x))
  final_pred_pc3 = ifelse(pred_pc3 < min(test_y), min(test_y), ifelse(pred_pc3 > max(test_y), max(test_y), pred_pc3))
  confusion_fpc3 = confusionMatrix(as.factor(final_pred_pc3), as.factor(test_y))
  
  # Functional Partial Least Squares Regression (No Penalty)
  func_pls = fregre.pls(train_x, train_y, 1:8)
  pred_pls = round(predict(func_pls, test_x))
  final_pred_pls = ifelse(pred_pls < min(test_y), min(test_y), ifelse(pred_pls > max(test_y), max(test_y), pred_pls))
  confusion_pls = confusionMatrix(as.factor(final_pred_pls), as.factor(test_y))
  
  # Functional Partial Least Squares Regression (2nd Deriv Penalization)
  func_pls2 = fregre.pls.cv(train_x, train_y, 3, lambda = 1:2, P=c(0,0,1))
  pred_pls2 = round(predict(func_pls2$fregre.pls, test_x))
  final_pred_pls2 = ifelse(pred_pls2 < min(test_y), min(test_y), ifelse(pred_pls2 > max(test_y), max(test_y), pred_pls2))
  confusion_pls2 = confusionMatrix(as.factor(final_pred_pls2), as.factor(test_y))
  
  # Functional Non-Parametric Regression
  func_np = fregre.np(train_x, train_y, Ker = AKer.tri, metric = semimetric.deriv)
  pred_np = round(predict(func_np, test_x))
  final_pred_np = ifelse(pred_np < min(test_y), min(test_y), ifelse(pred_np > max(test_y), max(test_y), pred_np))
  confusion_np = confusionMatrix(as.factor(final_pred_np), as.factor(test_y))
  
  print("Done: Functional Method Modelling")

  #####################################
  # Running Functional Neural Network #
  #####################################
  
  # Setting seeds
  set.seed(i)
  use_session_with_seed(
    i,
    disable_gpu = F,
    disable_parallel_cpu = F,
    quiet = T
  )
  
  # Setting up FNN model
  model_fnn <- keras_model_sequential()
  model_fnn %>% 
    layer_dense(units = 128,
                activation = "relu") %>%
    layer_dense(units = 64,
                activation = "relu") %>%
    layer_dropout(0.4) %>%
    layer_dense(units = 128,
                activation = "sigmoid") %>%
    layer_dense(units = length(unique(resp)), activation = 'softmax')
  
  
  # Setting parameters for FNN model
  model_fnn %>% compile(
    optimizer = optimizer_adam(lr = 5e-03), 
    loss = 'sparse_categorical_crossentropy',
    metrics = c('accuracy')
  )
  
  # Early stopping
  early_stop <- callback_early_stopping(monitor = "val_loss", patience = 15)
  
  # Training FNN model
  model_fnn %>% fit(pre_train, 
                    train_y, 
                    epochs = 300,  
                    validation_split = 0.2,
                    callbacks = list(early_stop),
                    verbose = 0)
  
  # Predictions
  test_predictions <- model_fnn %>% predict(pre_test)
  preds_fnn = apply(test_predictions, 1, function(x){return(which.max(x))}) - 1
  
  # Plotting
  confusion_fnn = confusionMatrix(as.factor(preds_fnn), as.factor(test_y))
  
  # Storing weights
  func_weights[i,] = rowMeans(get_weights(model_fnn)[[1]])
  
  print("Done: FNN Modelling")
  
  ###################
  # Storing Results #
  ###################
  
  error_mat_flm[i, ] = c(confusion_flm$overall[1], 
                         mean(confusion_flm$byClass[,1], na.rm = T),
                         mean(confusion_flm$byClass[,2], na.rm = T), 
                         mean(confusion_flm$byClass[,3], na.rm = T),
                         mean(confusion_flm$byClass[,4], na.rm = T))
  error_mat_pc1[i, ] = c(confusion_fpc$overall[1], 
                         mean(confusion_fpc$byClass[,1], na.rm = T),
                         mean(confusion_fpc$byClass[,2], na.rm = T), 
                         mean(confusion_fpc$byClass[,3], na.rm = T),
                         mean(confusion_fpc$byClass[,4], na.rm = T))
  error_mat_pc2[i, ] = c(confusion_fpc2$overall[1], 
                         mean(confusion_fpc2$byClass[,1], na.rm = T),
                         mean(confusion_fpc2$byClass[,2], na.rm = T), 
                         mean(confusion_fpc2$byClass[,3], na.rm = T),
                         mean(confusion_fpc2$byClass[,4], na.rm = T))
  error_mat_pc3[i, ] = c(confusion_fpc3$overall[1], 
                         mean(confusion_fpc3$byClass[,1], na.rm = T),
                         mean(confusion_fpc3$byClass[,2], na.rm = T), 
                         mean(confusion_fpc3$byClass[,3], na.rm = T),
                         mean(confusion_fpc3$byClass[,4], na.rm = T))
  error_mat_pls1[i, ] = c(confusion_pls$overall[1], 
                          mean(confusion_pls$byClass[,1], na.rm = T),
                          mean(confusion_pls$byClass[,2], na.rm = T), 
                          mean(confusion_pls$byClass[,3], na.rm = T),
                          mean(confusion_pls$byClass[,4], na.rm = T))
  error_mat_pls2[i, ] = c(confusion_pls2$overall[1], 
                          mean(confusion_pls2$byClass[,1], na.rm = T),
                          mean(confusion_pls2$byClass[,2], na.rm = T), 
                          mean(confusion_pls2$byClass[,3], na.rm = T),
                          mean(confusion_pls2$byClass[,4], na.rm = T))
  error_mat_np[i, ] = c(confusion_np$overall[1], 
                        mean(confusion_np$byClass[,1], na.rm = T),
                        mean(confusion_np$byClass[,2], na.rm = T), 
                        mean(confusion_np$byClass[,3], na.rm = T),
                        mean(confusion_np$byClass[,4], na.rm = T))
  error_mat_fnn[i, ] = c(confusion_fnn$overall[1], 
                         mean(confusion_fnn$byClass[,1], na.rm = T),
                         mean(confusion_fnn$byClass[,2], na.rm = T), 
                         mean(confusion_fnn$byClass[,3], na.rm = T),
                         mean(confusion_fnn$byClass[,4], na.rm = T))
  error_mat_fglm[i, ] = c(confusion_gflm$overall[1], 
                          mean(confusion_gflm$byClass[,1], na.rm = T),
                          mean(confusion_gflm$byClass[,2], na.rm = T), 
                          mean(confusion_gflm$byClass[,3], na.rm = T),
                          mean(confusion_gflm$byClass[,4], na.rm = T))
  
  # Resetting things
  K <- backend()
  K$clear_session()
  options(warn=-1)
  
  # Printing iteration number
  print(paste0("Done Iteration: ", i))
  
}

# Initializing final table: average of errors
Final_Table = matrix(nrow = num_models, ncol = num_measures + 1)

# Collecting errors
Final_Table[1, ] = c(colMeans(error_mat_flm, na.rm = T), sd(error_mat_flm[,1]))
Final_Table[2, ] = c(colMeans(error_mat_np, na.rm = T), sd(error_mat_np[,1]))
Final_Table[3, ] = c(colMeans(error_mat_pc1, na.rm = T), sd(error_mat_pc1[,1]))
Final_Table[4, ] = c(colMeans(error_mat_pc2, na.rm = T), sd(error_mat_pc2[,1]))
Final_Table[5, ] = c(colMeans(error_mat_pc3, na.rm = T), sd(error_mat_pc3[,1]))
Final_Table[6, ] = c(colMeans(error_mat_pls1, na.rm = T), sd(error_mat_pls1[,1]))
Final_Table[7, ] = c(colMeans(error_mat_pls2, na.rm = T), sd(error_mat_pls2[,1]))
Final_Table[8, ] = c(colMeans(error_mat_fglm, na.rm = T), sd(error_mat_fglm[,1]))
Final_Table[9, ] = c(colMeans(error_mat_fnn, na.rm = T), sd(error_mat_fnn[,1]))

# Editing names
rownames(Final_Table) = c("FLM", "FNP", "FPC_1", "FPC_2", "FPC_3", "FPLS_1", "FPLS_2",
                          "fGLM", "FNN")
colnames(Final_Table) = c("Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "SD_Error")

# Looking at results
Final_Table

##################### Functional Weights #######################

#######################################

### Functional Linear Model (Basis) ###

# Setting up grid
# l=2^(-4:10)
# 
# # Running functional linear model
# func_basis = fregre.basis.cv(fdata_obj, 
#                              resp, 
#                              type.basis = "fourier",
#                              lambda=l, 
#                              type.CV = GCV.S, 
#                              par.CV = list(trim=0.15))
# 
# # Pulling out the coefficients
# coefficients_lm = func_basis$fregre.basis$coefficients
# 
# # Setting up data set
# beta_coef_lm <- data.frame(time = timepts, 
#                            beta_evals = final_beta_fourier(timepts, scale(c(coefficients_lm[,1])), range = c(min(timepts), max(timepts))))

#######################################

# Running FNN for weather
# fnn_final = FNN(resp = resp, 
#                 func_cov = final_data, 
#                 scalar_cov = NULL,
#                 basis_choice = c("fourier", "fourier", "fourier"), 
#                 num_basis = c(5, 7, 9),
#                 hidden_layers = 2,
#                 neurons_per_layer = c(16, 8),
#                 activations_in_layers = c("relu", "sigmoid"),
#                 domain_range = list(c(min(timepts), max(timepts)), 
#                                     c(min(timepts), max(timepts)), 
#                                     c(min(timepts), max(timepts))),
#                 epochs = 250,
#                 output_size = 1,
#                 loss_choice = "mse",
#                 metric_choice = list("mean_squared_error"),
#                 val_split = 0.2,
#                 patience_param = 25,
#                 learn_rate = 0.05,
#                 early_stop = T,
#                 print_info = F)
# 
# # Getting the FNC
# coefficients_fnn = rowMeans(get_weights(fnn_final$model)[[1]])[1:5]

# # Setting up data set
# beta_coef_fnn <- data.frame(time = timepts, 
#                             beta_evals = final_beta_fourier(timepts, scale(colMeans(func_weights)), range = c(min(timepts), max(timepts))))
# 
# #### Putting Together #####
# 
# # Getting range
# timepts = seq(75, 100, length.out = 201)
# 
# # Plotting
# beta_coef_fnn %>% 
#   ggplot(aes(x = timepts, y = beta_evals, color = "red")) +
#   geom_line(size = 1.5) +
#   geom_line(data = beta_coef_lm, 
#             aes(x = timepts, y = beta_evals, color = "black"),
#             size = 1.2,
#             linetype = "dashed") + 
#   theme_bw() +
#   xlab("Time") +
#   ylab("beta(t)") +
#   theme(plot.title = element_text(hjust = 0.5)) +
#   theme(axis.text=element_text(size=14, face = "bold"),
#         axis.title=element_text(size=14,face="bold")) +
#   scale_colour_manual(name = 'Model: ', 
#                       values =c('black'='black','red'='red'), 
#                       labels = c('Functional Linear Model', 'Functional Neural Network')) +
#   theme(legend.background = element_rect(fill="lightblue",
#                                          size=0.5, linetype="solid", 
#                                          colour ="darkblue"),
#         legend.position = "bottom",
#         legend.title = element_text(size = 14),
#         legend.text = element_text(size = 12))
# 
# 
# beta_coef_fnn %>% 
#   ggplot(aes(x = timepts, y = beta_evals, color = "red")) +
#   geom_line(size = 1.5) +
#   geom_line(data = beta_coef_lm, 
#             aes(x = timepts, y = beta_evals, color = "black"),
#             size = 1.2,
#             linetype = "dashed") + 
#   theme_bw() +
#   xlab("Temperature") +
#   ylab("beta(C)") +
#   theme(plot.title = element_text(hjust = 0.5)) +
#   theme(axis.text=element_text(size=14, face = "bold"),
#         axis.title=element_text(size=14,face="bold")) +
#   scale_colour_manual(name = 'Model: ', 
#                       values =c('black'='black','red'='red'), 
#                       labels = c('Functional Linear Model', 'Functional Neural Network')) +
#   theme(legend.title = element_text(size = 14),
#         legend.text = element_text(size = 12),
#         legend.position = "None")

