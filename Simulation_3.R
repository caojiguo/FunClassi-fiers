#################################
# FNNs Classification Paper     #
#                               #
# Sim 3 code for paper          #
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
library(funData)
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

# Initializing information
num_sims = 150
num_models = 3
num_folds = 2
num_obs = 300
final_results = matrix(nrow = num_sims, ncol = num_models)
final_sensitivity = matrix(nrow = num_sims, ncol = num_models)
final_specificity = matrix(nrow = num_sims, ncol = num_models)
final_PPV = matrix(nrow = num_sims, ncol = num_models)
final_NPV = matrix(nrow = num_sims, ncol = num_models)

# Initializing parameters + data matrix
data_mat = matrix(nrow = num_obs, ncol = 100)
continuum_points = seq(0, 1, length.out = 100)
resp_vec = c()

# Running simulations
for (j in 1:num_sims) {
  
  ##########################################################################
  
  # Creating Functional Observations #
  
  # generating observations
  for (k in 1:num_obs) {
    
    # Random values
    ran_num = runif(1)
    
    # Parameters for particular obs
    # a = rnorm(1)
    # b = rexp(1)
    epsilon = runif(1)
    
    # Storing values
    if(ran_num > 0.5){
      weighted_fourier <- simMultiFunData(type = "weighted",
                                          argvals = list(list(seq(0, 1, length.out = 100))),
                                          M = c(5,5), eFunType = c("Fourier"), eValType = "linear", N = 1)
      data_mat[k, ] = sin(weighted_fourier$simData[[1]]@X*(2*pi)) + epsilon
      resp_vec[k] = 0
      
    } else {
      weighted_poly <- simMultiFunData(type = "weighted",
                                       argvals = list(list(seq(0, 1, length.out = 100))),
                                       M = c(5,5), eFunType = c("Fourier"), eValType = "linear", N = 1)
      data_mat[k, ] = (3/5)*sin(weighted_poly$simData[[1]]@X*(2*pi)) + epsilon
      resp_vec[k] = 1
      
    }
    
  }
  
  # Getting data in one form data
  full_resp = resp_vec
  full_df = as.data.frame(data_mat)
  
  # Making classification bins
  resp = full_resp
  
  ##########################################################################
  
  # Running Models #
  
  # define the time points on which the functional predictor is observed. 
  timepts = continuum_points
  
  # define the fourier basis 
  nbasis = 35
  spline_basis = create.fourier.basis(c(min(timepts), max(timepts)), nbasis)
  
  # convert the functional predictor into a fda object
  fd =  Data2fd(timepts, t(full_df), spline_basis)
  deriv1 = deriv.fd(fd)
  deriv2 = deriv.fd(deriv1)
  # plot(fd[which(resp == 1),])
  # plot(fd[which(resp == 0),])
  
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
  
  # Creating folds
  fold_ind = createFolds(resp, k = num_folds)
  
  # number of measures
  num_measures = 5
  
  # Initializing matrices for results
  error_mat_flm = matrix(nrow = num_folds, ncol = num_measures)
  error_mat_fnn = matrix(nrow = num_folds, ncol = num_measures)
  error_mat_nn = matrix(nrow = num_folds, ncol = num_measures)
  
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
                             num_basis = c(21),
                             domain_range = list(c(min(timepts), max(timepts))),
                             covariate_scaling = T,
                             raw_data = F)
  }
  
  
  # Looping to get results
  for (i in 1:1) {
    
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
  
    ###################################
    # Running usual functional models #
    ###################################
    
    # Functional Linear Model (Basis)
    l=2^(-2:8)
    func_basis = fregre.basis.cv(train_x, train_y, type.basis = "fourier",
                                 lambda=l, type.CV = GCV.S, par.CV = list(trim=0.15))
    pred_basis = round(predict(func_basis[[1]], test_x))
    final_pred_basis = ifelse(pred_basis < min(test_y), min(test_y), ifelse(pred_basis > max(test_y), max(test_y), pred_basis))
    confusion_flm = confusionMatrix(as.factor(final_pred_basis), as.factor(test_y))
    
    
    ###################################
    # Running multivariate models     #
    ###################################
    
      # Setting up MV data
    MV_train = as.data.frame(full_df[-fold_ind[[i]],])
    MV_test = as.data.frame(full_df[fold_ind[[i]],])
    train_y = resp[-fold_ind[[i]]]
    test_y = resp[fold_ind[[i]]]
    
    # Running NN
    
    # Initializing
    min_error = 99999
    
    # random split
    train_split = sample(1:nrow(MV_train), floor(0.75*nrow(MV_train)))
    
    # Setting up FNN model
    for(u in 1:10){
      
      # setting up model
      model_nn <- keras_model_sequential()
      model_nn %>% 
        layer_dense(units = 64, activation = 'relu') %>%
        layer_dense(units = 64, activation = 'relu') %>%
        layer_dense(units = 32, activation = 'relu') %>%
        layer_dense(units = length(unique(resp)), activation = 'softmax')
      
      # Setting parameters for NN model
      model_nn %>% compile(
        optimizer = optimizer_adam(lr = 0.0085), 
        loss = 'sparse_categorical_crossentropy',
        metrics = c('accuracy')
      )
      
      # Early stopping
      early_stop <- callback_early_stopping(monitor = "val_loss", patience = 15)
      
      # Training FNN model
      model_nn %>% fit(as.matrix(MV_train[train_split,]), 
                       train_y[train_split], 
                       epochs = 250,  
                       validation_split = 0.2,
                       callbacks = list(early_stop),
                       verbose = 0)
      
      # Predictions
      test_predictions <- model_nn %>% predict(as.matrix(MV_train[-train_split,]))
      preds_train = apply(test_predictions, 1, function(x){return(which.max(x))}) - 1
      
      # Plotting
      confusion_nn_train = confusionMatrix(as.factor(preds_train), 
                                           as.factor(train_y[-train_split]))
      
      # Checking error
      if(confusion_nn_train$overall[1] < min_error){
        
        # Predictions
        test_predictions <- model_nn %>% predict(as.matrix(MV_test))
        preds = apply(test_predictions, 1, function(x){return(which.max(x))}) - 1
        
        # Plotting
        confusion_nn = confusionMatrix(as.factor(preds), as.factor(test_y))
        
        # Updating error
        min_error = confusion_nn_train$overall[1]
        
      }
      
    }
    
    
    #####################################
    # Running Functional Neural Network #
    #####################################
    
    # Setting up FNN model
    model_fnn <- keras_model_sequential()
    model_fnn %>% 
      layer_dense(units = 128, activation = 'relu') %>%
      layer_dense(units = 32, activation = 'relu') %>%
      layer_dense(units = length(unique(resp)), activation = 'softmax')
    
    # Setting parameters for FNN model
    model_fnn %>% compile(
      optimizer = optimizer_adam(lr = 0.01), 
      loss = 'sparse_categorical_crossentropy',
      metrics = c('accuracy')
    )
    
    # Early stopping
    early_stop <- callback_early_stopping(monitor = "val_loss", patience = 15)
    
    # Training FNN model
    model_fnn %>% fit(pre_train, 
                      train_y, 
                      epochs = 250,  
                      validation_split = 0.2,
                      callbacks = list(early_stop),
                      verbose = 0)
    
    # Predictions
    test_predictions <- model_fnn %>% predict(pre_test)
    preds = apply(test_predictions, 1, function(x){return(which.max(x))}) - 1
    
    # Plotting
    confusion_fnn = confusionMatrix(as.factor(preds), as.factor(test_y))
    
    # print("Done: FNN Modelling")
    
    ###################
    # Storing Results #
    ###################
    
    error_mat_flm[i, ] = c(confusion_flm$overall[1], confusion_flm$byClass[c(1, 2, 3, 4)])
    error_mat_fnn[i, ] = c(confusion_fnn$overall[1], confusion_fnn$byClass[c(1, 2, 3, 4)])
    error_mat_nn[i, ] = c(confusion_nn$overall[1], confusion_nn$byClass[c(1, 2, 3, 4)])
    
    # Resetting things
    K <- backend()
    K$clear_session()
    options(warn=-1)
    
    # Printing iteration number
    # print(paste0("Done Iteration: ", i))
    
  }
  
  # Initializing final table: average of errors
  Final_Table = matrix(nrow = num_models, ncol = num_measures + 1)
  
  # Collecting errors
  Final_Table[1, ] = c(colMeans(error_mat_flm, na.rm = T), sd(error_mat_flm[,1]))
  Final_Table[2, ] = c(colMeans(error_mat_nn, na.rm = T), sd(error_mat_nn[,1]))
  Final_Table[3, ] = c(colMeans(error_mat_fnn, na.rm = T), sd(error_mat_fnn[,1]))
  
  # Editing names
  rownames(Final_Table) = c("FLM", "NN", "FNN")
  colnames(Final_Table) = c("Error", "Sensitivity", "Specificity", "PPV", "NPV", "SD_Error")
  
  # Storing Results
  final_results[j, ] = 1 - Final_Table[, 1]
  final_sensitivity[j, ] = Final_Table[, 2]
  final_specificity[j, ] = Final_Table[, 3]
  final_PPV[j, ] = Final_Table[, 4]
  final_NPV[j, ] = Final_Table[, 5]
  
  # Printing
  print(paste0("Done Replication Number: ", j))
  
}

# Saving table
write.table(final_results, file="sim3Pred_class.csv", row.names = F)

# Getting minimums
# mspe_div_mins = apply(final_results, 1, function(x){return(min(x))})

# Initializing
# mspe_div = matrix(nrow = nrow(final_results), ncol = ncol(final_results))

# Getting relative measures
# for (i in 1:num_sims) {
#   mspe_div[i, ] = final_results[i,]/mspe_div_mins[i]
# }

# names
# colnames(mspe_div) = c("FLM", "NN", "FNN")
colnames(final_results) = c("FLM", "NN", "FNN")

# Creating relative boxplots

# turning into df
# df_MSPE <- data.frame(mspe_div)
df_MSPE <- data.frame(final_results)

# Creating boxplots
# plot1_rel <- ggplot(stack(df_MSPE), aes(x = ind, y = values)) +
#   geom_boxplot(fill='#A4A4A4', color="darkgreen") + 
#   theme_bw() + 
#   xlab("") +
#   ylab("Simulation: 1\nRelative MSPE") +
#   theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
#   scale_y_continuous(limits = c(0, 10)) +
#   theme(axis.text=element_text(size=14, face= "bold"),
#         axis.title=element_text(size=14, face="bold")) +
#   geom_hline(yintercept = 1, linetype = "dashed")

plot3 <- ggplot(stack(df_MSPE), aes(x = ind, y = values)) +
  geom_boxplot(fill='white', color="black") + 
  theme_bw() + 
  xlab("") +
  ylab("Simulation: 3\nMSPE") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  scale_y_continuous(limits = c(0, 1)) +
  theme(axis.text=element_text(size=14, face= "bold"),
        axis.title=element_text(size=14, face="bold")) +
  geom_hline(yintercept = 0, linetype = "dashed")

# Creating final table
final_sim3_results = data.frame(Error = colMeans(final_results, na.rm = T),
                                Sensitivity = colMeans(final_sensitivity, na.rm = T),
                                Specificity = colMeans(final_specificity, na.rm = T),
                                PPV = colMeans(final_PPV, na.rm = T),
                                NPV = colMeans(final_NPV, na.rm = T),
                                SD_Error = apply(final_results, 2, sd, na.rm = T))

# Saving table
write.table(final_sim3_results, file="AverageSim3Pred_class.csv", row.names = F)
