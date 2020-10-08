#################################
# FNNs Classification Paper     #
#                               #
# Accompanying File - Tuning    #
#                               #
# Anonymized                    #
#################################

# Flags
FLAGS <- flags(
  flag_numeric('dropout1', 0.3),
  flag_integer('neurons1', 128),
  flag_integer('neurons2', 128),
  flag_integer('neurons3', 128),
  flag_numeric('lr', 0.001),
  flag_string("activation1", "relu"),
  flag_string("activation2", "relu"),
  flag_string("activation3", "relu")
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
  layer_dense(units = length(unique(resp)), activation = 'softmax')

# Setting parameters for FNN model
model_fnn %>% compile(
  optimizer = optimizer_adam(lr = FLAGS$lr), 
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)

# Early stopping
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 15)

# Training FNN model
model_fnn %>% fit(pre_train$data, 
                  train_y, 
                  epochs = 300,  
                  validation_split = 0.2,
                  callbacks = list(early_stop),
                  verbose = 0)

# Predictions
test_predictions <- model_fnn %>% predict(pre_test$data)
preds = apply(test_predictions, 1, function(x){return(which.max(x))}) - 1

# Plotting
confusion_fnn = confusionMatrix(as.factor(preds), as.factor(test_y))
accuracy = confusion_fnn$overall[1]

# Saving model
save_model_hdf5(model_fnn, 'model.h5')

# Printing accuracy
cat('Test accuracy:', accuracy, '\n')

# Clearing backend
K <- backend()
K$clear_session()