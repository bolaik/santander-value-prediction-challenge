library(tidyverse)
library(xgboost)
library(magrittr)
library(caret)
library(h2o)

set.seed(0)

#---------------------------
cat("Loading data...\n")

tr <- read_csv("../input/train.csv") %>% select(-ID)
te <- read_csv("../input/test.csv") %>% select(-ID)

target <- log1p(tr$target)
tr$target <- NULL

#---------------------------
cat("Removing constant & duplicate features...\n")

zero_var <- names(tr)[sapply(tr, var) == 0]
tr %<>% select(-one_of(zero_var))
te %<>% select(-one_of(zero_var))

dup_var <- names(tr)[duplicated(lapply(tr, c))]
tr %<>% select(-one_of(dup_var))
te %<>% select(-one_of(dup_var))

cor_var <- model.matrix(~.-1, tr) %>% 
  cor(method = "spearman") %>% 
  findCorrelation(cutoff = 0.98, names = TRUE) %>% 
  str_replace_all("`", "")
tr %<>% select(-one_of(cor_var))
te %<>% select(-one_of(cor_var))

#---------------------------
cat("Creating AEC features...\n")

h2o.no_progress()
h2o.init(nthreads = 4, max_mem_size = "9G")

tr_h2o <- as.h2o(tr)
te_h2o <- as.h2o(te)

n_comp <- 6
m_aec <- h2o.deeplearning(training_frame = h2o.rbind(tr_h2o, te_h2o),
                          x = 1:ncol(tr_h2o),
                          autoencoder = T,
                          activation="Tanh",
                          reproducible = TRUE,
                          seed = 0,
                          sparse = T,
                          hidden = c(32, n_comp, 32),
                          max_w2 = 5,
                          epochs = 10)

tr <- cbind(tr, as.data.frame(h2o.deepfeatures(m_aec, tr_h2o, layer = 2)))
te <- cbind(te, as.data.frame(h2o.deepfeatures(m_aec, te_h2o, layer = 2)))

h2o.shutdown(prompt = FALSE)

#---------------------------
cat("Preparing data...\n")

dtest <- xgb.DMatrix(data = data.matrix(te))
tri <- createDataPartition(target, p = 0.9, list = F) %>% c()
dtrain <- xgb.DMatrix(data = data.matrix(tr[tri, ]), label = target[tri])
dval <- xgb.DMatrix(data = data.matrix(tr[-tri, ]), label = target[-tri])
cols <- names(tr)
rm(tr, te, target, tri, zero_var, dup_var, cor_var, m_aec, tr_h2o, te_h2o)
gc()

#---------------------------
cat("Training model...\n")

p <- list(objective = "reg:linear",
          booster = "gbtree",
          eval_metric = "rmse",
          nthread = 4,
          eta = 0.007,
          max_depth = 22,
          min_child_weight = 57,
          gamma = 1.444155,
          subsample = 0.6731153,
          colsample_bytree = 0.05427895,
          colsample_bylevel = 0.70762806,
          alpha = 0,
          lambda = 0,
          nrounds = 7000)

m_xgb <- xgb.train(p, dtrain, p$nrounds, list(val = dval), print_every_n = 100, early_stopping_rounds = 700)

xgb.importance(cols, model = m_xgb) %>% 
  xgb.plot.importance(top_n = 30)

#---------------------------
cat("Making submission file...\n")

read_csv("../input/sample_submission.csv") %>%  
  mutate(target = expm1(predict(m_xgb, dtest))) %>%
  write_csv(paste0("xgb_aec_", round(m_xgb$best_score, 5), ".csv"))