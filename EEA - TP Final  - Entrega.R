# Inicializamos librerías
library(e1071)
library(caTools)
library(caret)
library(readr)
library(dplyr)
library(tidytext)
library(tidyverse)
library(data.table) 
library(randomForest)
library(caret)
library(e1071)

rm(list = ls())

#Obtenemos los datos
Datos <- read_csv("/home/dell/Descargas/emails.csv")
#https://www.kaggle.com/datasets/balaka18/e-mail-spam-classification-dataset-csv

# Analisis exploratorio previo
# Observamos estructura de datos
str(Datos)

# Chequeamos si hay filas con valores nulos
sum(!complete.cases(Datos))
# No hay, entonces procedemos

# Veo la distribución de los correos según la columna Prediction (No Spam = 0 / Spam = 1)
# Esta va a ser nuestra probabilidad previa. 
Porcentaje <- table(Datos$Prediction)
prop.table(Porcentaje)

# Eliminamos columna Email No con datos de cadena de textos
Datos <- Datos[, !(colnames(Datos) %in% c("Email No."))]

# Verificamos que se haya eliminado la columna
print(Datos)

# Modificamos el nombre de columnas con nombres reservados
Datos <- Datos %>% rename ("in2" = "in")
Datos <- Datos %>% rename ("else2" = "else")
Datos <- Datos %>% rename ("for2" = "for")
Datos <- Datos %>% rename ("if2" = "if")
Datos <- Datos %>% rename ("next2" = "next")
Datos <- Datos %>% rename ("while2" = "while")
Datos <- Datos %>% rename ("function2" = "function")


#Configuro la semilla para poder reproducirlo:
set.seed(32)
# Agregamos columna con ID en reemplazo de Email No.
Datos$id <- 1:nrow(Datos)
#Separamos 70% del dataset como training (entrenamiento) y 30% como test (evaluación) 
train <- Datos %>% dplyr::sample_frac(0.70)
test  <- dplyr::anti_join(Datos, train, by = 'id')

# Factorizamos la columna Prediction para la clasificación que haremos con RF
train$Prediction <- factor(train$Prediction)

# Modelo Naive Bayes sin TF - IDF  
set.seed(32)  # Setting Seed
classifier_nb <- naiveBayes(Prediction ~ ., data = train, laplace = 1)
classifier_nb

# Predicción del modelo de NB
y_pred_nb <- predict(classifier_nb, newdata = test)

# Confusion Matrix para el modelo de NB
cm_nb <- table(test$Prediction, y_pred_nb)
cm_nb

# Evaluación del modelo de NB
confusionMatrix(cm_nb)

# Modelo Random Forest sin TF - IDF  
# Definimos una validación cruzada de 5 folds para encontrar hiperparametros optimos
trControl <- trainControl(method = "cv",
                          number = 5,
                          search = "grid")

## Procedemos a buscar el mejor mtry
set.seed(32)
tuneGrid <- expand.grid(.mtry = c(1: 10))
rf_mtry <- train(Prediction~.,
                 data = train,
                 method = "rf",
                 metric = "Accuracy",
                 tuneGrid = tuneGrid,
                 trControl = trControl,
                 importance = TRUE,
                 nodesize = 10,
                 ntree = 200)
print(rf_mtry)

rf_mtry$bestTune$mtry
max(rf_mtry$results$Accuracy)

best_mtry <- rf_mtry$bestTune$mtry 
best_mtry

## Ahora buscamos el valor óptimo para maxnodes
store_maxnode <- list()
tuneGrid_b <- expand.grid(.mtry = best_mtry)
for (maxnodes in c(1: 15)) {
  set.seed(32)
  rf_maxnode <- train(Prediction~.,
                      data = train,
                      method = "rf",
                      metric = "Accuracy",
                      tuneGrid = tuneGrid_b,
                      trControl = trControl,
                      importance = TRUE,
                      nodesize = 10,
                      maxnodes = maxnodes,
                      ntree = 200)
  key <- toString(maxnodes)
  store_maxnode[[key]] <- rf_maxnode
}
results_node <- resamples(store_maxnode)
summary(results_node)

# Ahora buscamos el ntree optimo
store_maxtrees <- list()
for (ntree in c(200, 250, 300, 350, 400, 450, 500, 550, 600)) {
  set.seed(32)
  rf_maxtrees <- train(Prediction~.,
                       data = train,
                       method = "rf",
                       metric = "Accuracy",
                       tuneGrid = tuneGrid_b,
                       trControl = trControl,
                       importance = TRUE,
                       nodesize = 10,
                       maxnodes = 15,
                       ntree = ntree)
  key <- toString(ntree)
  store_maxtrees[[key]] <- rf_maxtrees
}
results_tree <- resamples(store_maxtrees)
summary(results_tree)

# Ajustamos el modelo
classifier_rf <- train(Prediction~.,
                train,
                method = "rf",
                metric = "Accuracy",
                tuneGrid = tuneGrid_b,
                trControl = trControl,
                importance = TRUE,
                nodesize = 10,
                ntree = 200,
                maxnodes = 15)

y_pred_rf <- predict(classifier_rf, newdata = test)

y_pred_rf

# Confusion Matrix para el modelo de RF
cm_rf <- table(test$Prediction, y_pred_rf)
cm_rf

confusionMatrix(cm_rf)

# Ahora procedemos a procesar el dataser con TF - IDF  
# Primero traspongo el Dataframe
Datos_2 <- Datos %>% 
  pivot_longer(# Columnas a pivotear al formato "largo"
    cols = c(2:3001),       
    # nombre de la nueva columna sobre la que se pivotea el dataframe
    names_to = "Palabras", 
    # nombre de la nueva columna de valores
    values_to = "Cantidad" )
Datos_2

#2 - Agrego total de palabras por documento:
Datos_2_total <- Datos_2 %>%
  
  group_by(`Email No.`, Prediction) %>%
  
  summarize(total = sum(Prediction, Cantidad))

Datos_2_total

Datos_3 <- left_join(Datos_2, Datos_2_total)
Datos_3

#3 - Rankeo por frecuencia
freq_by_rank <- Datos_3 %>%
  
  group_by(`Email No.`, Prediction) %>%
  
  mutate(rank = row_number(),
         
         `term frequency` = Cantidad/total)

freq_by_rank


#4 - Usamos la función bind_tf_idf () para calcular el tf-idf de cada una de las palabras
Datos_3 <- Datos_3 %>%
  
  bind_tf_idf(Palabras, `Email No.`, Cantidad)

Datos_3

# Configuro la semilla para poder reproducirlo:
set.seed(32)
#Creamos columna con ID
Datos_3$id <- 1:nrow(Datos_3)
#Separamos dataset de entrenamiento y evaluación
train_3 <- Datos_3 %>% dplyr::sample_frac(0.70)
test_3  <- dplyr::anti_join(Datos_3, train_3, by = 'id')

# Factorizamos la columna Prediction para la clasificación que haremos con RF
train_3$Prediction <- factor(train_3$Prediction)

#Modelo Naive Bayes con TF IDF
set.seed(32)  # Setting Seed
classifier_nb_3  <- naiveBayes(Prediction ~ ., data = train_3, laplace = 1)
classifier_nb_3

# Prediccion de Naive Bayes con el Dataset de TF IDF
y_pred_nb_3 <- predict(classifier_nb_3, newdata = test_3)

# Confusion Matrix
cm_nb_3 <- table(test_3$Prediction, y_pred_nb_3)
cm_nb_3

# Model Evaluation
confusionMatrix(cm_nb_3)


