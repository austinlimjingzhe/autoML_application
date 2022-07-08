library(shiny)
library(shinythemes)
library(DT)
library(DataExplorer)
library(plyr)
library(tidyverse)
library(RColorBrewer)
library(lattice)
library(caret)
library(Matrix)
library(glmnet)
library(kernlab)
library(naivebayes)
library(rpart)
library(nnet)
library(randomForest)
library(gbm)
library(xgboost)
library(ModelMetrics)
library(MLmetrics)
library(yardstick)
library(iml)
library(lubridate)
library(e1071)

#supporting function
get_param_string<-function(model){
  if(model$method=="glm"){return("none")}
  params=""
  for(i in 1:ncol(model$bestTune)){params=paste0(params,names(model$bestTune)[i],"=",round(model$bestTune[names(model$bestTune)[i]],5),", ")}
  params=gsub(", $","",params)
  return(params)
}

ui<-fluidPage(theme = shinytheme("cerulean"),
              navbarPage("autoML",
                         tabPanel("Upload Dataset",
                                  sidebarPanel(fileInput("upload", "Choose a file", buttonLabel = "Upload...",accept = ".csv"),
                                               checkboxInput("header", "Header", T),
                                               checkboxInput("stringFactor","Strings As Factors",T),
                                               radioButtons("sep", "Separator",
                                                            choices = c(Comma = ",",
                                                                        Semicolon = ";",
                                                                        Tab = "\t"),selected = ",")),
                                  mainPanel(dataTableOutput("dataset"),
                                            hr(),
                                            plotOutput("structure"),
                                            uiOutput("dataDensity"),
                                            uiOutput("dataBars"),
                                            plotOutput("correlations"),
                                            style="overflow-x: scroll")),
                         tabPanel("Modelling",
                                  sidebarPanel(div(strong("Control Panel"),style="text-align:center"),
                                               hr(),
                                               uiOutput("target"),
                                               uiOutput("modelType"),
                                               numericInput("traintestSplit","Enter training split percentage (0-1):",
                                                            min = 0,max = 1,value = 0.7,step = 0.1),
                                               numericInput("seed","Random Seed:",
                                                            min = 0,max = 1000000,value = 1),
                                               uiOutput("metricsOptimize"),
                                               hr(),
                                               fluidRow(column(6,checkboxInput("hyperTune","Hypertune Parameters:*",F)),
                                                        column(6,checkboxInput("crossVal","Cross Validate:",T))),
                                               uiOutput("hyperParam1"),
                                               uiOutput("hyperParam2"),
                                               uiOutput("hyperParam3"),
                                               hr(),
                                               uiOutput("cvConsole"),
                                               actionButton("build","Build Model",width = "100%",icon=icon("trowel-bricks")),
                                               p(),
                                               p("*If unchecked, default model will use a default grid search for best parameters")),
                                  mainPanel(textOutput("metric1"),
                                            plotOutput("table1"),
                                            plotOutput("table2"),
                                            plotOutput("table3"),
                                            plotOutput("table4"))),
                         tabPanel("Explanations",
                                  sidebarPanel(div(strong("Control Panel"),style="text-align:center"),
                                               hr(),
                                               uiOutput("modelSelect"),
                                               h2("Global Interpretation"),
                                               radioButtons("globalFeatures","Select a Method to Visualize:",
                                                            choices = c("Feature Importance"="imp",
                                                                        "Feature Effects"="eff",
                                                                        "Feature Interactions"="inter")),
                                               uiOutput("featureParam"),
                                               h2("Local Interpretation"),
                                               radioButtons("localFramework","Select a Framework:",
                                                            choices = c("SHAP"="shap","LIME"="lime")),
                                               uiOutput("localParam"),
                                               actionButton("explain","Understand",width="100%",icon=icon("person-chalkboard"))),
                                  mainPanel(plotOutput("globalPlot"),
                                            plotOutput("localPlot"))),
                         tabPanel("Leaderboards",
                                  fluidRow(column(4,p(" "),
                                                  actionButton("resetBoard","Reset Leaderboard",width = "200px")),
                                           column(4,uiOutput("boardFilter"))),
                                  hr(),
                                  DT::dataTableOutput("leaderboard")),
                         tabPanel("Predictions",
                                  sidebarPanel(div(strong("Control Panel"),style="text-align:center"),
                                               hr(),
                                               uiOutput("modelSelect2"),
                                               radioButtons("predictType","Choose a Prediction",
                                                            choices = c("Upload New Data"="newdata",
                                                                        "Make Single Prediction"="singlepred")),
                                               uiOutput("predictActions")),
                                  mainPanel(dataTableOutput("predResults")))))

server <- function(input, output, session) {
  
  # Upload Dataset
  #####
  v<-reactiveValues(df=NULL)
  observeEvent(input$upload, {
    #reads and loads the uploaded file into the reactiveValues container
    v$df <- read.csv(input$upload$datapath,
                     header = input$header,
                     sep = input$sep,
                     stringsAsFactors = input$stringFactor)
    #displays the dataset in a datatable
    output$dataset <- renderDataTable({
      return(v$df)
    })
    #displays the class and missingness of the data
    output$structure<-renderPlot({
      df<-profile_missing(v$df)
      df$class=sapply(v$df,class)
      df$class=ordered(df$class,levels=unique(df$class))
      
      gg<-ggplot(df,aes(y=reorder(feature,as.numeric(class)),x=num_missing,fill=class))+
        geom_bar(stat = "identity")+
        ylab("Feature")+
        xlab("Missing rows")+
        ggtitle("Missingness Plot")+
        geom_label(aes(label=round(pct_missing,2)))+
        scale_fill_brewer(palette="Pastel1")+
        theme(legend.position="bottom",
              plot.title = element_text(hjust = 0.5))
      return(gg)
    })
    #displays the correlation heatmap between all variables, discrete variables with more than 20 levels are dropped
    output$correlations<-renderPlot({
      req(v$df)
      plot_correlation(v$df,
                       title="Correlation between Variables",
                       theme_config = list(plot.title = element_text(hjust = 0.5)))
    })
    #display the distributions of data, since there may be multiple pages for DataExplorer to print, 
    #find the number of pages first, then use lapply() to generate the appropriate number of outputs to display
    explore<-split_columns(v$df,binary_as_factor = T)
    pagesDensity<-ceiling(explore$num_continuous/16)
    pagesBars<-ceiling(explore$num_discrete/9)
    
    output$dataDensity<-renderUI({
      lapply(1:pagesDensity, function(i){
        desityId <- paste0("densityplot_", i)
        plotOutput(desityId)
        
        output[[desityId]]<-renderPlot({
          pagenum<-paste0("page_",i)
          plottitle<-paste("Density Plots Page",i)
          return(plot_density(v$df,title = plottitle, theme_config=list("plot.title" = element_text(hjust = 0.5)))[[pagenum]])
        })
      })
    })
    output$dataBars<-renderUI({
      lapply(1:pagesBars, function(i){
        barId <- paste0("barplot_", i)
        plotOutput(barId)
        
        output[[barId]]<-renderPlot({
          pagenum<-paste0("page_",i)
          plottitle<-paste("Bar Plots Page",i)
          return(plot_bar(v$df,title = plottitle, theme_config=list("plot.title" = element_text(hjust = 0.5)))[[pagenum]])
        })
      })
    })
  })
  ##### 
  #end of section
  
  # Leaderboards
  #####
  #generate a blank leaderboard and a copy to enable resets 
  #j is a running count of the number of models built
  mods=reactiveValues(j=1)
  board<-data.frame(model=character(),type=character(),target=character(),`train-test ratio`=numeric(),seed=numeric(),parameters=character(),metric=character(),value=numeric())
  boardCopy<-data.table::copy(board)
  lb<-reactiveValues(df=board)
  #filter is available to filter results by a particular metric
  output$boardFilter<-renderUI({
    req(lb$df)
    selectizeInput("metricFilter",label="Filter by:",multiple=T,
                   choices=unique(lb$df[["metric"]]),selected=unique(lb$df[["metric"]]))
  })
  #displays the leaderboard
  output$leaderboard<-renderDataTable({
    lb$df%>%
      filter(metric %in% input$metricFilter)
  })
  #clears the leaderboard
  observeEvent(input$resetBoard,{
    lb$df<-boardCopy
    mods$j<-1
  })
  #####
  #end of section
  
  # Modelling
  #####
  #select the target variable and type of model to implement
  output$target<-renderUI({
    req(v$df)
    selectInput("targetVar","Select the Target Variable:",
                choices = names(v$df),selected = names(v$df)[1])
  })
  output$modelType<-renderUI({
    req(input$targetVar)
    explore<-split_columns(v$df,binary_as_factor = T)
    if(input$targetVar %in% names(explore$discrete)){
      selectInput("modelsList", label="Choose a Classification Model:",
                  list(`Linear` = list("generalized linear model","regularized linear model","linear svm"),
                       `Non-linear`=list("knn","naive bayes","decision tree","polynomial svm","radial svm","neural networks"),
                       `Ensemble`=list("random forest","stochastic gradient boost","xgboost")),
                  selected = "generalized linear model")
    }else{
      selectInput("modelsList", label="Choose a Regression Model:",
                  list(`Linear` = list("generalized linear model","regularized linear model","linear svm"),
                       `Non-linear`=list("knn","decision tree","polynomial svm","radial svm","neural networks"),
                       `Ensemble`=list("random forest","stochastic gradient boost","xgboost")),
                  selected = "generalized linear model")
    }
  })
  #caret allows us to grid search to optimize parameters on certain metrics
  output$metricsOptimize<-renderUI({
    req(input$targetVar)
    explore<-split_columns(v$df,binary_as_factor = T)
    if(input$targetVar %in% names(explore$continuous)){
      selectInput("metricsOptim","Choose a Metric to Optimize:",
                  choices = c("RMSE","Rsquared","MAE"), selected = "RMSE")
    }else{
      selectInput("metricsOptim","Choose a Metric to Optimize:",
                  choices = c("Accuracy","Kappa","logLoss",
                              "AUC","Precision","Recall","F",
                              "ROC","Sens","Spec"),
                  selected = "Accuracy")
    }
  })
  #allow up to 3 hyperparameters for simplicity as multiple methods have dozens of options to tune
  output$hyperParam1<-renderUI({
    req(input$hyperTune)
    req(input$modelsList)
    req(v$df)
    if(input$modelsList=="regularized linear model"){
      numericInput("alpha1","Mix % (alpha):",min = 0,max = 1,value = 0)
    }else if(input$modelsList=="linear svm"){
      numericInput("cost1","Cost (C):",min = 1,max = 100,value = 1)
    }else if(input$modelsList=="knn"){
      numericInput("neighbors1","Neighbors (k):",min=1, max=20,value = 5)
    }else if(input$modelsList=="naive bayes"){
      numericInput("laplace1","Laplace Correct (fL):",min=0, max=1,value = 0)
    }else if(input$modelsList=="decision tree"){
      numericInput("cp","Complexity (cp):",min=0, max=1,value = 0.1)
    }else if(input$modelsList=="polynomial svm"){
      numericInput("deg1","Degree:",min=1, max=10,value = 3)
    }else if(input$modelsList=="radial svm"){
      numericInput("sigma","Sigma:",min=0, max=3,value = 1.64243)
    }else if(input$modelsList=="neural networks"){
      numericInput("size","# Hidden Units:",min=1, max=ncol(v$df),value = ncol(v$df)-1)
    }else if(input$modelsList=="random forest"){
      numericInput("mtry"," # Predictors:",min=1, max=ncol(v$df),value = ncol(v$df)-1)
    }else if(input$modelsList %in% c("stochastic gradient boost","xgboost")){
      numericInput("nrounds1"," # Iterations:",min=1, max=1000,value = 100)
    }
  })
  output$hyperParam2<-renderUI({
    req(input$hyperTune)
    req(input$modelsList)
    req(v$df)
    if(input$modelsList=="regularized linear model"){
      numericInput("lambda1","Lambda:",min = 0,max = 1,value = 0.1)
    }else if(input$modelsList=="naive bayes"){
      checkboxInput("kernel1","Use Kernel (T/F):",F)
    }else if(input$modelsList %in% c("polynomial svm","radial svm")){
      numericInput("cost2","Cost (C):",min = 1,max = 100,value = 1)
      param2=if(input$hyperTune==T) input$cost2 else 1
    }else if(input$modelsList=="neural networks"){
      numericInput("decay","Decay:",min = 0,max = 1,value = 0.1)
    }else if(input$modelsList %in% c("stochastic gradient boost","xgboost")){
      numericInput("maxDepth1","Max Depth:",min=1, max=ncol(v$df)-1,value = ncol(v$df)-1)
    }
  })
  output$hyperParam3<-renderUI({
    req(input$hyperTune)
    req(input$modelsList)
    req(v$df)
    if(input$modelsList=="naive bayes"){
      numericInput("bandwidth1","Bandwidth Adjust:",min=0, max=1,value = 1)
    }else if(input$modelsList=="polynomial svm"){
      numericInput("scale1","Scale:",min=0, max=1,value = 0.1)
    }else if(input$modelsList %in% c("stochastic gradient boost","xgboost")){
      numericInput("eta1","Shrinkage:",min=0, max=1,value = 0.3)
    }
  })
  #define the number of folds for K-fold cross validation
  output$cvConsole<-renderUI({
    if(input$crossVal==T){
      fluidRow(column(12,numericInput("folds","# Folds:",min=1,max=25,value=10)),
               hr())
    }
  })

  #when the 'build' is triggered several things happen:
    #1. initialize all the inputs as variables
  observeEvent(input$build,{
    # req(v$df)
    # req(input$targetVar)
    # req(input$metricsOptim)
    # req(input$modelsList)
    # req(input$crossVal)
    # req(input$hyperTune)
    explore<-split_columns(v$df,binary_as_factor = T)
    if(input$targetVar %in% names(explore$discrete)){
      if(class(v$df[[input$targetVar]]) %in% c("numeric","integer")){
        v$df[[input$targetVar]]<-ifelse(v$df[[input$targetVar]]==1,"Y","N")
      }
      v$df[[input$targetVar]]<-as.factor(v$df[[input$targetVar]])
    }
    n <- nrow(v$df)
    set.seed(input$seed)
    training <- sample(1:n,floor(input$traintestSplit*n))
    form <- as.formula(paste0(input$targetVar,"~."))
    
    metricsG1<-c("Accuracy","Kappa")
    metricsG2<-c("logLoss")
    metricsG3<-c("AUC","Precision","Recall","F")
    metricsG4<-c("ROC","Sens","Spec")
    metricsG5<-c("RMSE","Rsquared","MAE")
    #2. Define the trainControl of caret
    if(input$targetVar %in% names(explore$discrete) & input$metricsOptim %in% metricsG1){
      crossvals<-trainControl(method = "cv",number = input$folds, classProbs = T)
      met=input$metricsOptim
      mets=metricsG1
    }else if(input$targetVar %in% names(explore$discrete) & input$metricsOptim %in% metricsG2){
      crossvals<-trainControl(method = "cv",number = input$folds, classProbs = T, summaryFunction = mnLogLoss)
      met=input$metricsOptim
      mets=metricsG2
    }else if(input$targetVar %in% names(explore$discrete) & input$metricsOptim %in% metricsG3){
      crossvals<-trainControl(method = "cv",number = input$folds, classProbs = T, summaryFunction = prSummary)
      met=input$metricsOptim
      mets=metricsG3
    }else if(input$targetVar %in% names(explore$discrete) & input$metricsOptim %in% metricsG4){
      crossvals<-trainControl(method = "cv",number = input$folds, classProbs = T, summaryFunction = twoClassSummary)
      met=input$metricsOptim
      mets=metricsG4
    }else{
      crossvals<-trainControl(method = "cv",number = input$folds)
      met=input$metricsOptim
      mets=metricsG5
    }
    #3. Train models, 
    #glm has no hyperparameters to tune but requires a family input for classification, 
    #glmnet also requires a family input but HAS hyperparameters to tune, hence both are singled out.
    if(input$modelsList=="generalized linear model"){
      method="glm"
      fam=if(input$targetVar %in% names(explore$discrete))"binomial" else "gaussian"
      if(input$crossVal==T){
        model=caret::train(form,v$df[training,],method=method,metric=met,family=fam,trControl=crossvals)
      }else{
        model=caret::train(form,v$df[training,],method=method,family=fam,metric=met)
      }
    }else if(input$modelsList=="regularized linear model"){
      method="glmnet"
      param1=if(input$hyperTune==T) input$alpha1 else 0
      param2=if(input$hyperTune==T) input$lambda1 else 0.1
      fam=if(input$targetVar %in% names(explore$discrete))"binomial"else"gaussian"
      grid=expand.grid(alpha=param1,lambda=param2)
      if(input$crossVal==T & input$hyperTune==T){
        model=caret::train(form,v$df[training,],method=method,family=fam,metric=met,tuneGrid=grid,trControl=crossvals)
      }else if(input$crossVal==T & input$hyperTune==F){
        model=caret::train(form,v$df[training,],method=method,family=fam,metric=met,trControl=crossvals)
      }else if(input$crossVal==F & input$hyperTune==T){
        model=caret::train(form,v$df[training,],method=method,family=fam,metric=met,tuneGrid=grid)
      }else{
        model=caret::train(form,v$df[training,],method=method,family=fam,metric=met)
      }
    }else{
      if(input$modelsList=="linear svm"){
        method="svmLinear"
        param1=if(input$hyperTune==T) input$cost1 else 1
        grid=expand.grid(cost=param1)
      }else if(input$modelsList=="knn"){
        method="knn"
        param1=if(input$hyperTune==T) input$neighbors1 else 5
        grid=expand.grid(k=param1)
      }else if(input$modelsList=="naive bayes"){
        method="naive_bayes"
        param1=if(input$hyperTune==T) input$laplace1 else 0
        param2=if(input$hyperTune==T) input$kernel1 else F
        param3=if(input$hyperTune==T) input$bandwidth1 else 1
        if(input$targetVar %in% names(explore$continuous)){output$metric1<-renderText({"Naive Bayes is not compatible with Continuous Target!"})}
        grid=expand.grid(fL=param1,usekernel=param2,adjust=param3)
      }else if(input$modelsList=="decision tree"){
        method='rpart'
        param1=if(input$hyperTune==T) input$cp else 0.1
        grid=expand.grid(cp=param1)
      }else if(input$modelsList=="polynomial svm"){
        method="svmPoly"
        param1=if(input$hyperTune==T) input$deg1 else 3
        param2=if(input$hyperTune==T) input$cost2 else 1
        param3=if(input$hyperTune==T) input$scale1 else 0.1
        grid=expand.grid(degree=param1,C=param2,scale=param3)
      }else if(input$modelsList=="radial svm"){
        method="svmRadial"
        param1=if(input$hyperTune==T) input$sigma else 1.64243
        param2=if(input$hyperTune==T) input$cost2 else 1
        expand.grid(sigma=param1,C=param2)
      }else if(input$modelsList=="neural networks"){
        method="nnet"
        param1=if(input$hyperTune==T) input$size else ncol(v$df)-1
        param2=if(input$hyperTune==T) input$decay else 0.1
        grid=expand.grid(size=param1,decay=param2)
      }else if(input$modelsList=="random forest"){
        method="rf"
        param1=if(input$hyperTune==T) input$mtry else ncol(v$df)-1 
        grid=expand.grid(mtry=param1)
      }else if(input$modelsList=="xgboost"){
        method="xgbTree"
        param1=if(input$hyperTune==T) input$nrounds1 else 100
        param2=if(input$hyperTune==T) input$maxDepth1 else ncol(v$df)-1
        param3=if(input$hyperTune==T) input$eta1 else 0.3
        grid=expand.grid(nrounds=param1,max_depth=param2,eta=param3)
      }else if(input$modelsList=="stochastic gradient boost"){
        method="gbm"
        param1=if(input$hyperTune==T) input$nrounds1 else 100
        param2=if(input$hyperTune==T) input$maxDepth1 else ncol(v$df)-1
        param3=if(input$hyperTune==T) input$eta1 else 0.3
        grid=expand.grid(n.tress=param1,interaction.depth=param2,shrinkage=param3)
      }
      
      if(input$crossVal==T & input$hyperTune==T){
        model=caret::train(form,v$df[training,],method=method,metric=met,tuneGrid=grid,trControl=crossvals)
      }else if (input$crossVal==T & input$hyperTune==F){
        model=caret::train(form,v$df[training,],method=method,metric=met,trControl=crossvals)
      }else if (input$crossVal==F & input$hyperTune==T){
        model=caret::train(form,v$df[training,],method=method,metric=met,tuneGrid=grid)
      }else{
        model=caret::train(form,v$df[training,],method=method,metric=met)
      }
    }
    #save the models built into the reactiveValues Container
    modelName=paste0("model_",mods$j)
    v[[modelName]]<-model
    mods$j=mods$j+1
    #4. Record the Metrics of each model into the leaderboard
    paramStore<-get_param_string(model)
    record=data.frame(model=modelName,type=input$modelsList,target=input$targetVar,`train-test ratio`=input$traintestSplit,seed=input$seed,parameters=paramStore,metric=model$metric,value=round(model$results[as.integer(rownames(model$bestTune)),][[met]],3))
    lb$df<-rbind(lb$df,record)
    #5. Visualize the metrics into plots. 4 different plots will be made:
      # 1. Barchart of metrics
    if(input$metricsOptim %in% metricsG1){
      ypreds<-predict(model,v$df[-training,])
      test_metrics=data.frame(Accuracy=Accuracy(ypreds,v$df[-training,which(names(v$df)==input$targetVar)]),
                              Kappa=ModelMetrics::kappa(v$df[-training,which(names(v$df)==input$targetVar)],ypreds),
                              type="test")
    }else if(input$metricsOptim %in% metricsG2){
      ypreds<-predict(model,v$df[-training,],type="prob")[,2]
      test_metrics=data.frame(logLoss=LogLoss(ypreds,v$df[-training,which(names(v$df)==input$targetVar)]),type="test")
    }else if(input$metricsOptim %in% metricsG3){
      ypreds<-predict(model,v$df[-training,],type="prob")[,2]
      test_metrics=data.frame(AUC=AUC(ypreds,v$df[-training,which(names(v$df)==input$targetVar)]),
                              Precision=Precision(ypreds,v$df[-training,which(names(v$df)==input$targetVar)]),
                              Recall=Recall(ypreds,v$df[-training,which(names(v$df)==input$targetVar)]),
                              `F`=F1_Score(v$df[-training,which(names(v$df)==input$targetVar)],ypreds),type="test")
    }else if(input$metricsOptim %in% metricsG4){
      ypreds<-predict(model,v$df[-training,],type="prob")[,2]
      test_metrics=data.frame(ROC=PRAUC(ypreds,v$df[-training,which(names(v$df)==input$targetVar)]),
                              Sens=Sensitivity(ypreds,v$df[-training,which(names(v$df)==input$targetVar)]),
                              Spec=Specificity(ypreds,v$df[-training,which(names(v$df)==input$targetVar)]),type="test")
    }else{
      ypreds<-predict(model,v$df[-training,])
      test_metrics=data.frame(RMSE=RMSE(ypreds,v$df[-training,which(names(v$df)==input$targetVar)]),
                              Rsquared=R2_Score(ypreds,v$df[-training,which(names(v$df)==input$targetVar)]),
                              MAE=MAE(ypreds,v$df[-training,which(names(v$df)==input$targetVar)]),type="test")
    }
    
    metrics_df<-model$results[as.integer(rownames(model$bestTune)),]%>%
      dplyr::select(mets)%>%
      mutate(type="train")%>%
      rbind(test_metrics)%>%
      pivot_longer(cols=mets,names_to = "metric")
    metricsPlot<-ggplot(metrics_df,aes(x=metric,y=value,fill=type))+
      geom_bar(stat = "identity",position = "dodge")+
      ggtitle("Metrics")+
      theme(plot.title = element_text(hjust = 0.5))
      # 2. Predicted against Actuals (QQplot/Confusion Matrix)
    if(input$targetVar %in% names(explore$continuous)){
      actualvpred_df=data.frame(actual=v$df[-training,which(names(v$df)==input$targetVar)],
                                predicted=ypreds)
      actualvpredPlot<-ggplot(actualvpred_df,aes(x=predicted,y=actual))+
        geom_point()+
        geom_abline()+
        ggtitle("QQ-Plot of Actuals against Predicted")+
        theme(plot.title = element_text(hjust = 0.5))
    }else{
      ypred_factors=predict(model,v$df[-training,])
      cm<-caret::confusionMatrix(ypred_factors,v$df[-training,which(names(v$df)==input$targetVar)],dnn=c("Predicted","Actual"))$table%>%
        conf_mat()
      actualvpredPlot<-autoplot(cm,type="heatmap")+
        scale_fill_gradient(low = "pink", high = "cyan")+
        ggtitle("Confusion Matrix")+
        theme(plot.title = element_text(hjust = 0.5))
    }
      # 3. Coefficients (only for glm)
    if(input$modelsList=="generalized linear model"){
      coef_df=data.frame(coef(model$finalModel))%>%
        dplyr::rename(value=names(data.frame(coef(model$finalModel))))%>%
        mutate(is_positive=ifelse(value>0,"Y","N"))
      coefPlot<-ggplot(coef_df,aes(x=value,y=rownames(coef_df),fill=is_positive))+
        geom_bar(stat="identity")+
        theme(legend.position = "none")+
        geom_label(aes(label=round(value,3)))+
        ggtitle("Coefficients")+
        ylab("Predictors")+
        theme(plot.title = element_text(hjust = 0.5))
    }
    #   4. Probability densities (classification models)
    if(input$targetVar %in% names(explore$discrete)){
      predDensities<- predict(model, v$df[-training,], type="prob")%>%
        mutate(type="test")
      densities_df<-predict(model, v$df[training,], type="prob")%>%
        mutate(type="train")%>%
        rbind(predDensities)%>%
        pivot_longer(cols=unique(v$df[[input$targetVar]]),names_to = "class")
      probdensityPlot<-ggplot(densities_df,aes(x=value,fill=class))+
        geom_density(alpha=0.5)+
        facet_wrap(~type)+
        ggtitle("Probability Density Distribution of Classes")+
        theme(plot.title = element_text(hjust = 0.5))
    }
    
    output$table1<-renderPlot({
      return(metricsPlot)
    })
    output$table2<-renderPlot({
      return(actualvpredPlot)
    })
    output$table3<-renderPlot({
      if(input$modelsList=="generalized linear model"){return(coefPlot)}
    })
    output$table4<-renderPlot({
      if(input$targetVar %in% names(explore$discrete)){return(probdensityPlot)}
    })
  })
  #####
  #end of section
  
  # Explanable ML
  #####
  # Choose from saved models the model you want to understand
  output$modelSelect<-renderUI({
    req(lb$df)
    selectInput("modeltoExplain",label="Model:",choices=lb$df[["model"]], selected = lb$df[["model"]][1])
  })
  #For global interpretation, feature effects and feature interaction allow the isolation of a single Predictor
  output$featureParam<-renderUI({
    req(input$modeltoExplain)
    if(input$globalFeatures %in% c("eff","inter")){
      selectInput("effintsFeat","Select a Specific Feature:",
                  choices=c("all",names(v[[input$modeltoExplain]]$trainingData)[-1]),
                  selected="all")
    }
  })
  #For local interpretation, we can define a specific datapoint in the dataset to view the explanation
  output$localParam<-renderUI({
    req(v$df)
    sliderInput("dataIndex","Select a Data Point to View:",
                min=1,max=nrow(v$df),value = 1, step = 1)
  })
  #once the 'explain' button is triggered, several 2 things occur
    #1.Global Interpretation occurs 
  observeEvent(input$explain,{
    req(input$modeltoExplain)
    req(v$df)
    if(v[[input$modeltoExplain]]$modelType=="Regression"){
      loss="rmse"
      predictorEnv<-Predictor$new(v[[input$modeltoExplain]],data=v$df[,-which(names(v$df)==input$targetVar)],y=v$df[input$targetVar])
    }else{
      loss="ce"
      predictorEnv<-Predictor$new(v[[input$modeltoExplain]],data=v$df[,-which(names(v$df)==input$targetVar)],y=v$df[input$targetVar],type="prob",class=levels(v$df[input$targetVar])[2])
    }
    
    if(input$globalFeatures=="imp"){
      imp <- FeatureImp$new(predictorEnv, loss = loss)
      featImp_plot<-plot(imp)+
        ggtitle("Feature Importance")+
        theme(plot.title = element_text(hjust = 0.5))
      output$globalPlot<-renderPlot({return(featImp_plot)})
    }else if(input$globalFeatures=="eff"){
      if(input$effintsFeat=="all"){
        effs <- FeatureEffects$new(predictorEnv)
        featEffs_plot<-plot(effs)
      }else{
        effs <- FeatureEffect$new(predictorEnv, feature = input$effintsFeat)
        featEffs_plot<-plot(effs)+
          ggtitle(paste("Accumulated Local Effects of",input$effintsFeat))+
          theme(plot.title = element_text(hjust = 0.5))
      }
      output$globalPlot<-renderPlot({return(featEffs_plot)})
    }else{
      if(input$effintsFeat=="all"){
        inters <- Interaction$new(predictorEnv)
        featInter_plot<-plot(inters)+
          ggtitle("Feature Interactions")+
          theme(plot.title = element_text(hjust=0.5))
      }else{
        inters <- Interaction$new(predictorEnv,feature = input$effintsFeat)
        featInter_plot<-plot(inters)+
          ggtitle(paste("Feature Interactions with",input$effintsFeat))+
          theme(plot.title = element_text(hjust=0.5))
      }
      output$globalPlot<-renderPlot({return(featInter_plot)})
    }
    #2. Local Interpretation occurs
    if(input$localFramework=="lime"){
      lime_explain <- LocalModel$new(predictorEnv, x.interest = v$df[input$dataIndex,-which(names(v$df)==input$targetVar)])
      lime_plot<-plot(lime_explain)+
        ggtitle(paste0("Local Prediction Explanation of Datapoint #",input$dataIndex))+ 
        theme(plot.title = element_text(hjust=0.5))
      output$localPlot<-renderPlot({return(lime_plot)})
    }else{
      shap_explain<-Shapley$new(predictorEnv, x.interest = v$df[input$dataIndex,-which(names(v$df)==input$targetVar)])
      shap_plot<-shap_explain$plot()+
        ggtitle(paste0("Local Prediction Explanation of Datapoint #",input$dataIndex))+
        theme(plot.title = element_text(hjust=0.5))
      output$localPlot<-renderPlot({return(shap_plot)})
    }
  })
  #####
  #end of section
  
  # Make New Predictions
  #####
  #Users can choose to use one of the saved models to predict data on
  output$modelSelect2<-renderUI({
    req(lb$df)
    selectInput("modeltoPredict",label="Model:",choices=lb$df[["model"]], selected = lb$df[["model"]][1])
  })
  #Users can either: Upload a new dataset to make predictions, download the train-test predictions or make a new single prediction
  output$predictActions<-renderUI({
    if(input$predictType=="newdata"){
      fluidRow(column(12,fileInput("upload2", "Choose a file", buttonLabel = "Upload...",accept = ".csv")),
               hr(),
               column(12,downloadButton("dlnewPreds", "Download Predictions", style = "width:100%;")))
    }else{
      fluidRow(column(12,uiOutput("newpredParams")),
               column(12,actionButton("prednewSingle","Predict",width = "100%")))
    }
  })
  #Generate the correct number of fields to input for a new single prediction
  output$newpredParams<-renderUI({
    req(input$modeltoPredict)
    m=names(v[[input$modeltoPredict]]$trainingData)[-1]
    lapply(1:length(m),function(i){
      xparamContainer<-paste0("xContainer_", i)
      uiOutput(xparamContainer)
      
      output[[xparamContainer]]<-renderUI({
        xparamID<-paste0("xParam_", i)
        xparamTypes<-paste0("x",i,"_class")
        paramName<-m[i]
        fluidRow(column(9,textInput(xparamID,paramName,placeholder="Type in a new parameter value")),
                 column(3,selectInput(xparamTypes,"class:",choices = c("factor","numeric"))))
      })
    })
  })
  #uploading a new dataset to predict on
  newData<-reactiveValues(df=NULL)
  observeEvent(input$upload2,{
    req(input$modeltoPredict)
    newData$df=read.csv(input$upload$datapath,stringsAsFactors = T)
    newPreds=predict(v[[input$modeltoPredict]],newData$df)
    output_df=data.frame(Index=1:nrow(newData$df),Prediction=newPreds)
    newData$outputData<-output_df
    output$predResults<-renderDataTable({
      return(output_df)
    })
  })
  output$dlnewPreds <- downloadHandler(
    filename = paste0("new_predictions_",today(),".csv"),
    content = function(file) {
      write.csv(newData$outputData, file, row.names = FALSE)
    }
  )
  #making a single prediction
  observeEvent(input$prednewSingle,{
    req(input$modeltoPredict)
    m=names(v[[input$modeltoPredict]]$trainingData)[-1]
    inputList=list()
    for(i in 1:length(m)){
      xparamID<-paste0("xParam_", i)
      xparamTypes<-paste0("x",i,"_class")
      if(input[[xparamTypes]]=="numeric"){
        inputList=append(inputList,as.numeric(input[[xparamID]]))
      }else{
        inputList=append(inputList,input[[xparamID]])
      }
    }
    singlePred_df=data.frame(inputList)
    names(singlePred_df)=m
    singlePred=predict(v[[input$modeltoPredict]],singlePred_df)
    output_df=data.frame(Index=1,Prediction=singlePred)
    output$predResults<-renderDataTable({
      return(output_df)
    })
  })
  
  #####
  #end of section
}
shinyApp(ui,server)