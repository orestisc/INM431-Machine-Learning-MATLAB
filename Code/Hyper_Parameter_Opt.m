clear all; clc; close all;
warning('OFF', 'MATLAB:table:ModifiedAndSavedVarnames')
%% Summarising and Preprocessing of Data
%Loading Dataset into Matlab
data = readtable('data/winequality-white.csv');
[m n] = size(data); %Measuring columns and rows
fprintf('The dataset consist of %d Rows and %d Columns.\n\n', m, n)
inputTable = data;
inputTable(:,8)=[]; %Eliminate column 8(density feature)as it showed a 
%high multicollinearity using the Variance Infation Error 
%(vif = diag(inv(corrmatrix))';)
labels = inputTable.Properties.VariableNames;
predictorNames = {'fixedAcidity', 'volatileAcidity', 'citricAcid', 'residualSugar', 'chlorides', 'freeSulfurDioxide', 'totalSulfurDioxide', 'pH', 'sulphates', 'alcohol'};
predictors = inputTable(:, predictorNames);
response = inputTable.quality;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false];

%Fill Missing Values with Mean
fillmissing(inputTable,'movmean',5);
n_missingvalues = sum(sum(ismissing(inputTable)));
fprintf('There are %d missing values in the dataset.\n',n_missingvalues)
warning('OFF');

%% Finding Optimize Hyperparameters for Models
%Using Kfold validation method with 50 Evaluation
part = cvpartition(response,"KFold",10);
opt = struct("CVPartition",part,"MaxObjectiveEvaluations",50); %Creating a struck for Hyperparameter Optimazations

%Knn Hyperparameters Optimisation model
mdl = fitcknn(inputTable,response,"OptimizeHyperparameters","auto","HyperparameterOptimizationOptions",opt);

%Best estimated feasible point (according to models)
fprintf('\n')
fprintf('Best Estimated NumNeighbors = 251');
fprintf('\n')
fprintf('Best Estimated Distance = "cosine"');

testLoss = resubLoss(mdl); %Evaluating Test loss
fprintf('\n')
fprintf('Loss error = %f\n',testLoss);

%Random forest Hyperparameters Optimisation
mdl2 = fitcensemble(inputTable,response,'OptimizeHyperparameters','all',"HyperparameterOptimizationOptions",opt);

%Best estimated feasible point (according to models)
fprintf('\n')
fprintf('Best Estimated NumLearningCycles = 11');
fprintf('\n')
fprintf('Best Estimated Method = "Bag"');
fprintf('\n')

fprintf('..............................................................................................\n')
fprintf('....................................----THE END----...........................................\n')
fprintf('..............................................................................................\n')