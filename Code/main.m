clear all; clc; close all;
warning('OFF', 'MATLAB:table:ModifiedAndSavedVarnames')
fprintf('...................................................\n')
fprintf('WINE QUALITY MACHINE LEARNING CLASSIFICATION MODELS\n')
fprintf('...................................................\n')
fprintf('\n')

%% Summarising and Preprocessing of Data
%Loading Dataset into Matlab
data = readtable('data/winequality-white.csv');
[m n] = size(data); %Measuring columns and rows
fprintf('The dataset consist of %d Rows and %d Columns.\n\n', m, n)
inputTable = data;

labels = inputTable.Properties.VariableNames; %Labeling features
predictorNames = {'fixedAcidity', 'volatileAcidity', 'citricAcid', 'residualSugar', 'chlorides', 'freeSulfurDioxide', 'totalSulfurDioxide','density', 'pH', 'sulphates', 'alcohol'};
predictors = inputTable(:, predictorNames); 
response = inputTable.quality;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false, false];

%Fill Missing Values with Mean
fillmissing(inputTable,'movmean',5);
n_missingvalues = sum(sum(ismissing(inputTable))); %Calculating how many missing values are left
fprintf('There are %d missing values in the dataset.\n',n_missingvalues)

%Summary Table
summary(inputTable)

%Quality Distribution
fprintf('Percentage of quality of wines in dataset:\n')
tabulate(inputTable.quality) %Visualizing percentage of quality of win
%% Exploratory Data Analysis

%Plotting Histograms for Feature variables
figure('Name','Histograms of Features','position',[10 10 1200 600])
for col_index = 1:n-1
    subplot(4,3,col_index) %Plotting all histograms in one figure for better visualisation
    histogram(inputTable{:, col_index})
    title(inputTable.Properties.VariableNames{col_index})
    ylabel('Quantity')
end

%Plotting Histogram for Target variable
figure('Name','Histogram of Target','position',[10 10 700 400])
histogram(inputTable.quality)
title('quality');
ylabel('Quantity');

%Plotting correlation map between features
figure('Name','Correlation between features','position',[10 10 700 400])
corrmatrix = corr(table2array(inputTable),'type','Pearson'); %Creating correlation matrix
imagesc(corrmatrix) %Visualizing correlation matrix as an image
set(gca, 'XTick', 1:12); 
set(gca, 'YTick', 1:12);
set(gca, 'XTickLabel', labels, 'FontSize', 9)
set(gca, 'YTickLabel', labels, 'FontSize', 9)
set(gca, 'XTickLabelRotation', 45)
title('Correlation map between features (Pearson)', 'FontSize', 11)
colormap('winter'); %Colouring the map
brighten(-0.2);
caxis([-1 1]); %Colouring between interval[-1,1]
colorbar;
%Plotting correlation values on map
vals = num2str(corrmatrix(:), '%0.2f');
vals = strtrim(cellstr(vals)); %Converting to cell array of character vectors and removing leading and trailing whitespace from strings
[x, y] = meshgrid(1:12); %Create 2-D grid coordinates
valsh = text(x(:), y(:), vals(:), 'HorizontalAlignment', 'center');
set(valsh, 'color', 'black','FontSize', 10) %Plotting array into figure

inputTable(:,8)=[]; %Eliminate column 8(density feature)as it showed a 
%high multicollinearity using the Variance Infation Error 
%(vif = diag(inv(corrmatrix))';)
predictorNames = {'fixedAcidity', 'volatileAcidity', 'citricAcid', 'residualSugar', 'chlorides', 'freeSulfurDioxide', 'totalSulfurDioxide', 'pH', 'sulphates', 'alcohol'};
predictors = inputTable(:, predictorNames); 
response = inputTable.quality;
isCategoricalPredictor = [false, false, false, false, false, false, false, false, false, false];
%% Splitting data
% Set up holdout validation
rng(5); %For reproducibility
cvp = cvpartition(response, 'Holdout', 0.25); %Training 75% and letting 25% for testing
xtrain = predictors(cvp.training, :); %Training features
ytrain = response(cvp.training, :); %Training target
trainingIsCategoricalPredictor = isCategoricalPredictor;

%% K-NN Classification Model
% Creation and evaluation of K-NN Classification Model 
fprintf('............................................\n')
fprintf('K-NN Classification model\n')


mdlknn = fitcknn(xtrain,ytrain,'Distance',... 
    'Euclidean','Exponent', [],'NumNeighbors', 1,'DistanceWeight', 'Equal', ...
    'Standardize', true,'ClassNames', [3; 4; 5; 6; 7; 8; 9]); %Classification model

% Create the result struct with predict function
knnPredictFcn = @(x) predict(mdlknn, x);
knnValPredictFcn = @(x) knnPredictFcn(x);

% Compute validation predictions
xtest = predictors(cvp.test, :); %Testing data for features
ytest = response(cvp.test, :); %Testing data for target
[knnpred, KnnValScores] = knnValPredictFcn(xtest); %Function for validation predictions

% Compute validation accuracy
knnCorrPred = (knnpred == ytest); %Predicting and comparing predictions with response
knnMissing = isnan(ytest);
knnCorrPred = knnCorrPred(~knnMissing); %Correct predictions
%Accuracy = (TP+TN)/(TP+TN+FP+FN)
KnnValAccuracy = sum(knnCorrPred)/length(knnCorrPred); %Computing models Accuraccy
fprintf('\n')
fprintf('Accuracy: %.3f\n', KnnValAccuracy) 
C_knn = confusionmat(ytest,knnpred); %Creating Confusion Matrix
KnnValScores;
KnnTrainLoss = resubLoss(mdlknn); %Evaluating Train loss of model
fprintf('Train Loss error = %f\n',KnnTrainLoss);
KnnTestLoss = loss(mdlknn,knnpred,ytest); %Evaluating Test loss of model
fprintf('Test loss error = %f\n',KnnTestLoss);
%% Plots of K-NN Classification model
%Plot confusion matrix
figure('Name','Confusion chart','position',[10 10 700 400])
CCKnn = confusionchart(C_knn); %Visualiszing confusion chart
CCKnn.Title = 'K-NN Confusion Matrix';
    
%Plot for Actual Vs Predicted Results
figure('Name','Plot of Actual Vs Predicted','position',[10 10 700 400])
plot(ytrain,'b*-','LineWidth',1), hold on %Plotting Actual Results
plot(knnpred,'r.-','LineWidth',1,'MarkerSize',15) %Plotting Predicted Results
title('K-NN Classification: Actual Vs Predicted')
% Observe first hundred points
xlim([0 100])
legend({'Actual','Predicted'})
xlabel('Training Data point');
ylabel('Wine quality');

%% Random Forest Classification
% Creation and evaluation of Random Forest Classification Model 
fprintf('............................................\n')
fprintf('Random Forest Classification model\n')


tree = templateTree('MaxNumSplits', 20); %Creating Template tree
%Random forest classification model with 'RUSBoost' method
RF = fitcensemble(xtrain,ytrain, ...
    'Method', 'RUSBoost','NumLearningCycles', 30,'Learners', tree, ...
    'LearnRate', 0.01,'ClassNames', [3; 4; 5; 6; 7; 8; 9]); 

% Create the result struct with predict function
RFPredictFcn = @(y) predict(RF, y);
RFValPredictFcn = @(y) RFPredictFcn(y);
% Compute validation predictions
[RFpred, RFValScores] = RFValPredictFcn(xtest); %Function for validation predictions

% Compute validation accuracy
RFCorrPred = (RFpred == ytest); %Predicting and comparing predictions with response
RFMissing = isnan(ytest);
RFCorrPred = RFCorrPred(~RFMissing);
RFValAccuracy = sum(RFCorrPred)/length(RFCorrPred); %Computing models Accuraccy
fprintf('\n')
fprintf('Accuracy: %.3f\n', RFValAccuracy)
C_RF = confusionmat(ytest,RFpred); %Creating Confusion Matrix
RFValScores;
RFTrainingtLoss = resubLoss(RF,'mode','cumulative'); %Evaluating Train loss of model
RFTestLoss = loss(RF,xtest,ytest,'mode','cumulative'); %Evaluating Test loss of model
%% Plots for Random Forest Classification model
%Plot of Confusion matrix
figure('Name','Confusion chart','position',[10 10 700 400])
cmRF = confusionchart(C_RF);%Visualiszing confusion chart
cmRF.Title = 'Random Forest Confusion Matrix';

%Plot for Actual Vs Predicted Results
figure('Name','Plot of Actual vs Predicted','position',[10 10 700 400])
plot(ytest,'b*-','LineWidth',1), hold on %Plotting Actual Results
plot(RFpred,'r.-','LineWidth',1,'MarkerSize',15) %Plotting Predicted Results
title('Random Forest : Actual vs Predicted')
% Observe first hundred points
xlim([0 100])
legend({'Actual','Predicted'})
xlabel('Training Data point');
ylabel('Wine quality');

%Plotting Predictor importance
[predictorImp,sortedIndex] = sort(RF.predictorImportance); %Sorting Predictors Importance
figure('Name','Precitors Importance','position',[10 10 700 400])
barh(predictorImp) %Plot in bar graph
set(gca,'ytickLabel',predictorNames(sortedIndex))
title('Predictor Importance','FontSize',11)

%Plot for Training Set Loss
figure('Name','Training Set Loss','position',[10 10 700 400])
plot(RFTrainingtLoss), hold on %Plotting Train loss of model
plot(RFTestLoss,'r') %Plotting Test loss of model
legend({'Training Set Loss','Test Set Loss'})
xlabel('Number of trees');
ylabel('Mean Squared Error');
title('RF:Training Loss vs Test Loss')

%% Optimized Models

%Knn-Classification Optimized with best estimated feasible point(according
%to models)
%Optimized with:
%                Distance = 'hamming'
%                NumNeighbor = 16
fprintf('............................................\n')
fprintf('Optimized K-NN Classification model with:\n')
fprintf('               Distance = "cosine"\n')
fprintf('               NumNeighbor = 251\n')


oKnn = fitcknn(xtrain,ytrain,'Distance', 'cosine', ...
    'Exponent', [],'NumNeighbors', 251,'DistanceWeight', 'SquaredInverse', ...
    'Standardize', true,'ClassNames', [3; 4; 5; 6; 7; 8; 9]);

% Create the result struct with predict function
oKnnPredictFcn = @(p) predict(oKnn, p);
oKnnValPredictFcn = @(p) oKnnPredictFcn(p);
% Compute validation predictions
[oKnnpred, oKnnValScores] = oKnnValPredictFcn(xtest); %Function for validation predictions

% Compute validation accuracy
oKnnCorrPred = (oKnnpred == ytest); %Predicting and comparing predictions with response
oKnnMissing = isnan(ytest);
oKnnCorrPred = oKnnCorrPred(~oKnnMissing);
oKnnValAccuracy = sum(oKnnCorrPred)/length(oKnnCorrPred); %Computing models Accuraccy
fprintf('\n')
fprintf('Optimizied Model Accuracy: %.3f\n', oKnnValAccuracy)

oKnnTrainLoss = resubLoss(oKnn); %Evaluating Train loss of model
fprintf('Train Loss error= %.3f\n', oKnnTrainLoss)
oKnnTestLoss = loss(oKnn,oKnnpred,ytest);
fprintf('Test loss error = %.3f\n', oKnnTestLoss)

C_oKnn = confusionmat(ytest,oKnnpred); %Creating Confusion Matrix

%Plot of Confusion matrix
figure('Name','Confusion chart','position',[10 10 700 400])
CCKnn = confusionchart(C_oKnn); %Visualiszing confusion chart
CCKnn.Title = 'Optimised K-NN Confusion Matrix';

%.........................................................................................................................
% Random Forest Optimized with best estimated feasible point(according
%to models)
%Optimized with:
%                Method = 'Bag'
%                NumLearningCycles = 13
%                MinLeafSize = 2
%                NumVariablesToSample = 11
%                MaxNumSplits = 13
fprintf('............................................\n')
fprintf('Optimized Random Forest Classification model with:\n')
fprintf('               Method = "Bag"\n')
fprintf('               NumLearningCycles = 11\n')
fprintf('               MinLeafSize = 2\n')
fprintf('               NumVariablesToSample = 11\n')
fprintf('               MaxNumSplits = 4450\n')


otree = templateTree('MaxNumSplits', 4450,'MinLeafSize',2,'SplitCriterion','deviance','NumVariablesToSample',11); %Creating Template tree
%Random forest classification model with 'bag' method
oRF = fitcensemble(xtrain,ytrain, ...
    'Method', 'Bag','NumLearningCycles', 11,'Learners', otree, ...
    'ClassNames', [3; 4; 5; 6; 7; 8; 9]);

% Create the result struct with predict function
oRFPredictFcn = @(z) predict(oRF, z);
oRFValPredictFcn = @(z) oRFPredictFcn(z);
% Compute validation predictions
[oRFpred, oRFValScores] = oRFValPredictFcn(xtest); %Function for validation predictions

% Compute validation accuracy
oRFCorrPred = (oRFpred == ytest); %Predicting and comparing predictions with response
oRFMissing = isnan(ytest);
oRFCorrPred = oRFCorrPred(~oRFMissing);
oRFValAccuracy = sum(oRFCorrPred)/length(oRFCorrPred); %Computing models Accuraccy
fprintf('\n')
fprintf('Optimized Model Accuracy: %.3f\n', oRFValAccuracy)
oRFTrainingtLoss = resubLoss(oRF,'mode','cumulative'); %Evaluating Train loss of model
oRFTestLoss = loss(oRF,xtest,ytest,'mode','cumulative'); %Evaluating Test loss of modelg

C_oRF = confusionmat(ytest,oRFpred); %Creating Confusion Matrix

%Plot of Confusion matrix
figure('Name','Confusion chart','position',[10 10 700 400])
CCoRF = confusionchart(C_oRF); %Visualiszing confusion chart
CCoRF.Title = 'Optimised Random Forest Confusion Matrix';


%Plot for Training Set Loss
figure('Name','Training Set Loss','position',[10 10 700 400])
plot(oRFTrainingtLoss), hold on
plot(oRFTestLoss,'r')
legend({'Training Set Loss','Test Set Loss'})
xlabel('Number of trees');
ylabel('Mean Squared Error');
title('Optimized RF:Training Loss vs Test Loss')

fprintf('..............................................................................................\n')
fprintf('....................................----THE END----...........................................\n')
fprintf('..............................................................................................\n')