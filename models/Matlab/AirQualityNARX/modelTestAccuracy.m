%% Test error of Bayesian Optimized: SVM, GPR, Tree, Ensemble models

%% Load test data

T = readtable('6vars/BETN073.csv');
X_test = T(7306:7676, 3:9);
y_test = T(7306:7676, 2);


%% Load training data
% Indices: B3653:I7305
X_train = T(3653:7305, 3:9);
y_train = T(3653:7305, 2);

%%
X_test_matrix = table2array(X_test);
y_test_matrix = table2array(y_test);

%%
X_test.Properties.VariableNames{'Var3'} = 'VarName3';
X_test.Properties.VariableNames{'Var4'} = 'VarName4';
X_test.Properties.VariableNames{'Var5'} = 'VarName5';
X_test.Properties.VariableNames{'Var6'} = 'VarName6';
X_test.Properties.VariableNames{'Var7'} = 'e11';
X_test.Properties.VariableNames{'Var8'} = 'VarName8';
X_test.Properties.VariableNames{'Var9'} = 'VarName9';
%% Naive-1
disp("10-fold cross-validation");
y_test_prediction = [y_test_matrix(1); y_test_matrix(1:end-1)];
MAPE = mean((abs(y_test_prediction - y_test_matrix))./y_test_matrix);
IOA = index_of_agreement(y_test_matrix, y_test_prediction);
disp("Naive-1 MAPE: " + round(MAPE * 100, 2) + "% IOA: " + round(IOA * 100, 2) + "% (Bayesian optimization 100 iters)")

%% Tree
load('best_Tree_Bayes100.mat');
y_test_prediction = trainedTreeBayes100.predictFcn(X_test);
MAPE = mean((abs(y_test_prediction - y_test_matrix))./y_test_matrix);
IOA = index_of_agreement(y_test_matrix, y_test_prediction);
disp("Tree MAPE: " + round(MAPE * 100, 2) + "% IOA: " + round(IOA * 100, 2) + "% (Bayesian optimization 100 iters)")

%% Ensemble
load('best_Ensemble_Bayes100.mat');
y_test_prediction = trainedEnsembleBayes100.predictFcn(X_test);
MAPE = mean((abs(y_test_prediction - y_test_matrix))./y_test_matrix);
IOA = index_of_agreement(y_test_matrix, y_test_prediction);
disp("Ensemble MAPE: " + round(MAPE * 100, 2) + "% IOA: " + round(IOA * 100, 2) + "% (Bayesian optimization 100 iters)")


%% Gaussian SVM
load('best_Gaussian_SVM_Bayes100.mat');
y_test_prediction = trainedGaussianSVMBayes100.predictFcn(X_test);
MAPE = mean((abs(y_test_prediction - y_test_matrix))./y_test_matrix);
IOA = index_of_agreement(y_test_matrix, y_test_prediction);
disp("Gaussian SVM MAPE: " + round(MAPE * 100, 2) + "% IOA: " + round(IOA * 100, 2) + "% (Bayesian optimization 100 iters)")


%% GPR
load('best_GPR_Bayes100.mat');
y_test_prediction = trainedGPRBayes100.predictFcn(X_test);
MAPE = mean((abs(y_test_prediction - y_test_matrix))./y_test_matrix);
IOA = index_of_agreement(y_test_matrix, y_test_prediction);
disp("GPR MAPE: " + round(MAPE * 100, 2) + "% IOA: " + round(IOA * 100, 2) + "% (Bayesian optimization 100 iters)")


%%
function ioa = index_of_agreement(validation, prediction)
    
    % Calculates Index Of Agreement (IOA).
    % :param validation: actual values
    % :param prediction: predicted values
    % :return: IOA float.   
    ioa =  1 - (sum((validation - prediction) .^ 2)) / (sum((abs(prediction - mean(validation)) + abs(validation - mean(validation))) .^ 2));
end