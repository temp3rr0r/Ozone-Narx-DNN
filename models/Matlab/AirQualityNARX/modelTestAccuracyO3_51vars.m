%% Test error of Bayesian Optimized: SVM, GPR, Tree, Ensemble models

%% Load test data
T = readtable('../../NarxModelSearch/data/4stations51vars/BETN_12_66_73_121_51vars_O3_O3-1_19900101To2000101.csv');
X_test = T(7306:7670, [6:57, 59]);
y_test = T(7306:7670, 5);
%% Load training data
% Indices: D3653:D7305, F3653:BD7305, BG3653:BG7305
X_train = T(3653:7305, [6:57, 59]);
y_train = T(3653:7305, 5);
X_train_matrix = table2array(X_train);
y_train_matrix = table2array(y_train);
X_train_matrix_normalized = normalize(table2array(X_train));
y_train_matrix_normalized = normalize(table2array(y_train));
%%
X_test_matrix = table2array(X_test);
y_test_matrix = table2array(y_test);
X_test_matrix_normalized = normalize(table2array(X_test));
y_test_matrix_normalized = normalize(table2array(y_test));

%%
for i = 6:57
    X_test.Properties.VariableNames{char('Var' + string(i))} = char('VarName' + string(i));
end 
%%
i = 59;
X_test.Properties.VariableNames{char('Var' + string(i))} = char('VarName' + string(i));
i = 41;
X_test.Properties.VariableNames{char('VarName' + string(i))} = char('e05');
i = 45;
X_test.Properties.VariableNames{char('VarName' + string(i))} = char('e13');
i = 49;
X_test.Properties.VariableNames{char('VarName' + string(i))} = char('e1');

%% Naive-1
disp("10-fold cross-validation");
y_test_prediction = [y_test_matrix(1); y_test_matrix(1:end-1)];
MAPE = mean((abs(y_test_prediction - y_test_matrix))./y_test_matrix);
IOA = index_of_agreement(y_test_matrix, y_test_prediction);
disp("Naive-1 MAPE: " + round(MAPE * 100, 2) + "% IOA: " + round(IOA * 100, 2) + "% (Bayesian optimization 100 iters)")

%% Tree
load('trainedTreeBo350_51vars.mat');
y_test_prediction = trainedTreeBO350_51vars.predictFcn(X_test);
MAPE = mean((abs(y_test_prediction - y_test_matrix))./y_test_matrix);
IOA = index_of_agreement(y_test_matrix, y_test_prediction);
disp("Tree MAPE: " + round(MAPE * 100, 2) + "% IOA: " + round(IOA * 100, 2) + "% (Bayesian optimization 100 iters)")

%% Ensemble
load('trainedEnsembleBO350_51vars.mat');
y_test_prediction = trainedEnsembleBO350_51vars.predictFcn(X_test);
MAPE = mean((abs(y_test_prediction - y_test_matrix))./y_test_matrix);
IOA = index_of_agreement(y_test_matrix, y_test_prediction);
disp("Ensemble MAPE: " + round(MAPE * 100, 2) + "% IOA: " + round(IOA * 100, 2) + "% (Bayesian optimization 100 iters)")
%% Gaussian SVM
load('trainedGaussianSVMBO350_51vars.mat');
y_test_prediction = trainedGaussianSVMBO350_51vars.predictFcn(X_test);
MAPE = mean((abs(y_test_prediction - y_test_matrix))./y_test_matrix);
IOA = index_of_agreement(y_test_matrix, y_test_prediction);
disp("Gaussian SVM MAPE: " + round(MAPE * 100, 2) + "% IOA: " + round(IOA * 100, 2) + "% (Bayesian optimization 100 iters)")
%% GPR
load('trainedExponentialGPRBO350_51vars.mat');
y_test_prediction = trainedExponentialGPRBO350_51vars.predictFcn(X_test);
MAPE = mean((abs(y_test_prediction - y_test_matrix))./y_test_matrix);
IOA = index_of_agreement(y_test_matrix, y_test_prediction);
disp("GPR MAPE: " + round(MAPE * 100, 2) + "% IOA: " + round(IOA * 100, 2) + "% (Bayesian optimization 100 iters)")
%% Train/test autocorrelation & partial correlation
figure;
subplot(2,1,1)
autocorr(y_train_matrix_normalized);
subplot(2,1,2)
parcorr(y_train_matrix_normalized);
figure;
subplot(2,1,1)
autocorr(y_test_matrix_normalized);
subplot(2,1,2)
parcorr(y_test_matrix_normalized);
%% Train/test difference
figure;
subplot(2,1,1);
plot(diff(y_train_matrix_normalized));
title("Train difference");
subplot(2,1,2)
plot(diff(y_test_matrix_normalized));
title("Test difference");
%% Train/test DIFFERENCE autocorrelation & partial correlation
figure;
subplot(2,1,1)
autocorr(diff(y_train_matrix_normalized));
subplot(2,1,2)
parcorr(diff(y_train_matrix_normalized));
figure;
subplot(2,1,1)
autocorr(diff(y_test_matrix_normalized));
subplot(2,1,2)
parcorr(diff(y_test_matrix_normalized));
%%
close all;
%%
function ioa = index_of_agreement(validation, prediction)
    
    % Calculates Index Of Agreement (IOA).
    % :param validation: actual values
    % :param prediction: predicted values
    % :return: IOA float.   
    ioa =  1 - (sum((validation - prediction) .^ 2)) / (sum((abs(prediction - mean(validation)) + abs(validation - mean(validation))) .^ 2));
end