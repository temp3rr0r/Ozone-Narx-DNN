%% Test error of Bayesian Optimized: SVM, GPR, Tree, Ensemble models

%% Load test data
T = readtable('../../NarxModelSearch/data/O3_BETN_calendar_1995To2019_single_BETN073/O3_BETN.csv');
X_test = T(7307:7671, 3:62);
y_test = T(7307:7671, 2);
%% Load training data
% Indices: B3654:BJ7306
X_train = T(3654:7306, 3:62);
y_train = T(3654:7306, 2);
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
X_test.Properties.VariableNames{'x10FG'} = 'FG';
X_test.Properties.VariableNames{'x10U'} = 'U';
X_test.Properties.VariableNames{'x2T'} = 'T';
X_test.Properties.VariableNames{'x2D'} = 'D';
X_test.Properties.VariableNames{'O3_BETN073_1'} = 'O3_BETN0731';

%% Naive-1
disp("10-fold cross-validation");
y_test_prediction = [y_test_matrix(1); y_test_matrix(1:end-1)];
MAPE = mean((abs(y_test_prediction - y_test_matrix))./y_test_matrix);
IOA = index_of_agreement(y_test_matrix, y_test_prediction);
disp("Naive-1 MAPE: " + round(MAPE * 100, 2) + "% IOA: " + round(IOA * 100, 2) + "% (Bayesian optimization 100 iters)")
%%
%resid(y_test_matrix, y_test_prediction)

%% Tree
load('trainedTreeBO350_59vars.mat');
y_test_prediction = trainedTreeBO350_59vars.predictFcn(X_test);
MAPE = mean((abs(y_test_prediction - y_test_matrix))./y_test_matrix);
IOA = index_of_agreement(y_test_matrix, y_test_prediction);
disp("Tree MAPE: " + round(MAPE * 100, 2) + "% IOA: " + round(IOA * 100, 2) + "% (Bayesian optimization 100 iters)")

%% Ensemble
load('trainedTreeBO350_59vars.mat');
y_test_prediction = trainedTreeBO350_59vars.predictFcn(X_test);
MAPE = mean((abs(y_test_prediction - y_test_matrix))./y_test_matrix);
IOA = index_of_agreement(y_test_matrix, y_test_prediction);
disp("Ensemble MAPE: " + round(MAPE * 100, 2) + "% IOA: " + round(IOA * 100, 2) + "% (Bayesian optimization 100 iters)")
%% SVM
load('trainedMediumGaussianSVM_59vars.mat');
y_test_prediction = trainedMediumGaussianSVM_59vars.predictFcn(X_test);
MAPE = mean((abs(y_test_prediction - y_test_matrix))./y_test_matrix);
IOA = index_of_agreement(y_test_matrix, y_test_prediction);
disp("SVM MAPE: " + round(MAPE * 100, 2) + "% IOA: " + round(IOA * 100, 2) + "% (Bayesian optimization 100 iters)")

%% GPR
load('trainedRationalQuadraticGPR_59vars.mat');
y_test_prediction = trainedRationalQuadraticGPR_59vars.predictFcn(X_test);
MAPE = mean((abs(y_test_prediction - y_test_matrix))./y_test_matrix);
IOA = index_of_agreement(y_test_matrix, y_test_prediction);
disp("GPR MAPE: " + round(MAPE * 100, 2) + "% IOA: " + round(IOA * 100, 2) + "% (Bayesian optimization 100 iters)")

%% LSSVM
load('trainedMediumGaussianSVM_59vars.mat');
y_test_prediction = trainedMediumGaussianSVM_59vars.predictFcn(X_test);
MAPE = mean((abs(y_test_prediction - y_test_matrix))./y_test_matrix);
IOA = index_of_agreement(y_test_matrix, y_test_prediction);
disp("LSSVM MAPE: " + round(MAPE * 100, 2) + "% IOA: " + round(IOA * 100, 2) + "% (Bayesian optimization 100 iters)")
%% FS-LSSVM
load('trainedMediumGaussianSVM_59vars.mat');
y_test_prediction = trainedMediumGaussianSVM_59vars.predictFcn(X_test);
MAPE = mean((abs(y_test_prediction - y_test_matrix))./y_test_matrix);
IOA = index_of_agreement(y_test_matrix, y_test_prediction);
disp("FS-LSSVM MAPE: " + round(MAPE * 100, 2) + "% IOA: " + round(IOA * 100, 2) + "% (Bayesian optimization 100 iters)")
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