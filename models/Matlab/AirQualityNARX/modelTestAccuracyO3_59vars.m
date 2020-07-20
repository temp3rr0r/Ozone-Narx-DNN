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

%% LSSVM classification
X = 2.*rand(100,2)-1;
Y = sign(sin(X(:,1))+X(:,2));
gam = 10;
sig2 = 0.4;
type = 'classification';
[alpha, b] = trainlssvm({X, Y, type, gam, sig2, 'RBF_kernel'});
plotlssvm({X,Y,type,gam,sig2, 'RBF_kernel'},{alpha,b});

%% LSSVM regression 0
X = [linspace(-1,1,50) linspace(-1,1,50)]';
Y = (15*(X.^2-1).^2.*X.^4).*exp(-X)+normrnd(0,0.1,length(X),1);
type = 'function estimation';
[gam,sig2] = tunelssvm({X,Y,type,[],[],'RBF_kernel'},'simplex', 'leaveoneoutlssvm',{'mse'});
[alpha,b] = trainlssvm({X,Y,type,gam,sig2,'RBF_kernel'});
plotlssvm({X,Y,type,gam,sig2,'RBF_kernel'},{alpha,b});
%% Train LSSVM regression
X = X_train_matrix(:, :);
Y = y_train_matrix(:);
type = 'f';
[gam,sig2] = tunelssvm({X,Y,type,[],[],'RBF_kernel'},'simplex', 'crossvalidatelssvm',{10, 'mse'});
[alpha,b] = trainlssvm({X,Y,type,gam,sig2,'RBF_kernel'});
%%
plotlssvm({X,Y,type,gam,sig2,'RBF_kernel'},{alpha,b});
%% Train LSSVM
Xs = X_test_matrix;
Ys = y_test_matrix;
Yt = simlssvm({X,Y,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b},Xs);
RMSE = sqrt(mean((Yt - Ys).^2));  % Root Mean Squared Error
MAE = mean(abs(Yt - Ys));  % Root Mean Squared Error
MSE = mean((Yt - Ys).^2);  % Mean Squared Error
MAPE = mean((abs(Yt - Ys))./Ys);
IOA = index_of_agreement(y_test_matrix, y_test_prediction);
disp("LSSVM MAPE: " + round(MAPE * 100, 2) + "% IOA: " + round(IOA * 100, 2) + " MSE: " + round(MSE, 2) + " RMSE: " + round(RMSE, 2) + " MAE: " + round(MAE, 2))
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