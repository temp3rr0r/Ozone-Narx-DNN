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
RMSE = sqrt(mean((y_test_prediction - y_test_matrix).^2));  % Root Mean Squared Error
MAE = mean(abs(y_test_prediction - y_test_matrix));
MAE_naive_1 = MAE;
MASE = MAE/MAE_naive_1;
MSE = mean((y_test_prediction - y_test_matrix).^2);  % Mean Squared Error
MAPE = mean((abs(y_test_prediction - y_test_matrix))./y_test_matrix);
sMAPE = symmetric_MAPE(y_test_matrix, y_test_prediction);
IOA = index_of_agreement(y_test_matrix, y_test_prediction);
disp("Naive-1" + char(10) + " MASE: " + round(MASE, 3) + " sMAPE: " + round(sMAPE * 100, 2) + "% MAPE: " + round(MAPE * 100, 2) + "% IOA: " + round(IOA * 100, 2) + "% MSE: " + round(MSE, 2) + " RMSE: " + round(RMSE, 2) + " MAE: " + round(MAE, 2))
%% Tree
load('trainedTreeBO350_59vars.mat');
y_test_prediction = trainedTreeBO350_59vars.predictFcn(X_test);
RMSE = sqrt(mean((y_test_prediction - y_test_matrix).^2));  % Root Mean Squared Error
MAE = mean(abs(y_test_prediction - y_test_matrix));
MASE = MAE/MAE_naive_1;
MSE = mean((y_test_prediction - y_test_matrix).^2);  % Mean Squared Error
MAPE = mean((abs(y_test_prediction - y_test_matrix))./y_test_matrix);
sMAPE = symmetric_MAPE(y_test_matrix, y_test_prediction);
IOA = index_of_agreement(y_test_matrix, y_test_prediction);
disp("Tree" + char(10) + " MASE: " + round(MASE, 3) + " sMAPE: " + round(sMAPE * 100, 2) + "% MAPE: " + round(MAPE * 100, 2) + "% IOA: " + round(IOA * 100, 2) + "% MSE: " + round(MSE, 2) + " RMSE: " + round(RMSE, 2) + " MAE: " + round(MAE, 2) + " (Bayesian optimization 350 iters)")
%% Ensemble
load('trainedEnsembleBO350_59vars.mat');
y_test_prediction = trainedEnsembleBO350_59vars.predictFcn(X_test);
RMSE = sqrt(mean((y_test_prediction - y_test_matrix).^2));  % Root Mean Squared Error
MAE = mean(abs(y_test_prediction - y_test_matrix));
MASE = MAE/MAE_naive_1;
MSE = mean((y_test_prediction - y_test_matrix).^2);  % Mean Squared Error
MAPE = mean((abs(y_test_prediction - y_test_matrix))./y_test_matrix);
sMAPE = symmetric_MAPE(y_test_matrix, y_test_prediction);
IOA = index_of_agreement(y_test_matrix, y_test_prediction);
disp("Ensemble" + char(10) + " MASE: " + round(MASE, 3) + " sMAPE: " + round(sMAPE * 100, 2) + "% MAPE: " + round(MAPE * 100, 2) + "% IOA: " + round(IOA * 100, 2) + "% MSE: " + round(MSE, 2) + " RMSE: " + round(RMSE, 2) + " MAE: " + round(MAE, 2) + " (Bayesian optimization 350 iters)")

writematrix(y_test_prediction,'ensemble_y_test_prediction.csv') 
%% SVM
load('trainedMediumGaussianSVM_59vars.mat');
y_test_prediction = trainedMediumGaussianSVM_59vars.predictFcn(X_test);
RMSE = sqrt(mean((y_test_prediction - y_test_matrix).^2));  % Root Mean Squared Error
MAE = mean(abs(y_test_prediction - y_test_matrix));
MASE = MAE/MAE_naive_1;
MSE = mean((y_test_prediction - y_test_matrix).^2);  % Mean Squared Error
MAPE = mean((abs(y_test_prediction - y_test_matrix))./y_test_matrix);
sMAPE = symmetric_MAPE(y_test_matrix, y_test_prediction);
IOA = index_of_agreement(y_test_matrix, y_test_prediction);
disp("Medium Gaussian SVM" + char(10) + " MASE: " + round(MASE, 3) + " sMAPE: " + round(sMAPE * 100, 2) + "% MAPE: " + round(MAPE * 100, 2) + "% IOA: " + round(IOA * 100, 2) + "% MSE: " + round(MSE, 2) + " RMSE: " + round(RMSE, 2) + " MAE: " + round(MAE, 2) + " (best kernel)")
%% GPR
load('trainedRationalQuadraticGPR_59vars.mat');
y_test_prediction = trainedRationalQuadraticGPR_59vars.predictFcn(X_test);
RMSE = sqrt(mean((y_test_prediction - y_test_matrix).^2));  % Root Mean Squared Error
MAE = mean(abs(y_test_prediction - y_test_matrix));
MASE = MAE/MAE_naive_1;
MSE = mean((y_test_prediction - y_test_matrix).^2);  % Mean Squared Error
MAPE = mean((abs(y_test_prediction - y_test_matrix))./y_test_matrix);
sMAPE = symmetric_MAPE(y_test_matrix, y_test_prediction);
IOA = index_of_agreement(y_test_matrix, y_test_prediction);
disp("Rational Quadratic GPR" + char(10) + " MASE: " + round(MASE, 3) + " sMAPE: " + round(sMAPE * 100, 2) + "% MAPE: " + round(MAPE * 100, 2) + "% IOA: " + round(IOA * 100, 2) + "% MSE: " + round(MSE, 2) + " RMSE: " + round(RMSE, 2) + " MAE: " + round(MAE, 2) + " (best kernel out of all)")
%% LSSVM
train = false;
show_plots = false;
addpath('LSSVMlabv1_8_R2009b_R2011a');

X = X_train_matrix_normalized;
Y = y_train_matrix_normalized;
X(isnan(X)) = 0; % Remove NaNs (from division with std of zero)
Y(isnan(Y)) = 0;

if train == true
    % Train LSSVM regression
    type = 'f';
    kernel = 'RBF_kernel';
    [gam,sig2] = tunelssvm({X,Y,type,[],[],kernel}, 'simplex', 'crossvalidatelssvm',{10, 'mse'});
    [alpha,b] = trainlssvm({X,Y,type,gam,sig2,'RBF_kernel'});
    if show_plots
        plotlssvm({X,Y,type,gam,sig2,'RBF_kernel'},{alpha,b});
    end
    % Store model as struct
    trainedLSSVM_59vars = struct('type', type, 'gam', gam, 'sig2', sig2,...
        'kernel', kernel, 'alpha', alpha, 'b', b);
    save('trainedLSSVM_59vars.mat', 'trainedLSSVM_59vars');    
end 
% Test LSSVM
Xs = X_test_matrix_normalized;
Ys = y_test_matrix;
Xs(isnan(Xs))=0; % Remove NaNs
Ys(isnan(Ys))=0;

load('trainedLSSVM_59vars.mat');
type = trainedLSSVM_59vars.type;
gam = trainedLSSVM_59vars.gam;
sig2 = trainedLSSVM_59vars.sig2;
kernel = trainedLSSVM_59vars.kernel;

Yt = simlssvm({X,Y,trainedLSSVM_59vars.type,trainedLSSVM_59vars.gam, ...
    trainedLSSVM_59vars.sig2,trainedLSSVM_59vars.kernel,'preprocess'},...
    {trainedLSSVM_59vars.alpha, trainedLSSVM_59vars.b},Xs);
Yt = Yt .* std(y_train_matrix) + mean(y_train_matrix); % Remove standardization
if show_plots
    plot(1:length(Yt), Ys, 1:length(Yt), Yt);
end
RMSE = sqrt(mean((Yt - Ys).^2));  % Root Mean Squared Error
MAE = mean(abs(Yt - Ys));
MASE = MAE/MAE_naive_1;
MSE = mean((Yt - Ys).^2);  % Mean Squared Error
MAPE = mean((abs(Yt - Ys))./Ys);
sMAPE = symmetric_MAPE(Ys, Yt);
IOA = index_of_agreement(y_test_matrix, y_test_prediction);
disp("LSSVM" + char(10) + " MASE: " + round(MASE, 3) + " sMAPE: " + round(sMAPE * 100, 2) + "% MAPE: " + round(MAPE * 100, 2) + "% IOA: " + round(IOA * 100, 2) + "% MSE: " + round(MSE, 2) + " RMSE: " + round(RMSE, 2) + " MAE: " + round(MAE, 2))
%% FS-LSSVM
load('trainedMediumGaussianSVM_59vars.mat');
y_test_prediction = trainedMediumGaussianSVM_59vars.predictFcn(X_test);
RMSE = sqrt(mean((y_test_prediction - y_test_matrix).^2));  % Root Mean Squared Error
MAE = mean(abs(y_test_prediction - y_test_matrix));
MASE = MAE/MAE_naive_1;
MSE = mean((y_test_prediction - y_test_matrix).^2);  % Mean Squared Error
MAPE = mean((abs(y_test_prediction - y_test_matrix))./y_test_matrix);
sMAPE = symmetric_MAPE(y_test_matrix, y_test_prediction);
IOA = index_of_agreement(y_test_matrix, y_test_prediction);
disp("FS-LSSVM" + char(10) + " MASE: " + round(MASE, 3) + " sMAPE: " + round(sMAPE * 100, 2) + "% MAPE: " + round(MAPE * 100, 2) + "% IOA: " + round(IOA * 100, 2) + "% MSE: " + round(MSE, 2) + " RMSE: " + round(RMSE, 2) + " MAE: " + round(MAE, 2))
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
function smape = symmetric_MAPE(y, f)
    smape = 2.0*mean(abs(y-f)./(abs(y)+abs(f)));
end

function ioa = index_of_agreement(validation, prediction)
    
    % Calculates Index Of Agreement (IOA).
    % :param validation: actual values
    % :param prediction: predicted values
    % :return: IOA float.   
    ioa =  1 - (sum((validation - prediction) .^ 2)) / (sum((abs(prediction - mean(validation)) + abs(validation - mean(validation))) .^ 2));
end