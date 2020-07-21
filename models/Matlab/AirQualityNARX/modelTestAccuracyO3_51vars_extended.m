%% Load test data
%T = readtable('../../NarxModelSearch/data/4stations51vars/BETN_12_66_73_121_51vars_O3_O3-1_19900101To2000101.csv');
%T = readtable('../../NarxModelSearch/data/4stations51vars/BETN_12_66_73_121_51vars_O3_O3-1_19900101To2000101_ts.csv');
%X_test = T(7306:7670, [6:57, 59]);
%y_test = T(7306:7670, 5);
T = readtable('data_51vars_comparisons.csv');
X_test = T(end-365:end, 3:54);
y_test = T(end-365:end, 2);
show_plots = false;
%% Load training data
% Indices: D3653:D7305, F3653:BD7305, BG3653:BG7305
% X_train = T(3653:7305, [6:57, 59]);
% y_train = T(3653:7305, 5);
% Indices: D3653:D7305, F3653:BD7305, BG3653:BG7305
X_train = T(2:end-365, 3:54);
y_train = T(2:end-365, 2);
X_train_matrix = table2array(X_train);
y_train_matrix = table2array(y_train);
X_train_matrix_normalized = normalize(table2array(X_train));
y_train_matrix_normalized = normalize(table2array(y_train));
%%
X_test_matrix = table2array(X_test);
y_test_matrix = table2array(y_test);
X_test_matrix_normalized = normalize(table2array(X_test));
y_test_matrix_normalized = normalize(table2array(y_test));

% %%
% for i = 6:57
%     X_test.Properties.VariableNames{char('Var' + string(i))} = char('VarName' + string(i));
% end 
% %%
% i = 59;
% X_test.Properties.VariableNames{char('Var' + string(i))} = char('VarName' + string(i));
% i = 41;
% X_test.Properties.VariableNames{char('VarName' + string(i))} = char('e05');
% i = 45;
% X_test.Properties.VariableNames{char('VarName' + string(i))} = char('e13');
% i = 49;
% X_test.Properties.VariableNames{char('VarName' + string(i))} = char('e1');
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
%%
if show_plots
    plot(y_test_matrix);
    hold on;
    plot(y_test_prediction);
    legend({'expected', 'predicted'});
end
%% Tree
load('trainedTreeBo350_51vars.mat');
y_test_prediction = trainedTreeBO350_51vars.predictFcn(X_test);
RMSE = sqrt(mean((y_test_prediction - y_test_matrix).^2));  % Root Mean Squared Error
MAE = mean(abs(y_test_prediction - y_test_matrix));
MASE = MAE/MAE_naive_1;
MSE = mean((y_test_prediction - y_test_matrix).^2);  % Mean Squared Error
MAPE = mean((abs(y_test_prediction - y_test_matrix))./y_test_matrix);
sMAPE = symmetric_MAPE(y_test_matrix, y_test_prediction);
IOA = index_of_agreement(y_test_matrix, y_test_prediction);
disp("Tree" + char(10) + " MASE: " + round(MASE, 3) + " sMAPE: " + round(sMAPE * 100, 2) + "% MAPE: " + round(MAPE * 100, 2) + "% IOA: " + round(IOA * 100, 2) + "% MSE: " + round(MSE, 2) + " RMSE: " + round(RMSE, 2) + " MAE: " + round(MAE, 2) + " (Bayesian optimization 350 iters)")
%% Ensemble
load('trainedEnsembleBO350_51vars.mat');
y_test_prediction = trainedEnsembleBO350_51vars.predictFcn(X_test);
RMSE = sqrt(mean((y_test_prediction - y_test_matrix).^2));  % Root Mean Squared Error
MAE = mean(abs(y_test_prediction - y_test_matrix));
MASE = MAE/MAE_naive_1;
MSE = mean((y_test_prediction - y_test_matrix).^2);  % Mean Squared Error
MAPE = mean((abs(y_test_prediction - y_test_matrix))./y_test_matrix);
sMAPE = symmetric_MAPE(y_test_matrix, y_test_prediction);
IOA = index_of_agreement(y_test_matrix, y_test_prediction);
disp("Ensemble" + char(10) + " MASE: " + round(MASE, 3) + " sMAPE: " + round(sMAPE * 100, 2) + "% MAPE: " + round(MAPE * 100, 2) + "% IOA: " + round(IOA * 100, 2) + "% MSE: " + round(MSE, 2) + " RMSE: " + round(RMSE, 2) + " MAE: " + round(MAE, 2) + " (Bayesian optimization 350 iters)")
%% SVM
load('trainedGaussianSVMBO350_51vars.mat');
y_test_prediction = trainedGaussianSVMBO350_51vars.predictFcn(X_test);
RMSE = sqrt(mean((y_test_prediction - y_test_matrix).^2));  % Root Mean Squared Error
MAE = mean(abs(y_test_prediction - y_test_matrix));
MASE = MAE/MAE_naive_1;
MSE = mean((y_test_prediction - y_test_matrix).^2);  % Mean Squared Error
MAPE = mean((abs(y_test_prediction - y_test_matrix))./y_test_matrix);
sMAPE = symmetric_MAPE(y_test_matrix, y_test_prediction);
IOA = index_of_agreement(y_test_matrix, y_test_prediction);
disp("Medium Gaussian SVM" + char(10) + " MASE: " + round(MASE, 3) + " sMAPE: " + round(sMAPE * 100, 2) + "% MAPE: " + round(MAPE * 100, 2) + "% IOA: " + round(IOA * 100, 2) + "% MSE: " + round(MSE, 2) + " RMSE: " + round(RMSE, 2) + " MAE: " + round(MAE, 2) + " (best kernel)")
%% GPR
load('trainedExponentialGPRBO350_51vars.mat');
y_test_prediction = trainedExponentialGPRBO350_51vars.predictFcn(X_test);
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
    trainedLSSVM_51vars = struct('type', type, 'gam', gam, 'sig2', sig2,...
        'kernel', kernel, 'alpha', alpha, 'b', b);
    save('trainedLSSVM_51vars.mat', 'trainedLSSVM_51vars');    
end 
% Test LSSVM
Xs = X_test_matrix_normalized;
Ys = y_test_matrix;
Xs(isnan(Xs))=0; % Remove NaNs
Ys(isnan(Ys))=0;

load('trainedLSSVM_51vars.mat');
type = trainedLSSVM_51vars.type;
gam = trainedLSSVM_51vars.gam;
sig2 = trainedLSSVM_51vars.sig2;
kernel = trainedLSSVM_51vars.kernel;

Yt = simlssvm({X,Y,trainedLSSVM_51vars.type,trainedLSSVM_51vars.gam, ...
    trainedLSSVM_51vars.sig2,trainedLSSVM_51vars.kernel,'preprocess'},...
    {trainedLSSVM_51vars.alpha, trainedLSSVM_51vars.b},Xs);
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
% load('trainedMediumGaussianSVM_59vars.mat');
% y_test_prediction = trainedMediumGaussianSVM_59vars.predictFcn(X_test);
% RMSE = sqrt(mean((y_test_prediction - y_test_matrix).^2));  % Root Mean Squared Error
% MAE = mean(abs(y_test_prediction - y_test_matrix));
% MASE = MAE/MAE_naive_1;
% MSE = mean((y_test_prediction - y_test_matrix).^2);  % Mean Squared Error
% MAPE = mean((abs(y_test_prediction - y_test_matrix))./y_test_matrix);
% sMAPE = symmetric_MAPE(y_test_matrix, y_test_prediction);
% IOA = index_of_agreement(y_test_matrix, y_test_prediction);
% disp("FS-LSSVM" + char(10) + " MASE: " + round(MASE, 3) + " sMAPE: " + round(sMAPE * 100, 2) + "% MAPE: " + round(MAPE * 100, 2) + "% IOA: " + round(IOA * 100, 2) + "% MSE: " + round(MSE, 2) + " RMSE: " + round(RMSE, 2) + " MAE: " + round(MAE, 2))

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