%% Test error of Bayesian Optimized: SVM, GPR, Tree, Ensemble models

%% Load test data
T = readtable('../../NarxModelSearch/data/6vars/BETN073.csv');
X_test = T(7306:7670, 3:9);
y_test = T(7306:7670, 2);
%% Load training data
% Indices: B3653:I7305
X_train = T(3653:7305, 3:9);
y_train = T(3653:7305, 2);
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
load('best_Tree_Bayes100.mat');
y_test_prediction = trainedTreeBayes100.predictFcn(X_test);
RMSE = sqrt(mean((y_test_prediction - y_test_matrix).^2));  % Root Mean Squared Error
MAE = mean(abs(y_test_prediction - y_test_matrix));
MASE = MAE/MAE_naive_1;
MSE = mean((y_test_prediction - y_test_matrix).^2);  % Mean Squared Error
MAPE = mean((abs(y_test_prediction - y_test_matrix))./y_test_matrix);
sMAPE = symmetric_MAPE(y_test_matrix, y_test_prediction);
IOA = index_of_agreement(y_test_matrix, y_test_prediction);
disp("Tree" + char(10) + " MASE: " + round(MASE, 3) + " sMAPE: " + round(sMAPE * 100, 2) + "% MAPE: " + round(MAPE * 100, 2) + "% IOA: " + round(IOA * 100, 2) + "% MSE: " + round(MSE, 2) + " RMSE: " + round(RMSE, 2) + " MAE: " + round(MAE, 2) + " (Bayesian optimization 100 iters)")
%% Ensemble
load('best_Ensemble_Bayes100.mat');
y_test_prediction = trainedEnsembleBayes100.predictFcn(X_test);
RMSE = sqrt(mean((y_test_prediction - y_test_matrix).^2));  % Root Mean Squared Error
MAE = mean(abs(y_test_prediction - y_test_matrix));
MASE = MAE/MAE_naive_1;
MSE = mean((y_test_prediction - y_test_matrix).^2);  % Mean Squared Error
MAPE = mean((abs(y_test_prediction - y_test_matrix))./y_test_matrix);
sMAPE = symmetric_MAPE(y_test_matrix, y_test_prediction);
IOA = index_of_agreement(y_test_matrix, y_test_prediction);
disp("Ensemble" + char(10) + " MASE: " + round(MASE, 3) + " sMAPE: " + round(sMAPE * 100, 2) + "% MAPE: " + round(MAPE * 100, 2) + "% IOA: " + round(IOA * 100, 2) + "% MSE: " + round(MSE, 2) + " RMSE: " + round(RMSE, 2) + " MAE: " + round(MAE, 2) + " (Bayesian optimization 100 iters)")
%% SVM
load('best_Gaussian_SVM_Bayes100.mat');
y_test_prediction = trainedGaussianSVMBayes100.predictFcn(X_test);
RMSE = sqrt(mean((y_test_prediction - y_test_matrix).^2));  % Root Mean Squared Error
MAE = mean(abs(y_test_prediction - y_test_matrix));
MASE = MAE/MAE_naive_1;
MSE = mean((y_test_prediction - y_test_matrix).^2);  % Mean Squared Error
MAPE = mean((abs(y_test_prediction - y_test_matrix))./y_test_matrix);
sMAPE = symmetric_MAPE(y_test_matrix, y_test_prediction);
IOA = index_of_agreement(y_test_matrix, y_test_prediction);
disp("Gaussian SVM" + char(10) + " MASE: " + round(MASE, 3) + " sMAPE: " + round(sMAPE * 100, 2) + "% MAPE: " + round(MAPE * 100, 2) + "% IOA: " + round(IOA * 100, 2) + "% MSE: " + round(MSE, 2) + " RMSE: " + round(RMSE, 2) + " MAE: " + round(MAE, 2) + " (best kernel)")
%% GPR
load('best_GPR_Bayes100.mat');
y_test_prediction = trainedGPRBayes100.predictFcn(X_test);
RMSE = sqrt(mean((y_test_prediction - y_test_matrix).^2));  % Root Mean Squared Error
MAE = mean(abs(y_test_prediction - y_test_matrix));
MASE = MAE/MAE_naive_1;
MSE = mean((y_test_prediction - y_test_matrix).^2);  % Mean Squared Error
MAPE = mean((abs(y_test_prediction - y_test_matrix))./y_test_matrix);
sMAPE = symmetric_MAPE(y_test_matrix, y_test_prediction);
IOA = index_of_agreement(y_test_matrix, y_test_prediction);
disp("GPR" + char(10) + " MASE: " + round(MASE, 3) + " sMAPE: " + round(sMAPE * 100, 2) + "% MAPE: " + round(MAPE * 100, 2) + "% IOA: " + round(IOA * 100, 2) + "% MSE: " + round(MSE, 2) + " RMSE: " + round(RMSE, 2) + " MAE: " + round(MAE, 2) + " (best kernel out of all)")
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
    trainedLSSVM_6vars = struct('type', type, 'gam', gam, 'sig2', sig2,...
        'kernel', kernel, 'alpha', alpha, 'b', b);
    save('trainedLSSVM_6vars.mat', 'trainedLSSVM_6vars');    
end 
% Test LSSVM
Xs = X_test_matrix_normalized;
Ys = y_test_matrix;
Xs(isnan(Xs))=0; % Remove NaNs
Ys(isnan(Ys))=0;

load('trainedLSSVM_6vars.mat');

Yt = simlssvm({X,Y,trainedLSSVM_6vars.type,trainedLSSVM_6vars.gam, ...
    trainedLSSVM_6vars.sig2,trainedLSSVM_6vars.kernel,'preprocess'},...
    {trainedLSSVM_6vars.alpha, trainedLSSVM_6vars.b},Xs);
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