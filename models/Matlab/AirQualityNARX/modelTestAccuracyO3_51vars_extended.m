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
% Indices: B2:BB3654
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
%%
for i = 3:54
    X_test.Properties.VariableNames{char('Var' + string(i))} = char('VarName' + string(i));
end 
%% Naive-1
disp("10-fold cross-validation");
y_test_prediction = [y_test_matrix(1); y_test_matrix(1:end-1)];
MAE = mean(abs(y_test_prediction - y_test_matrix));
MAE_naive_1 = MAE;
print_results(y_test_prediction, y_test_matrix, MAE_naive_1, "Naive-1");
if show_plots
    plot_auto_correlations(y_test_prediction, y_test_matrix, "Naive-1");
end
%%
if show_plots
    plot(y_test_matrix);
    hold on;
    plot(y_test_prediction);
    legend({'expected', 'predicted'});
end
%% BO Ensemble
load('trainedEnsembleBO350_51vars.mat');
y_test_prediction = trainedEnsembleBO350_51vars.predictFcn(X_test);
print_results(y_test_prediction, y_test_matrix, MAE_naive_1, "BO Ensemble (Bayesian optimization 350 iters)");
writematrix(y_test_prediction,'ensemble_y_test_prediction.csv')
writematrix(y_test_matrix,'ensemble_y_test_matrix.csv')
if show_plots
    plot_auto_correlations(y_test_prediction, y_test_matrix, "BO Ensemble");
end
%% Specific Ensemble
load('trainedBaggedTrees_51vars.mat');
y_test_prediction = trainedBaggedTrees_51vars.predictFcn(X_test);
print_results(y_test_prediction, y_test_matrix, MAE_naive_1, "Bagged Trees (best single ensemble)");

if show_plots
    plot_auto_correlations(y_test_prediction, y_test_matrix, "Bagged Trees");
end
%% Specific SVM
load('trainedMediumGaussianSVM_51vars.mat');
y_test_prediction = trainedMediumGaussianSVM_51vars.predictFcn(X_test);
print_results(y_test_prediction, y_test_matrix, MAE_naive_1, "Medium Gaussian SVM (best single SVM)");
if show_plots
    plot_auto_correlations(y_test_prediction, y_test_matrix, "Medium Gaussian SVM");
end
%% BO SVM
load('trainedSVMBO350_51vars.mat');
y_test_prediction = trainedSVMBO350_51vars.predictFcn(X_test);
print_results(y_test_prediction, y_test_matrix, MAE_naive_1, "BO SVM (Bayesian optimization 350 iters)");
% plot_auto_correlations(y_test_prediction, y_test_matrix);
%% Specific Linear regression
load('trainedInteractionLInearRegression_51vars.mat');
y_test_prediction = trainedInteractionLInearRegression_51vars.predictFcn(X_test);
print_results(y_test_prediction, y_test_matrix, MAE_naive_1, "Interactions Linear Regression (best single Linear Regression)");
if show_plots
    plot_auto_correlations(y_test_prediction, y_test_matrix, "Interactions Linear Regression");
end
%% Coarse Tree
load('trainedCoarseTree_51vars.mat');
y_test_prediction = trainedCoarseTree_51vars.predictFcn(X_test);
print_results(y_test_prediction, y_test_matrix, MAE_naive_1, "Coarse Tree (best single Tree)");
if show_plots
    plot_auto_correlations(y_test_prediction, y_test_matrix, "Coarse Tree");
end
%% Specific GPR
load('trainedExponentialGPR_51vars.mat');
y_test_prediction = trainedExponentialGPR_51vars.predictFcn(X_test);
print_results(y_test_prediction, y_test_matrix, MAE_naive_1, "Exponential GPR (best single GPR)");
if show_plots
    plot_auto_correlations(y_test_prediction, y_test_matrix, "Exponential GPR");
end
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

print_results(Yt, y_test_matrix, MAE_naive_1, "LSSVM (best single GPR)");
if show_plots
    plot_auto_correlations(Yt, y_test_matrix, "LSSVM");
end

writematrix(Yt,'LSSVM_y_test_prediction.csv')
writematrix(y_test_matrix,'LSSVM_y_test_matrix.csv')

% %% FS-LSSVM TODO
% train = false;
% addpath('LSSVMlabv1_8_R2009b_R2011a');
% 
% X = X_train_matrix_normalized;
% Y = y_train_matrix_normalized;
% X(isnan(X)) = 0; % Remove NaNs (from division with std of zero)
% Y(isnan(Y)) = 0;
% 
% if train == true
%     % Train FS-LSSVM regression
%     type = 'f';
%     kernel = 'RBF_kernel';
%     [gam,sig2] = tunelssvm({X,Y,type,[],[],kernel}, 'simplex', 'crossvalidatelssvm',{10, 'mse'});
%     [alpha,b] = trainlssvm({X,Y,type,gam,sig2,'RBF_kernel'});
%     if show_plots
%         plotlssvm({X,Y,type,gam,sig2,'RBF_kernel'},{alpha,b});
%     end
%     % Store model as struct
%     trainedLSSVM_51vars = struct('type', type, 'gam', gam, 'sig2', sig2,...
%         'kernel', kernel, 'alpha', alpha, 'b', b);
%     save('trainedLSSVM_51vars.mat', 'trainedLSSVM_51vars');    
% end 
% % Test LSSVM
% Xs = X_test_matrix_normalized;
% Ys = y_test_matrix;
% Xs(isnan(Xs))=0; % Remove NaNs
% Ys(isnan(Ys))=0;
% 
% load('trainedLSSVM_51vars.mat');
% type = trainedLSSVM_51vars.type;
% gam = trainedLSSVM_51vars.gam;
% sig2 = trainedLSSVM_51vars.sig2;
% kernel = trainedLSSVM_51vars.kernel;
% 
% Yt = simlssvm({X,Y,trainedLSSVM_51vars.type,trainedLSSVM_51vars.gam, ...
%     trainedLSSVM_51vars.sig2,trainedLSSVM_51vars.kernel,'preprocess'},...
%     {trainedLSSVM_51vars.alpha, trainedLSSVM_51vars.b},Xs);
% Yt = Yt .* std(y_train_matrix) + mean(y_train_matrix); % Remove standardization
% if show_plots
%     plot(1:length(Yt), Ys, 1:length(Yt), Yt);
% end
% RMSE = sqrt(mean((Yt - Ys).^2));  % Root Mean Squared Error
% MAE = mean(abs(Yt - Ys));
% MASE = MAE/MAE_naive_1;
% MSE = mean((Yt - Ys).^2);  % Mean Squared Error
% MAPE = mean((abs(Yt - Ys))./Ys);
% sMAPE = symmetric_MAPE(Ys, Yt);
% IOA = index_of_agreement(y_test_matrix, y_test_prediction);
% disp("LSSVM" + char(10) + " MASE: " + round(MASE, 3) + " sMAPE: " + round(sMAPE * 100, 2) + "% MAPE: " + round(MAPE * 100, 2) + "% IOA: " + round(IOA * 100, 2) + "% MSE: " + round(MSE, 2) + " RMSE: " + round(RMSE, 2) + " MAE: " + round(MAE, 2))
% %%
% % X,Y contains the dataset, svX is a subset of X
% X = X_train_matrix_normalized;
% Y = y_train_matrix_normalized;
% X(isnan(X)) = 0; % Remove NaNs (from division with std of zero)
% Y(isnan(Y)) = 0;
% sig2 = 1;
% 
% Nc = 15;
% svX=X(1:Nc,:);
% S=ceil(length(X)*rand(1));
% Sc=ceil(Nc*rand(1));
% svX(Sc,:) = X(S,:);
% 
% features = AFEm(svX, 'RBF_kernel', sig2, X); % Automatic Feature Extraction by Nystrom method
% [Cl3, gam_optimal] = bay_rr(features,Y,1,3); % Bayesian Inference for linear ridge regression
% [W,b] = ridgeregress(features, Y, gam_optimal); % Linear ridge regression
% Yh = features *W + b;
% 
% figure;
% plot(Y);
% hold on;
% plot(Yh);
% hold off;
% %%
% % X,Y contains the dataset, svX is a subset of X
% X = X_train_matrix_normalized;
% Y = y_train_matrix_normalized;
% X(isnan(X)) = 0; % Remove NaNs (from division with std of zero)
% Y(isnan(Y)) = 0;
% 
% caps = [10 20 50 100 200]; % Capacity subset: # support vectors
% sig2s = [.1 .2 .5 1 2 4 10];
% nb = 10; % # Eigenvalues
% best_performance = mse(Y-0);
% for i=1:length(caps)
%     for j=1:length(sig2s)
%         for t = 1:nb
%             
%             Nc = caps(i);
%             svX=X(1:Nc,:);
%             S=ceil(length(X)*rand(1));
%             Sc=ceil(Nc*rand(1));
%             svX(Sc,:) = X(S,:);
%             
%             features = AFEm(svX, 'RBF_kernel', sig2s(j), X); % Automatic Feature Extraction by Nystrom method
%             [Cl3, gam_optimal] = bay_rr(features,Y,1,3); % Bayesian Inference for linear ridge regression
%             [W,b] = ridgeregress(features, Y, gam_optimal); % Linear ridge regression
%             Yh = features *W + b;
%             performances(t) = mse(Y - Yh);
%         end
%         minimal_performances(i,j) = mean(performances);
%         if minimal_performances(i,j) < best_performance
%            cap_best = caps(i);
%            sig2_best = sig2s(j);
%            features_best = features;
%            W_best = W;
%            b_best = b;
%            Yh_best = Yh;
%         end        
%     end
% end
% 
% mse(Y - Yh)
% figure;
% plot(Y);
% hold on;
% plot(Yh);
% hold off;
% 
% mse(Y - Yh_best)
% figure;
% plot(Y);
% hold on;
% plot(Yh_best);
% hold off;
% 
% 
% %%
% X = X_train_matrix_normalized;
% Y = y_train_matrix_normalized;
% X(isnan(X)) = 0; % Remove NaNs (from division with std of zero)
% Y(isnan(Y)) = 0;
% x = X(:, :);
% y = Y(:);
% 
% Xs = X_test_matrix_normalized;
% Ys = y_test_matrix_normalized;
% Xs(isnan(Xs))=0; % Remove NaNs
% Ys(isnan(Ys))=0;
% x0 = Xs(:, :);
% y0 = Ys(:);
% 
% kernel = 'RBF_kernel';
% sigma2=.75;
% gamma=1;
% crit_old=-inf;
% Nc=30;
% Xs=x(1:Nc,:);
% Ys=y(1:Nc,:);
% 
% tv = 1;
% for tel=1:length(x)
%   % new candidate set
%   Xsp=Xs; Ysp=Ys;
%   S=ceil(length(x)*rand(1));
%   Sc=ceil(Nc*rand(1));
%   Xs(Sc,:) = x(S,:);
%   Ys(Sc,:) = y(S);
%   Ncc=Nc;
%   % automaticly extract features and compute entropy
%   crit = kentropy(Xs,kernel, sigma2);  
%   if crit <= crit_old
%     crit = crit_old;
%     Xs=Xsp;
%     Ys=Ysp;
%   else
%     crit_old = crit;
%     % ridge regression    
%     [features,U,lam] = AFEm(Xs,kernel, sigma2,x);
%     [w, b, Yh] = ridgeregress(features,y,gamma,features);
%     % make-a-plot
% %     plot(x,y,'*'); hold on
% %     plot(x,Yh,'r-')
% %     plot(Xs,Ys,'go','Linewidth',7)
% %     xlabel('X'); ylabel('Y'); 
% %     title(['Approximation by fixed size LS-SVM based on maximal entropy: ' num2str(crit)]);
% %     hold off;  drawnow
%     plot(y,'*'); hold on
%     plot(Yh,'r-')
%     plot(Ys,'go','Linewidth',7)
%     xlabel('t'); ylabel('Y'); 
%     title(['Approximation by fixed size LS-SVM based on maximal entropy: ' num2str(crit)]);
%     hold off;  drawnow
%   end
% end
% 
% features = AFEm(Xs,kernel, sigma2,x);    
% 
% try
%   [CostL3, gamma_optimal] = bay_rr(features,y,gamma,3);
% catch
%   gamma_optimal = gamma;
% end
% 
% [w,b] = ridgeregress(features,y,gamma_optimal);
% Yh0 = AFEm(Xs,kernel, sigma2,x0)*w+b;
% echo off;         
% 
% % plot(x,y,'*'); hold on
% % plot(x0,Yh0,'r-')
% % plot(Xs,Ys,'go','Linewidth',7)
% % xlabel('X'); ylabel('Y'); 
% % title(['Approximation by fixed size LS-SVM based on maximal entropy: ' num2str(crit)]);
% % hold off;  
% %
% plot(y,'*'); hold on
% plot(Yh0,'r-')
% plot(Ys,'go','Linewidth',7)
% xlabel('t'); ylabel('Y'); 
% title(['Approximation by fixed size LS-SVM based on maximal entropy: ' num2str(crit)]);
% hold off;  
% %
% figure
% plot(y,'*'); hold on
% plot(Yh,'r-')
% hold off;  
% %
% figure
% plot(y0,'*'); hold on
% plot(Yh0,'r-')
% hold off;  
% 
% %%
% x = sort(2.*randn(2000,1));
% x0 = sort(2.*randn(2000,1));
% 
% eval('y = sinc(x)+0.05.*randn(length(x),1);',...
%      'y = sin(pi.*x+12345*eps)./(pi*x+12345*eps)+0.05.*randn(length(x),1);');
% eval('y0 = sinc(x0)+0.05.*randn(length(x0),1);',...
%      'y0 = sin(pi.*x0+12345*eps)./(pi*x0+12345*eps)+0.05.*randn(length(x0),1);');
% 
% kernel = 'RBF_kernel';
% sigma2=.75;
% gamma=1;
% crit_old=-inf;
% Nc=15;
% Xs=x(1:Nc,:);
% Ys=y(1:Nc,:);
% 
% tv = 1;
% for tel=1:length(x)
%   % new candidate set
%   Xsp=Xs; Ysp=Ys;
%   S=ceil(length(x)*rand(1));
%   Sc=ceil(Nc*rand(1));
%   Xs(Sc,:) = x(S,:);
%   Ys(Sc,:) = y(S);
%   Ncc=Nc;
%   % automaticly extract features and compute entropy
%   crit = kentropy(Xs,kernel, sigma2);  
%   if crit <= crit_old
%     crit = crit_old;
%     Xs=Xsp;
%     Ys=Ysp;
%   else
%     crit_old = crit;
%     % ridge regression    
%     [features,U,lam] = AFEm(Xs,kernel, sigma2,x);
%     [w, b, Yh] = ridgeregress(features,y,gamma,features);
%     % make-a-plot
%     plot(x,y,'*'); hold on
%     plot(x,Yh,'r-')
%     plot(Xs,Ys,'go','Linewidth',7)
%     xlabel('X'); ylabel('Y'); 
%     title(['Approximation by fixed size LS-SVM based on maximal entropy: ' num2str(crit)]);
%     hold off;  drawnow
%   end
% end
% 
% features = AFEm(Xs,kernel, sigma2,x);    
% 
% try
%   [CostL3, gamma_optimal] = bay_rr(features,y,gamma,3);
% catch
%   gamma_optimal = gamma;
% end
% 
% [w,b] = ridgeregress(features,y,gamma_optimal);
% Yh0 = AFEm(Xs,kernel, sigma2,x0)*w+b;
% echo off;         
% 
% plot(x,y,'*'); hold on
% plot(x0,Yh0,'r-')
% plot(Xs,Ys,'go','Linewidth',7)
% xlabel('X'); ylabel('Y'); 
% title(['Approximation by fixed size LS-SVM based on maximal entropy: ' num2str(crit)]);
% hold off;  
%%
% % close all;
%%
function plot_auto_correlations(y_test_prediction, y_test_matrix, model_name)

    figure;
    subplot(3,2,1)
    autocorr(y_test_matrix);
    title(model_name + ": Expected Autocorrelation");
    subplot(3,2,2)
    parcorr(y_test_matrix);
    title(model_name + ": Expected Partial AutoCorrelation Function (PACF)");
    
    subplot(3,2,3)
    autocorr(y_test_prediction);
    title(model_name + ": Prediction Autocorrelation");
    subplot(3,2,4)
    parcorr(y_test_prediction);
    title(model_name + ": Prediction Partial AutoCorrelation Function (PACF)");
    
    resids = y_test_prediction - y_test_matrix;    
    subplot(3,2,5)
    autocorr(resids);
    title(model_name + ": Residuals Autocorrelation");
    subplot(3,2,6)
    parcorr(resids);
    title(model_name + ": Residuals Partial AutoCorrelation Function (PACF)");
end


function print_results(y_test_prediction, y_test_matrix, MAE_naive_1, model_name)
    RMSE = sqrt(mean((y_test_prediction - y_test_matrix).^2));  % Root Mean Squared Error
    MAE = mean(abs(y_test_prediction - y_test_matrix));    
    epsilon_t = y_test_matrix - y_test_prediction;    
    MASE = mean(abs(epsilon_t/MAE_naive_1));    
    MSE = mean((y_test_prediction - y_test_matrix).^2);  % Mean Squared Error
    MAPE = mean((abs(y_test_prediction - y_test_matrix))./y_test_matrix);
    sMAPE = symmetric_MAPE(y_test_matrix, y_test_prediction);
    IOA = index_of_agreement(y_test_matrix, y_test_prediction);
    disp(model_name);
    disp(" MASE: " + round(MASE, 3) + " sMAPE: " + round(sMAPE * 100, 2)...
        + "% MAPE: " + round(MAPE * 100, 2) + "% IOA: " + ...
        round(IOA * 100, 2) + "% MSE: " + round(MSE, 2) + " RMSE: " + ...
        round(RMSE, 2) + " MAE: " + round(MAE, 2))
end

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