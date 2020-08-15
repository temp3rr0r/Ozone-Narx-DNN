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
%% SVM
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


%% Average GPU DNN training Execution time
boRuns2 = readtable('../../NarxModelSearch/logs/boRuns.csv');
disp(round(mean(diff(table2array(boRuns2(:,1))))/3600, 2) + " hours +/- " + round(std(diff(table2array(boRuns2(:,1)))/3600), 2) + ' hours')

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
%% Spectral Analysis model estimation -> Bode -> Resonance freqs(amplitude) & Phase roll-off (delays)
betn_iddata = iddata(y_train_matrix_normalized, X_train_matrix_normalized, 1);
mdl_spa = spa(betn_iddata); % Spectral analysis model
figure
bode(mdl_spa); % Resonant frequencies (magnitude), Delays (phase roll-off)
% Resonant freq: 0.221 rad/sample
% No phase roll-off (gradual lowering/delay)

%% Impulse response model estimation -> step function -> delay estimation
Mimp = impulseest(betn_iddata, 60);
step(Mimp) % Overshooting & overdumping
% Mostly over-dumping
% u5 has some overshooting
%% Delay estimation (via ARX)
poles = 4;
zeros = 4;
min_delay = 0;
max_delay = 3;
max_tests = 5000;
betn_iddata_100 = iddata(y_train_matrix_normalized(300:360), X_train_matrix_normalized(300:360, :), 1);
delayest(betn_iddata_100, poles, zeros, min_delay, max_delay, max_tests)

%% ARX Model order estimation (all permutations vs MDL/AIC/MinError
disp('ARX Model order estimation, all inputs');
NN1 = struc(1:3, 1:3, 1:3); % Permutations (poles, zeros, delays)
for i = 1:7
    selstruc(arxstruc(betn_iddata(:, :, i), betn_iddata(:, :, i), NN1))
end

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