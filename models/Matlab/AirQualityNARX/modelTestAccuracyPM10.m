%% Test error of Bayesian Optimized: SVM, GPR, Tree, Ensemble models

%% Load data
T = readtable('PM10_BETN_1995To2019\PM10_BETN.csv');
%% data
% 2012: 6211:6576, 2018: 8403:8767
% 2 for BETN43 12 for BETN85
y_test_BETN43_2012 = T(6210:6576, 2);
y_test_BETN85_2012 = T(6210:6576, 12);
y_test_BETN85_2018 = T(8403:8766, 2);
y_test_BETN43_2018 = T(8403:8766, 12);

%%
naive1(y_test_BETN43_2012, 2012, "BETN43");
naive1(y_test_BETN85_2012, 2012, "BETN85");
naive1(y_test_BETN43_2018, 2018, "BETN43");
naive1(y_test_BETN85_2018, 2018, "BETN85");

%% Naive-1
function naive1(y_test, year, station) 
    y_test_matrix = table2array(y_test);
    y_test_prediction = [y_test_matrix(1); y_test_matrix(1:end-1)];    
    y_test_matrix = y_test_matrix(2:end);
    y_test_prediction = y_test_prediction(2:end);    
    MSE = mse(y_test_prediction - y_test_matrix);
    MAPE = mean((abs(y_test_prediction - y_test_matrix))./y_test_matrix);
    sMAPE = mean(2.*abs(y_test_matrix-y_test_prediction) ./ (abs(y_test_matrix) + abs(y_test_prediction)));
    IOA = index_of_agreement(y_test_matrix, y_test_prediction);
    disp("Naive-1 (" + year + ", " + station + ") MAPE: " + round(MAPE * 100, 2) + "%, sMAPE: " + round(sMAPE * 100, 2) + "%, MSE: " + round(MSE, 2) + " IOA: " + round(IOA * 100, 2) + "%")
end
%%

function ioa = index_of_agreement(validation, prediction)
    
    % Calculates Index Of Agreement (IOA).
    % :param validation: actual values
    % :param prediction: predicted values
    % :return: IOA float.   
    ioa =  1 - (sum((validation - prediction) .^ 2)) / (sum((abs(prediction - mean(validation)) + abs(validation - mean(validation))) .^ 2));
end