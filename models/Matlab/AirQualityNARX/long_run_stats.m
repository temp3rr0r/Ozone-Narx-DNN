%%

% Max islands global search (4000 iters)
island_train_18islands_4000 = [75.43];
island_test_18islands_4000 = [76.29];

% Max islands (400 iters) global search (1000 iters)
island_train_18islands_1000LS = [75.13];
island_test_18islands_1000LS = [76.27];

% 5 islands 500 iters
island_train_5islands = [75.13, 75.51, 74.93, 75.28, 74.93]; % TODO: Rerun experiment 5, some iterations missing.
island_test_5islands = [76.28, 76.20, 76.24, 76.23, 75.98];
island_train_median_5islands = [];
island_test_median_5islands = [71.90, 75.17, 74.69, 73.14, 73.39];

bo_train = [75.24, 75.41, 75.44, 75.09, 74.96];
bo_test = [76.28, 76.19, 76.34, 76.28, 76.30];
bo_train_median = [];
bo_test_median = [74.80, 74.81, 74.81, 74.70, 74.67];

rs_train = [75.03, 74.98, 75.02, 75.07, 75.19];
rs_test = [76.15, 76.23, 76.17, 76.23, 76.12];
rs_train_median = [];
rs_test_median = [73.96, 73.65, 74.22, 74.01, 74.09];

% 18 islands 350 iters + 150 local search
island_train_18islands_LS = [75.13, 75.04, 75.01, 75.38, 75.18];
island_test_18islands_LS = [76.20, 76.10, 76.33, 76.20, 76.12]; 

clc;

disp("Ozone air-quality model search" + char(10) + char(10) + "MISO (BETN073) ozone station." + char(10) + "Training: 2000-2001, testing: 2002, 6 weather variables. " + char(10) + "3-fold time-series cross-validation." + char(10) + "Max neurons per layer: 128." + char(10));

disp("Best possible Test: " + 76.34 + "% (BO). Best possible Train: " + 75.51 + "% (islands)." + char(10));

disp("Experiments" + char(10));

disp("Bayesian Optimization (BO):" + char(10) + ...
    "MAX Test: " + max(bo_test) + "% MAX Train: " + max(bo_train) + "%" + char(10) + ...
    "Test " + round(mean(bo_test), 2) + "% +/- " + round(std(bo_test), 2) + ...
    "%. Train " + round(mean(bo_train), 2) + "% +/- " + round(std(bo_train), 2) + ...
    "%. (Experiment samples: " + length(bo_test) + ")." + char(10));

disp("MAX Global search (18 islands: 5000 iterations):" + char(10) + ...
    "MAX Test: " + max(island_test_18islands_4000) + "% MAX Train: " + max(island_train_18islands_4000) + "%. " + ...
    "(Experiment samples: " + length(island_test_18islands_4000) + ")." + char(10));

disp("MAX Local search (18 islands: 400 iterations + local search: 1000 iterations):" + char(10) + ...
    "MAX Test: " + max(island_test_18islands_1000LS) + "% MAX Train: " + max(island_train_18islands_1000LS) + "%. " + ...
    "(Experiment samples: " + length(island_test_18islands_1000LS) + ")." + char(10));

disp("Island transpeciation (5 islands: 500 iterations):" + char(10) + ...
    "MAX Test: " + max(island_test_5islands) + "% MAX Train: " + max(island_train_5islands) + "%" + char(10) + ...
    "Test " + round(mean(island_test_5islands), 2) + "% +/- " + round(std(island_test_5islands), 2) + ...
    "%. Train " + round(mean(island_train_5islands), 2) + "% +/- " + round(std(island_train_5islands), 2) + ...
    "%. (Experiment samples: " + length(island_test_5islands) + ")." + char(10));

disp("Island transpeciation (18 islands: 350 iterations + local search: 150 iterations):" + char(10) + ...
    "MAX Test: " + max(island_test_18islands_LS) + "% MAX Train: " + max(island_train_18islands_LS) + "%" + char(10) + ...
    "TODO: Test " + round(mean(island_test_18islands_LS), 2) + "% +/- " + round(std(island_test_18islands_LS), 2) + ...
    "%. Train " + round(mean(island_train_18islands_LS), 2) + "% +/- " + round(std(island_train_18islands_LS), 2) + ...
    "%. (Experiment samples: " + length(island_test_18islands_LS) + ")." + char(10));

disp("Random Search (RS):" + char(10) + ...
    "MAX Test: " + max(rs_test) + "% MAX Train: " + max(rs_train) + "%" + char(10) + ...
    "Test " + round(mean(rs_test), 2) + "% +/- " + round(std(rs_test), 2) + ...
    "%. Train " + round(mean(rs_train), 2) + "% +/- " + round(std(rs_train), 2) + ...
    "%. (Experiment samples: " + length(rs_test) + ")." + char(10));