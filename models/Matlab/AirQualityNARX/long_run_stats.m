%%
island_train = [75.13, 75.51, 74.93, 75.28, 74.93];
island_test = [76.28, 76.20, 76.24, 76.23, 75.98];

bo_train = [75.24, 75.41, 75.44, 75.09, 74.96];
bo_test = [76.28, 76.19, 76.34, 76.28, 76.30];

rs_train = [75.03, 74.98, 75.04, 75.04];
rs_test = [76.15, 76.23, 76.17, 76.23];

% TODO: island + LS

clc;
disp("Island (5 islands)" + char(10) + "Train: " + round(mean(island_train), 2) + " +/- " + round(std(island_train), 2) + " Test: " + round(mean(island_test), 2) + " +/- " + round(std(island_test), 2));
disp("BO" + char(10) + "Train: " + round(mean(bo_train), 2) + " +/- " + round(std(bo_train), 2) + " Test: " + round(mean(bo_test), 2) + " +/- " + round(std(bo_test), 2));
disp("RS" + char(10) + "Train: " + round(mean(rs_train), 2) + " +/- " + round(std(rs_train), 2) + " Test: " + round(mean(rs_test), 2) + " +/- " + round(std(rs_test), 2));