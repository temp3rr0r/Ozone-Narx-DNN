%% Distribution function tool
%disttool
close all;
%% FFMI
% PDF
x = 10:.1:100;
y = normpdf(x,19.7, 3.3370);
figure
plot(x, y);
xlabel('FFMI');
ylabel('PDF');
title('FFMI PDF');
% CDF
cd = cumsum(y) ./10;
figure
plot(x,cd)
xlabel('FFMI');
ylabel('CDF');
title('FFMI CDF');
%% FMI
% PDF
x = 0:.1:50;
y = normpdf(x,5, 3.3085);
figure
plot(x, y);
xlabel('FMI');
ylabel('CDF');
title('FMI PDF');
% CDF
cd = cumsum(y)./10;
figure
plot(x,cd)
xlabel('FMI');
ylabel('CDF');
title('FMI CDF');
%% Bivariate Normal Distribution PDF/CDF
% https://nl.mathworks.com/help/stats/multivariate-normal-distribution.html
mu = [19.7 5];
covariance_FFMI_FMI = 3.3;
variance_FFMI = 3.3370; % 2.5
variance_FMI = 3.3; % 1.4
sigma = [variance_FFMI covariance_FFMI_FMI; covariance_FFMI_FMI variance_FMI];
% Create a grid of evenly spaced points in two-dimensional space.
x1 = 10:0.2:100;
x2 = 0:0.2:50;
[X1,X2] = meshgrid(x1,x2);
X = [X1(:) X2(:)];
% Evaluate the pdf of the normal distribution at the grid points.
y = mvnpdf(X,mu,sigma);
y = reshape(y,length(x2),length(x1));

% Plot the pdf values.
figure
surf(x1,x2,y)
caxis([min(y(:))-0.5*range(y(:)),max(y(:))])
z_max = max(max(y)) * 1.01;

axis_x_min = 10;
axis_x_max = 30;
axis_y_min = 1;
axis_y_max = 10;
axis([axis_x_min axis_x_max axis_y_min axis_y_max 0 z_max]);
xlabel('FFMI')
ylabel('FMI')
zlabel('Probability Density')
colorbar;
%% CDF
grid_point_side_count = 75;
[X1,X2] = meshgrid(linspace(axis_x_min, axis_x_max, grid_point_side_count)',linspace(axis_y_min, axis_y_max, grid_point_side_count)');
X = [X1(:) X2(:)];
% Evaluate the cdf of the normal distribution at the grid points.
p = mvncdf(X,mu,sigma);
% Plot the cdf values.
Z = reshape(p,grid_point_side_count, grid_point_side_count);
figure
surf(X1,X2,Z)
xlabel('FFMI')
ylabel('FMI')
zlabel('Cumulative Density')
colorbar;

%%
% Probability over Rectangular Region
current_FFMI = 23.55706122;
current_FMI = 9.389877551;
error = 0.005;

current_FFMI_min = current_FFMI * (1 - error);
current_FMI_min = current_FMI * (1 - error);
current_FFMI_max = current_FFMI * (1 + error);
current_FMI_max = current_FMI * (1 + error);

[p,err] = mvncdf([current_FFMI_min current_FMI_min],...
    [current_FFMI_max current_FMI_max], mu, sigma);

disp("Percentile (FFMI " + round(current_FFMI_min, 2) + "-" + ...
    round(current_FFMI_max, 2) + ", FMI " + round(current_FMI_min, 2)...
    + "-" + round(current_FMI_max, 2) +", varying: +/- " + round(error * 100, 2)+ "%): " + round((1-p) * 100, 4) + " (CDF error: " + err + ")");
%% Contour PDF
figure
contourf(x1,x2,y,[0.0001 0.001 0.01 0.05 0.15 0.25 0.35]);
axis([axis_x_min axis_x_max axis_y_min axis_y_max 0 z_max]);
line([current_FFMI_min current_FFMI_max current_FFMI_max current_FFMI_min current_FFMI_min],[current_FMI_min current_FMI_min current_FMI_max current_FMI_max current_FMI_min],'Linestyle','--','Color','k')
xlabel('FFMI')
ylabel('FMI')
title('Contour PDF')
colorbar;

%% TODO: LS fit mu, covariances, std

X_expected = [16.2 2.4; 16.4 2.5; 18.7 3.8; 19.7 5; 20.6 6; 24.1 7.9; 24.3 8.7];
y_expected = [0.05; 0.1; 0.25; 0.5; 0.75; 0.9; 0.95];

mu = [19.7 5];
covariance_FFMI_FMI = 1.0;
variance_FFMI = 2.5; % 2.5
variance_FMI = 1.4; % 1.4
sigma = [variance_FFMI covariance_FFMI_FMI; covariance_FFMI_FMI variance_FMI];

expected_ps = zeros(1, length(y_expected));
ps = zeros(1, length(y_expected));

for i = 1:length(y_expected)
    current_FFMI = X_expected(i, 1);
    current_FMI = X_expected(i, 2)
    expected_p = y_expected(i);
    [p,err] = mvncdf([current_FFMI current_FMI], mu, sigma);        
    expected_ps(i) = expected_p;
    ps(i) = p;
end

%%
% x0 = [-.5; 0];
% options = optimoptions('fminunc','Algorithm','quasi-newton');
% [x, fval] = fminunc(f,x0,options)
clc;

fun1([ 3.3370    3.3085    3.3])

% lb = [1.0, 1.1, 0.01];
% ub = [3.5, 2.9, 1];

lb = [3.3370, 3.3085, 0.01];
ub = [3.3370, 3.3085, 3.3];

nvars = length(lb);
options = optimoptions('particleswarm','SwarmSize',100,'FunctionTolerance',10e-9, 'MaxStallIterations', 20);

[x,fval,exitflag] = particleswarm(@fun1,nvars,lb,ub, options)
%%
fun1([    1.1000    1.1000    1.0000])
 
 %% FFMI
% PDF
x = 10:.1:100;
y = normpdf(x,19.7,2.5);
figure
plot(x, y);
xlabel('FFMI');
ylabel('PDF');
title('FFMI PDF');
% CDF
cd = cumsum(y) ./10;
figure
plot(x,cd)
xlabel('FFMI');
ylabel('CDF');
title('FFMI CDF');
%% 1D FFMI
clc;

fun_FFMI([3.3370    0         0])

lb = [1.0, 0.0, 0.0];
ub = [5.5, 0.0, 0.0];
nvars = length(lb);
options = optimoptions('particleswarm','SwarmSize',100,'FunctionTolerance',10e-6, 'MaxStallIterations', 20);

[x,fval,exitflag] = particleswarm(@fun_FFMI,nvars,lb,ub, options)%%
%% 1D FMI
clc;

fun_FMI([0.0   3.3085     0])

lb = [0.0, 1.1, 0.0];
ub = [0.0, 5, 0.0];
nvars = length(lb);
options = optimoptions('particleswarm','SwarmSize',100,'FunctionTolerance',10e-6, 'MaxStallIterations', 20);

[x,fval,exitflag] = particleswarm(@fun_FMI,nvars,lb,ub, options)
 %%
 function out1 = fun_FMI(x_1)        
    mu = [19.7 5];    
    X_expected = [16.2 2.4; 16.4 2.5; 18.7 3.8; 19.7 5; 20.6 6; 24.1 7.9; 24.3 8.7];
    y_expected = [0.05; 0.1; 0.25; 0.5; 0.75; 0.9; 0.95];
    
    variance_FFMI = x_1(1);
    variance_FMI = x_1(2);
    covariance_FFMI_FMI = x_1(3);
    sigma = [variance_FFMI covariance_FFMI_FMI; covariance_FFMI_FMI variance_FMI];    
    
    expected_ps = zeros(1, length(y_expected));
    ps = zeros(1, length(y_expected));
    for i = 1:length(y_expected)
        current_FFMI = X_expected(i, 1);
        current_FMI = X_expected(i, 2);
        expected_p = y_expected(i);
        %[p, err] = mvncdf([current_FFMI current_FMI], mu, sigma);     
        [p, err] = mvncdf(current_FMI, mu(2), variance_FMI);
        expected_ps(i) = expected_p;
        ps(i) = p;
    end        
    out1 = mse(ps, expected_ps);    
end
 
 function out1 = fun_FFMI(x_1)        
    mu = [19.7 5];    
    X_expected = [16.2 2.4; 16.4 2.5; 18.7 3.8; 19.7 5; 20.6 6; 24.1 7.9; 24.3 8.7];
    y_expected = [0.05; 0.1; 0.25; 0.5; 0.75; 0.9; 0.95];
    
    variance_FFMI = x_1(1);
    variance_FMI = x_1(2);
    covariance_FFMI_FMI = x_1(3);
    sigma = [variance_FFMI covariance_FFMI_FMI; covariance_FFMI_FMI variance_FMI];    
    
    expected_ps = zeros(1, length(y_expected));
    ps = zeros(1, length(y_expected));
    for i = 1:length(y_expected)
        current_FFMI = X_expected(i, 1);
        current_FMI = X_expected(i, 2);
        expected_p = y_expected(i);
        %[p, err] = mvncdf([current_FFMI current_FMI], mu, sigma);     
        [p, err] = mvncdf(current_FFMI, mu(1), variance_FFMI);
        expected_ps(i) = expected_p;
        ps(i) = p;
    end        
    out1 = mse(ps, expected_ps);    
end
 
function out1 = fun1(x_1)        
    mu = [19.7 5];    
    X_expected = [16.2 2.4; 16.4 2.5; 18.7 3.8; 19.7 5; 20.6 6; 24.1 7.9; 24.3 8.7];
    y_expected = [0.05; 0.1; 0.25; 0.5; 0.75; 0.9; 0.95];
    
    variance_FFMI = x_1(1);
    variance_FMI = x_1(2);
    covariance_FFMI_FMI = x_1(3);
    sigma = [variance_FFMI covariance_FFMI_FMI; covariance_FFMI_FMI variance_FMI];    
    
    expected_ps = zeros(1, length(y_expected));
    ps = zeros(1, length(y_expected));
    for i = 1:length(y_expected)
        current_FFMI = X_expected(i, 1);
        current_FMI = X_expected(i, 2);
        expected_p = y_expected(i);
        [p, err] = mvncdf([current_FFMI current_FMI], mu, sigma);        
        expected_ps(i) = expected_p;
        ps(i) = p;
    end
        
    out1 = mse(ps, expected_ps);    
end