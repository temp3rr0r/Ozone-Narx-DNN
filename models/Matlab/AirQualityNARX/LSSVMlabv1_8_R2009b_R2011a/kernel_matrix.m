function omega = kernel_matrix(Xtrain,kernel_type, kernel_pars,Xt)
% Construct the positive (semi-) definite and symmetric kernel matrix
%
% >> Omega = kernel_matrix(X, kernel_fct, sig2)
%
% This matrix should be positive definite if the kernel function
% satisfies the Mercer condition. Construct the kernel values for
% all test data points in the rows of Xt, relative to the points of X.
%
% >> Omega_Xt = kernel_matrix(X, kernel_fct, sig2, Xt)
%
%
% Full syntax
%
% >> Omega = kernel_matrix(X, kernel_fct, sig2)
% >> Omega = kernel_matrix(X, kernel_fct, sig2, Xt)
%
% Outputs
%   Omega  : N x N (N x Nt) kernel matrix
% Inputs
%   X      : N x d matrix with the inputs of the training data
%   kernel : Kernel type (by default 'RBF_kernel')
%   sig2   : Kernel parameter (bandwidth in the case of the 'RBF_kernel')
%   Xt(*)  : Nt x d matrix with the inputs of the test data
%
% See also:
%  RBF_kernel, lin_kernel, kpca, trainlssvm, kentropy


% Copyright (c) 2011,  KULeuven-ESAT-SCD, License & help @ http://www.esat.kuleuven.be/sista/lssvmlab

[nb_data,d] = size(Xtrain);


if strcmp(kernel_type,'RBF_kernel'),
    if nargin<4,
        XXh = sum(Xtrain.^2,2)*ones(1,nb_data);
        omega = XXh+XXh'-2*(Xtrain*Xtrain');
        omega = exp(-omega./(2*kernel_pars(1)));
    else
        XXh1 = sum(Xtrain.^2,2)*ones(1,size(Xt,1));
        XXh2 = sum(Xt.^2,2)*ones(1,nb_data);
        omega = XXh1+XXh2' - 2*Xtrain*Xt';
        omega = exp(-omega./(2*kernel_pars(1)));
    end
    
elseif strcmp(kernel_type,'RBF4_kernel'),
    if nargin<4,
        XXh = sum(Xtrain.^2,2)*ones(1,nb_data);
        omega = XXh+XXh'-2*(Xtrain*Xtrain');
        omega = 0.5*(3-omega./kernel_pars).*exp(-omega./(2*kernel_pars(1)));
    else
        XXh1 = sum(Xtrain.^2,2)*ones(1,size(Xt,1));
        XXh2 = sum(Xt.^2,2)*ones(1,nb_data);
        omega = XXh1+XXh2' - 2*Xtrain*Xt';
        omega = 0.5*(3-omega./kernel_pars).*exp(-omega./(2*kernel_pars(1)));
    end
    
% elseif strcmp(kernel_type,'sinc_kernel'),
%     if nargin<4,
%         omega = sum(Xtrain,2)*ones(1,size(Xtrain,1));
%         omega = omega - omega';
%         omega = sinc(omega./kernel_pars(1));
%     else
%         XXh1 = sum(Xtrain,2)*ones(1,size(Xt,1));
%         XXh2 = sum(Xt,2)*ones(1,nb_data);
%         omega = XXh1-XXh2';
%         omega = sinc(omega./kernel_pars(1));
%     end
    
elseif strcmp(kernel_type,'lin_kernel')
    if nargin<4,
        omega = Xtrain*Xtrain';
    else
        omega = Xtrain*Xt';
    end
    
elseif strcmp(kernel_type,'poly_kernel')
    if nargin<4,
        omega = (Xtrain*Xtrain'+kernel_pars(1)).^kernel_pars(2);
    else
        omega = (Xtrain*Xt'+kernel_pars(1)).^kernel_pars(2);
    end
    
% elseif strcmp(kernel_type,'wav_kernel')
%     if nargin<4,
%         XXh = sum(Xtrain.^2,2)*ones(1,nb_data);
%         omega = XXh+XXh'-2*(Xtrain*Xtrain');
%         
%         XXh1 = sum(Xtrain,2)*ones(1,nb_data);
%         omega1 = XXh1-XXh1';
%         omega = cos(kernel_pars(3)*omega1./kernel_pars(2)).*exp(-omega./kernel_pars(1));
%         
%     else
%         XXh1 = sum(Xtrain.^2,2)*ones(1,size(Xt,1));
%         XXh2 = sum(Xt.^2,2)*ones(1,nb_data);
%         omega = XXh1+XXh2' - 2*(Xtrain*Xt');
%         
%         XXh11 = sum(Xtrain,2)*ones(1,size(Xt,1));
%         XXh22 = sum(Xt,2)*ones(1,nb_data);
%         omega1 = XXh11-XXh22';
%         
%         omega = cos(kernel_pars(3)*omega1./kernel_pars(2)).*exp(-omega./kernel_pars(1));
%     end
end