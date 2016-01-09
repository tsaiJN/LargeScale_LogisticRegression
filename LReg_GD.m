% ========================================================================================================================
% Author: Chieh-En Tsai (Andy Tsai)
% Date:   2016/01/09
%
% LREG_GD:
%   Training regularized logistic regression model with gradient descent + back tracking line search
%
% Input:
%   - w0        (nx1):       initial model
%   - y         (lx1):       ground truth label
%   - x         (lxn):       training samples
%   - C         (1x) :       likelihood-regularizer trade off coefficient
%   - max_iter  (1x) :       maximum iteration
%   - yeta      (1x) :       backtracking line search parameter, control the degree of descent in objective each iteration
%   - epslon    (1x) :       relative stopping criterion |g| <= eps*|g0|
%
% Output:
%   - w         (nx1):       optimized model
%   - losses         :       the value of objective each iteration
%   - norms          :       |g| each iteration
% ========================================================================================================================

function [w, losses, norms] = LReg_GD(w0, y, x, C, max_iter, yeta, epslon)

losses = [];
norms  = [];

w = w0;

% convert from 0/1 label to -1/1 label
y = (y-0.5)*2;

cache_xw = x*w;
for iter = 1:max_iter
    hypothesis = exp(-1*y.*cache_xw);

% compute the gradient and direction
    g = w + C*x'*((1./(1+hypothesis) -1).*y);
    s = -1*g;

    alpha = 1;
    cache_xs = x*s;
    ws = w'*s;
    ss = s'*s;
    if iter == 1
        ss0 = ss;
    end

% check stoping criterion
    if ss <= epslon^2*ss0
        break;
    end

    neg_like = sum(log(1+hypothesis));
    
    loss = 0.5*w'*w + C*neg_like;
    fprintf('iter%3d f %.3e |g| %.3e CG   0 ', iter, loss, ss^0.5);
    losses = [losses, loss];
    norms  = [norms, ss];

    while true
        new_xw = cache_xw + alpha*cache_xs;
        new_hypo = exp(-1*y.*(new_xw));
        if alpha*ws+(0.5*alpha^2+yeta*alpha)*ss <= C*( neg_like - sum(log(1+new_hypo)) )
            break;
        end
        alpha = alpha / 2;
    end

    fprintf('step_size %.3e\n', alpha);

% update model
    cache_xw =  new_xw;
    w = w+alpha*s;

end

if iter == max_iter
    fprintf('Max iter reached !\n')
end

fprintf('=============== termination info ===================\n');

fprintf('Iter%3d f %.3e |g| %.3e CG %3d step_size %.3e\n', iter, loss, ss^0.5, alpha);

fprintf('====================================================\n');

end


