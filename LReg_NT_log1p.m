% ========================================================================================================================
% Author: Chieh-En Tsai (Andy Tsai)
% Date:   2016/01/09
%
% LREG_NT_LOG1P:
%   Training regularized logistic regression model with Newton method + back tracking line search
%
% Input:
%   - w0        (nx1):       initial model
%   - y         (lx1):       ground truth label
%   - x         (lxn):       training samples
%   - C         (1x) :       likelihood-regularizer trade off coefficient
%   - max_iter  (1x) :       maximum iteration
%   - yeta      (1x) :       backtracking line search parameter, control the degree of descent in objective each iteration
%   - epslon    (1x) :       relative stopping criterion
%   - caseed    (1x) :       conjugate gradient parameter
%   - criterion (1/2):       case 1: |g| <= eps*|g0|
%                            case 2: |g| <= eps*min(#pos, #neg)/(#pos+#neg)*|g0|, in order to balance the pos and neg samples
%
% Output:
%   - w         (nx1):       optimized model
%   - losses         :       the value of objective each iteration
%   - norms          :       |g| each iteration
% ========================================================================================================================

function [w, losses, norms] = LReg_NT_log1p(w0, y, x, C, max_iter, yeta, epslon, caseed, criterion)

losses = [];
norms  = [];

w = w0;

% convert from 0/1 label to -1/1 label
y(find(y<0.5)) = -1;

% uncomment if you wish to rearrange data
%x = [x(find(y>0), :); x(find(y<0), :)];
%y = [y(find(y>0)); y(find(y<0))];

cache_xw = x*w;

for iter = 1:max_iter
    hypothesis = exp(-1*y.*cache_xw);
    g = w + C*x'*((1./(1+hypothesis) -1).*y);
    gg = g'*g;

% =======================================================
% check stopping criterion
% =======================================================
    if iter == 1
        ss0 = gg;
        if criterion == 1
            stopping_criterion = epslon^2*gg;
        end
        if criterion == 2
            stopping_criterion = (epslon*(min(numel(find(y>0)), numel(find(y<0))))/numel(y))^2*gg;
        end
    end
    % check stoping criterion
    if gg <= stopping_criterion
        break;
    end
% ========================================================
% decide Newton direction before backtracing line search
% ========================================================
    D = hypothesis ./ (1+hypothesis).^2;
    s = zeros(numel(g), 1);
    r = -g;
    d = r;
    inner_stopping_criterion = caseed^2 * gg;
    norm_r = r'*r;

% start the inner loop
    inner_loop = 0;
    while true
        if norm_r <= inner_stopping_criterion
            break;
        end

        Dxd     = d + C*x'*(D.*(x*d)); % turns diagonal MM into elementwise multiply
        a       = norm_r / (d'*Dxd);
        s       = s + a*d;
        r_next  = r - a*Dxd;
        norm_r_next = r_next'*r_next;
        d       = r_next + (norm_r_next / norm_r)*d;
        r       = r_next;
        norm_r  = norm_r_next;

        inner_loop = inner_loop+1;
    end
    
% ========================================================
% backtracing line search
% ========================================================

    alpha = 1;
    cache_xs = x*s;
    ws = w'*s;
    ss = s'*s;

%    neg_like = sum(log(1+hypothesis));
    neg_like = sum(log1p(hypothesis));
    
    loss = 0.5*w'*w + C*neg_like;
    fprintf('iter%3d f %.3e |g| %.3e CG %3d ', iter, loss, gg^0.5, inner_loop);
    losses = [losses, loss];
    norms  = [norms, gg];

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

neg_like = sum(log(1+hypothesis));
loss = 0.5*w'*w + C*neg_like;
fprintf('=============== termination info ===================\n');

fprintf('Iter%3d f %.3e |g| %.3e CG %3d step_size %.3e\n', iter, loss, gg^0.5, inner_loop, alpha);

fprintf('====================================================\n');
%fprintf('check alpha = %g, should be equal to 1\n', alpha);

if iter == max_iter
    fprintf('Max iter reached !\n')
end


end


