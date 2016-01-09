% ==============================================================================================
% Author: Chieh-En Tsai (Andy Tsai)
% Date:   2016/01/09
%
% RUN_LOGREG.m
% DESCRIPTION:
%    quick demo interface for regularized logistic regression model:
%                                   1/2*|w|^2 + C*sum_through_each_sample(log(1+exp(-yi*wT*xi)))
%   
%    and evaluate on testing set with decision function:
%                                   p(1|x) = 1/(1+exp(-y*wT*x))
%   
%    y would be 0/1 label
%
% USAGE:
%    load training data into inst (#training samples x #feature)
%    and training label into label(#training samples x 1)
%
%    load testing data into tinst (#testing samples  x #feature)
%    and testing label into tlabel(#testing samples  x 1)
% ==============================================================================================

w0 = zeros(size(inst, 2), 1);
max_iter = 1000
C = 0.1
epslon = 0.01
yeta = 0.01
caseed = 0.1
criterion = 2

tic()
%[w, losses, norms] = LReg_GD(w0, label, inst, C, max_iter, yeta, epslon);
[w, losses, norms] = LReg_NT_log1p(w0, label, inst, C, max_iter, yeta, epslon, caseed, criterion);
toc()

% evaluate training accuracy
fprintf('\n\n\ntraining accuracy: %g\n', evaluate(w, label, inst));

% evaluate testing accuracy
fprintf('\ntesting accuracy %g\n\n\n', evaluate(w, tlabel, tinst));
