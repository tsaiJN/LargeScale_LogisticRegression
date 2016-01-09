% =================================================
% Author: Chieh-En Tsai (Andy Tsai)
% Date:   2016/01/09
%
% EVALUATE:
%     evaluate a trained model
% 
% INPUT:
%     - w   (nx1):      trained model
%     - y   (lx1):      testing label
%     - x   (lxn):      testing samples
% 
% OUTPUT:
%     - accuracy :      testing accuracy
% =================================================
function accuracy = evaluate(w, y, x)
    w = w(1:min(numel(w), size(x, 2)));
    y = (y-0.5)*2;
    ans = zeros(numel(y), 1)-1;

    p = -1*(log(1+exp(-(x*w))));
    ans(find(p >= log(0.5))) = 1;

    
    accuracy = 1 - sum(abs(ans - y)) / (2*numel(y));
    
end
