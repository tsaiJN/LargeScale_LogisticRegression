this is the programming homework of convex optimization and machine learning course

I implemented
    - Gradient-descent method with line search
    - Newton method with line search and Conjugate Gradient

to run these codes on kdd 2010 (bridge to algebra) dataset, you may wish to load the dataset first by libsvmread:
matlab:
> [label, inst] = libsvmread('path/to/your/training/data');
> [tlabel, tinst] = libsvmread('path/to/your/testing/data');
> run_LogReg

run_LogReg is a quick demo interface for regularized logistic regression model:
                       
                        1/2*|w|^2 + C*sum_through_each_sample(log(1+exp(-yi*wT*xi)))

and it evaluate on testing set with decision function:
                               
                        p(1|x) = 1/(1+exp(-y*wT*x))

y would be 0/1 label


more detailed description can be found in the comments in codes
