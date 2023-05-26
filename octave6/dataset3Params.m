function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%
%

num= [0.01 ; 0.03 ; .1 ; .3 ; 1 ; 3 ; 10 ; 30];
err = zeros(length(num));

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
%  edit out the model parameters tol and numpasses: 1e-3, 5
% The X: 211x1... After loop, X: 422x1.  The y: 211x1  && To show use first line size(X)


total = 1000000;

for indexC=1:size(err,1)
for indexS=1:size(err,1)   
    sim = gaussianKernel(X, y, num(indexS));
    model = svmTrain(X, y, num(indexC), sim);
    pred = svmPredict(model,Xval);
    err = mean(double(pred ~= yval));
    if total >= err,
    sigma = num(indexS);  C = num(indexC); total = err;
    end
end
end


% =========================================================================

end

% if err <= min(min(err));
%    sigma = num(indexS);  C = num(indexC) ;
   