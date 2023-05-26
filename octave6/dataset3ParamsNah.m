function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%
%

% I HAD TO ADD THIS - PROBABLY THERE IS A DIFFERENT METHOD  ARBITRARY C VALUES

num= [0.01 , 0.03 , .1 , .3 , 1 , 3 , 10 , 30]
err = zeros(length(values));

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



for indexC=1:size(err,1)
for indexS=1:size(err,1)
    sim = gaussianKernel(X, y, num(indexS));
    model = svmTrain(X, y, num(indexC), sim, 1e-3, 5);
    pred = svmPredict(model,Xval);
    err(indexC,indexS) = mean(double(pred ~= yval));
    if err(indexC,indexS) <= min(min(err));
    sigma = num(indexS);  C = num(indexC) ;
    end
end
end


% =========================================================================

end
