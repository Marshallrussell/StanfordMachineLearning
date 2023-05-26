function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%
%

values = [0.01 ; 0.03 ; .1 ; .3 ; 1 ; 3 ; 10 ; 30];
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

% Swith the vectors in GaussianKernel into for loops variables or NORMATIVE variables
% Switch the num vector to another name...  maybe its reacting because the name num
% Set up the min min and err matrix...  it made no success when i went into temp-var-total...
%
% Leave out the model parameters tol and numpasses: 1e-3, 5
% The X: 211x1... After loop, X: 422x1.  The y: 211x1  && To show use first line size(X)   [PROBLEMS]

% Train the SVM
% model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
total = 1000000;
    
for indexC=1:size(values,1)
for indexS=1:size(values,1)   
    model = svmTrain(X, y, values(indexC), @(x1, x2) gaussianKernel(x1 , x2 , values(indexS)) );
    pred = svmPredict(model,Xval);
    err = mean(double(pred ~= yval));
    if total >= err,
    sigma = values(indexS);  C = values(indexC); total = err;
end
end
end


% =========================================================================

end

%    ERROR HOLDS ONE VALUE INSTEAD OF MATRIXED
%    total = 1000000;
%    err = mean(double(pred ~= yval));
%    if total >= err,
%    sigma = num(indexS);  C = num(indexC); total = err;

%    USES NO TEMPORARY VALUES WITH MIN FUNCTION    
%    err(indexC,indexS) = mean(double(pred ~= yval));
%    if (err(indexC,indexS) <= min(min(err)))
%    sigma = values(indexS);   C = values(indexC) ;
%    end