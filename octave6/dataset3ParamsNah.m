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

numbers = [0.01 , 0.03 , .1 , .3 , 1 , 3 , 10 , 30]
errors = zeros(length(numbers),length(numbers));

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


for i=1:length(numbers)
for j=1:length(numbers)
C = i; sigma = j; x1=X; x2=y;
sim = gaussianKernel(x1, x2, sigma) ;
model = svmTrain(X, y, C, sim, 1e-3, 5) ;
pred = svmPredict(model,Xval) ;
errors[i,j] = mean(double(pred ~= yval)) ;
end
end

min = errors(1,1);
for i=1:size(errors,1)
for j=1:size(errors,2)
if errors(i,j) < min,
min = errors(i,j);
index_sigma = j; index_C = i;
end
end
end

C = numbers(index_C);
sigma = numbers(index_sigma);




% =========================================================================

end
