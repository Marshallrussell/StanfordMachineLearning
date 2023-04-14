function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% Compute the regression vectors and matrices, then use g = sigmoid(z)
z = X*all_theta' ;
t = max(z,[],2);


% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%
%  
% return the index which is the class i - this would return the max for each matrix 

%   Steps:  start with z, go to h and g, go to grad and j, then a new y
%   Steps: next you regularize, and regularize, and parametize

%   Steps: Matrix X is inverted, then Matrix X isn't inverted
%   Steps: next run to use g=sigmoid(z) and 0.5 instead of using z and 0
%   Steps: next run val = zeros(size(X,1), 1) and [val(i), p(i)] = max(z(i,:),[],2);
%   Steps: next run p = max(z,[],2) ;  //this return the index if switched the formula

for i=1:m;
for j=1:num_labels;
    if (t(i) == z(i,j)),
    p(i) = j;   
    end
end
end

% =========================================================================
%   THis works with max(z,[],1)  and max(g,[],1)
%   Older Algorithms and Details
%       
%       values should be zero or one (to satisfy the J equation with y and h)
%       p should be the class - i.e. 1 through num_labels
%       p(i) = num_labels;

end
