function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

g = sigmoid(X*theta);



J = 1/m * ((-y)'*(log(g)) - (1-y)'*(log(1-g))) + (lambda/(2*m)) * sum(theta(2:length(theta)) .^ 2) ;
grad = 1/m * X' * (sigmoid(X*theta)-y) + lambda / m * theta .* (ones(size(theta))-eye(size(theta))) ;

% =============================================================

grad = grad(:);

end
