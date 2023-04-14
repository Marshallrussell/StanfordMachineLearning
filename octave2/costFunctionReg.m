function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = X * theta ;

g = 1./(1+exp(-h));

J = 1/m * ((-y)'*(log(g)) - (1-y)'*(log(1-g))) + (lambda/(2*m)) * sum(theta(2:length(theta)) .^ 2) ;


% the first term should be grad(1)= -(1/m)*X(1)'*(g(1)-y(1));

% theta(2:length(theta)) = theta(2:length(theta)) * (1-lambda/m);
% theta = theta .* (ones(size(theta))-eye(size(theta))) + (1/m) * (X') * (g-y);

% grad = theta(2:size(theta)) - 1/m * X ' * (sigmoid(X*theta(2:size(theta))) - y) - lambda / m * theta(2:end) ;

grad = - 1/m * X' * (g -y);
grad = 1/m * X' * (sigmoid(X*theta)-y) + lambda / m * theta .* (ones(size(theta))-eye(size(theta))) ;
% =============================================================

end
