function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
% This is the sigmoid = 1/(1+e^-(theta_transpose*x))

d = 1 + exp(-z) ;
g = 1./ d; 



% =============================================================

end
% SUCCESS IN OCTAVE WITH THE CODE IT WORKS