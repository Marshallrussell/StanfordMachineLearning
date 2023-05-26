function sim = gaussianKernel(x1, x2, sigma)
%RBFKERNEL returns a radial basis function kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim

% Ensure that x1 and x2 are column vectors
x1 = x1(:);
x2 = x2(:);
n = length(x1);
 
% You need to return the following variables correctly.
sim = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the similarity between x1
%               and x2 computed using a Gaussian kernel with bandwidth
%               sigma
%
%

sim = exp(-sum((x1(1:n)-x2(1:n)).^2) / (2*sigma^2));


% =============================================================
    
end

%   VECTORIZE HERE
%   sim = exp(sum((x1(1:n)-x2(1:n)).^2) / (2*sigma^2));

%   SIZE COMPUTED WITH LOOPS AND INCREMENT
%   for j=1:n     
%   a += (x1(j)-x2(j))^2;   
%   end      
%   sim = exp(-a / (2*sigma^2));


%   I NEED TO PASS THE COLUMNS OF X AND Y
%   THE VECTORS X1 AND X2 ARE NOT COLUMNS IF YOU PASS IN X AND Y...