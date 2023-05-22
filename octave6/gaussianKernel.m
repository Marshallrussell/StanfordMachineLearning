function sim = gaussianKernel(x1, x2, sigma)
%RBFKERNEL returns a radial basis function kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

% You need to return the following variables correctly.
sim = 0;
sum = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the similarity between x1
%               and x2 computed using a Gaussian kernel with bandwidth
%               sigma
%
%

for i=1:length(x1)
sum +=  ( x1(i)-x2(i) ).^2 ;
end

sum = sqrt(sum);
sim = exp(-sum.^2/(2*sigma.^2));


%  CURRENT
% the first way was the normalized distance squared 
% i.e. the square all each summe and taken square root and then squared...

%  NEXT
% the possibly way is that the distance is squared, 
% then each difference squared is added to together...  
% but not sure if it should be square rooted and also not sure if it gets squared outside the normalization...






% =============================================================
    
end
