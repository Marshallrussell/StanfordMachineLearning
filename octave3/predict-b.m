function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

size(X,1)
size(X,2)
% Add ones to the X data matrix
X = [ones(size(X,1), 1) X];

size(X,1)
size(X,2)

size(Theta1,1)
size(Theta1,2)
% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
%  the a1 is the inputs,  the hidden layer is a2 , the output is a3 .  

Z = X * Theta1';        %  2x16  // 16x3 3x4   16 x 4
GZ = sigmoid(Z) ;

% Add ones to the Z data matrix
% Z = [ones(size(Z,1),1)  Z] ;        % //16 x 5

Z3 = GZ * Theta2 ;    % theta2 is a row    1x a3#   4x5 5x16    4x16  labels x trials     4x5 4x17 -- 5x17   label x trial  ... // 16x5  4x5   16x5 5x4 - 16x4

GZ3 = sigmoid(Z3) ;

d = max(GZ3, [], 2) ;
% trial x class

for i=1:size(GZ3,1);
for j=1:size(GZ3,2);
    if (GZ3(i,j)== d(i)),
    p(i) = j;
    end
end
end






% =========================================================================


end
