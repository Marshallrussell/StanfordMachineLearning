function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);
val = zeros(size(X,1),1);

% Add ones to the X data matrix
X = [ones(size(X,1), 1) X];

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
%  the a1 is the inputs,  the hidden layer is a2 , the output is a3 .    16 runs and 4 classes...  2,3,5 parameters

Z = X * Theta1';        %  // Four Classes... // 16x3 3x4   16x4   
G = sigmoid(Z) ;  

% Add ones to the G data matrix
G = [ones(size(G,1),1)  G] ;        %  //  16x5

L = G * Theta2' ;    %  Theta2 is a row...     trial x label  ...   // 16x5  4x5   16x5 5x4 - 16x4
H = sigmoid(L) ;

[val,p] = max(H,[],2);



% =========================================================================


end
