function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%%%%%%%%%%%%%% PART ONE FEED FORWARD %%%%%%%%%%%%%%%

a1 = [ones(m,1) X];                                         %5000x401

z2 = a1*Theta1';                                            %5000x401 * 401x25
a2 = sigmoid(z2);                                           %5000x25
a2 = [ones(m,1) a2];                                        %5000x26

z3 = a2*Theta2';                                            %5000x26 * 26x10
a3 = sigmoid(z3);                                           %5000x10
h = a3;



for i = 1:size(X,1)
y2 = zeros(size(Theta2,1),1);
y2(y(i)) = 1;
J += 1/m * (-y2'*(log(h(i,:)')) - (1-y2')*(log(1-h(i,:)' ) ) ) ;   % 10x1 * 1x10 5000x10 1x1
end

% use the 1x1 multiplication



%  ONE sum the j over the number of units in a3...
%  ONE recode the y vector into vectors should rearrange the equations with y in them. should change everthing.
%  TWO with the y changed, you need to sum over all the examples
%  TWO keep in mind the examples use one row at a time and the theta uses one row for each unit in the hidden layers
%  THREE put each row of the y vector in the form of a 10 by 1 vector,  
%  then use the 10 to multiply to the ten in the a3, 
%  then consider it the row of the 5000 and combine from the earlier row.
%  each row you find another 10 by 1 vector and another 10 x 5000 vector that you only use one row.
%  FOUR the first sum is performed by the vector multiplication,  the second is the loop += J...
grad = [0];
end
% audio engineer