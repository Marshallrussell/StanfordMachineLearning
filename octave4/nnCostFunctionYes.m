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

% How is the theta 2 dimension is 10 finally but the y dimension is 1.
% calculate the new exercise and the pdf
% move to the next week
% XX the ones before the sigmoid
% the y has only 1 dimension
% is z = sum over theta and a units
% then is z 1x1
% not summing   got to sum...  ?   for loop for the trials...



%%%%%%%%%%%%%% PART ONE FEED FORWARD %%%%%%%%%%%%%%%

a_1 = [ones(m, 1) X];                   %5000x401
% Add ones to the X data matrix
z_2 = a_1*Theta1';                      %5000x401 *  401x25
a_2 = sigmoid(z_2);                     %5000x25
a_2 = [ones(m, 1) a_2];                 %5000x26
% Add ones to the X data matrix
z_3 = a_2*Theta2' ;                     %5000x26  * 26x10  
a_3 = sigmoid(z_3);                     %5000x10

J = 1/m * sum((-y)'*(log(a_3)) - (1-y)'*(log(1-a_3))) ;  % 1x10  1x1

%%%%%%%%%%%%%%%% PART TWO BACKPROPAGATION %%%%%%%%%%%%

delta_3 = a_3 - y;         %5000X10 - 5000X10           THE Y IS ONLY 5000X1
size(delta_3)
size(y)
Theta2_grad = 1/m * ( delta_3' * a_2);              %10x5000 * 5000X26
delta_2 = delta_3 * Theta2 .* a_2 .* (1 - a_2);     %5000X10 * 10X26 - 5000X26
delta_2 = delta_2(:,2:size(delta_2,2));             %25X5000
Theta1_grad = 1/m * ( delta_2' * a_1);              %25x5000 * 5000X401

%%%%%%%%%%%%%%%% PART THREE GRAD %%%%%%%%%%%%%%%%%%%%%%

capdelta1 = delta_2'*a_1 ;  %25x5000   5000x401
capdelta2 = delta_3'*a_2 ;  %10x5000  5000x26

capD1 = capdelta1 + lambda * Theta1_grad ;
capD1(:,1) = capdelta1(:,1) ;

capD2 = capdelta2 + lambda * Theta2_grad ;
capD2(:,1) = capdelta2(:,1) ;



%%%%%%%%%%%%%%%% PART THREE COST %%%%%%%%%%%%%%%%%%%%%%
regularization = 0;

for l=1:2

if l==1,
theta_temp = Theta1;
else if l==2,
theta_temp = Theta2;
end

regularization_temp = 0;

    for i=1:size(theta_temp,1)
    for j=2:size(theta_temp,2)
        regularization_temp += ((lambda/(2*m)) * sum(sum(theta_temp(i , j) .^ 2))) ;
    end
    end

regularization += regularization_temp ;

end


% =========================================================================
% Update J for regularization

J = J + regularization ;

% Unroll gradients

grad = [Theta1_grad(:) ; Theta2_grad(:)];
D = [capD1(:) ; capD2(:)]

% Update grad for regularization

grad = grad + D ;



end
