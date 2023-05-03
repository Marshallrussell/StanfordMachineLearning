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

%%%%%%%%%%%%%%%%%%%%%% PART ONE %%%%%%%%%%
% theta = j columns x i rows and sum over L
% add regularization to the variable J... in part three*.
% J = 1/m * sum(sum((-y)'*(log(g)) - (1-y)'*(log(1-g)))) ;
% Feed Forward the neural network and return the cost in the variable J
% STEP - build up z to find a_l

a_1 = X;
% Add ones to the X data matrix
a_1 = [ones(m, 1) a_1];
z_2 = Theta1 * a_1';        %25x401 * 5000x401 '
a_2 = sigmoid(z_2);               %25x5000     NEW JCOST
% Add ones to the X data matrix
a_2 = [ones(m, 1) a_2'];    %5000x26
z_3 = Theta2 * a_2' ;      %10x26 * 26x5000   
a_3 = sigmoid(z_3');               %10x5000  5000x10        NEW JCOST
%check for bias activation in layer 1 and 2.  DONE.
% NOTE **recode the extra bias in theta2

% this a_3 = h(x) = a1_3 = g(theta2 * a_2)

J = 1/m * sum(sum((-y)'*(log(a_3)) - (1-y)'*(log(1-a_3))))

%%%%%%%%%%%%%%%%%%%%%% PART TWO %%%%%%%%%%
% NOW THE GRADIENT DOESNT WORK YET...
% grad = 1/m * X' * (sigmoid(X*theta)-y) + lambda / m * theta .* (ones(size(theta))-eye(size(theta))) ;



%STEP - build up z to find a_l
delta_3 = a_3 - y;         %5000X10 - 5000X10
size(Theta2)
size(delta_3)
size(a_2)
size(a_3)
% part one and part two are working.   

% STEP - compute the g_prime : Theta2 ' * delta_3 .* g_prime(z_l)
delta_2 = delta_3 * Theta2 .* a_2 .* (1 - a_2);       %5000X10 * 10X26 (THETA2) 5000X26 .- 5000X26 .- 5000X26
%check for regularization... IN STEP THREE....

%STEP - now it is ready to compute the gradients using a summation for each trial.
Theta2_grad = 1/m * ( delta_3' * a_2);        % 10x5000 * 5000X26
delta_2 = delta_2(:,2:size(delta_2,2)) ;
Theta1_grad = 1/m * ( delta_2' * a_1);        %  * 25x5000 * 5000X401

capdelta = 0;

capdelta_1 = capdelta_0 + delta_2*a_1' ;
cepdelta_2 = capdelta_1 + delta_3*a_2'


for l=1:1;
for j = 1:401;
for i=1:25;
capdelta_1 = capdelta_0 + delta_2*a_1';
if j != 1,
capd = capdelta_1 + theta1*lambda ; 
else 
capd = capdelta_0 ; 
end
end
end
end



for l=2:2;
for j = 1:26
for i=1:10;
capdelta_2 = capdelta_1 + delta_3*a_2';
if j != 1,
capd = capdelta_2 + theta2*lambda ; 
else 
capd = capdelta_2 ;
end
end
end
end

%10X26
%25X401   %accounts for theta1 being 25x401  instead of 26
% WORKS - TRY TO DIMENSIONIZE THE THETA1GRAD SO THE 26 LOOKS LIKE A 25...
% cant find any training sets...  this means skip the for loop to sum the training examples.  
% HOW DO YOU FIND H AND G FOR NN
% SAME WAY YOU ALWAYS DO -  BUILD UP X Z A FOR A-L


%%%%%%%%%%%%%%%% PART THREE %%%%%%%%%%%%%%%%%%%%%%
for l=1:2;
    if l==1,
        theta_temp = [ones(size(Theta1,2));Theta1];
    else if l==2,
        theta_temp = [ones(size(Theta2,2));Theta2];
    end
    for i=1:size(theta_temp,2);
        regularization_temp = ((lambda/(2*m)) * sum(sum(theta_temp(2:size(theta_temp,1),i) .^ 2))) ;
    end
    regularization += regularization_temp;
end

J = J + regularization ;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
