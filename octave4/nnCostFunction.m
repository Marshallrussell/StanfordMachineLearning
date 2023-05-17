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

%%%%%%%%%%%%%% PART ONE FEED FORWARD %%%%%%%%%%%%%%%

a1 = [ones(m,1) X];                                                 %5000x401

z2 = a1*Theta1';                                                    %5000x401 * 401x25
a2 = sigmoid(z2);                                                   %5000x25
a2 = [ones(m,1) a2];                                                %5000x26

z3 = a2*Theta2';                                                    %5000x26 * 26x10
a3 = sigmoid(z3);                                                   %5000x10
h = a3;


for i = 1:size(X,1)
y2 = zeros(size(Theta2,1),1);
y2(y(i)) = 1;   
J += 1/m * (-y2'*(log(h(i,:)')) - (1-y2')*(log(1-h(i,:)' ) ) ) ;    %10x1 * 1x10  1x1
y3(i,:) = y2' ;
end

%%%%%%%%%%%%%% PART TWO BACKPROPAGATION %%%%%%%%%%%%%%%


delta3 = h - y3 ;
Theta2_grad = 1/m * delta3' * a2  ;                                 %5000x10  * 5000x26   10x26

delta2 = delta3 * Theta2 .* a2 .* (1-a2)  ;                         %5000x10  * 10x26 .* 5000x26   
delta2 = delta2(:,2:size(delta2,2)) ;                               %5000x25
Theta1_grad = 1/m * delta2' * a1 ;                                  %5000x25 * 5000x401   25x401


%%%%%%%%%%%%%% PART THREE REGULARIZATION %%%%%%%%%%%%%%%

J += lambda/(2*m)*( sum(sum(Theta1.^2))+sum(sum(Theta2.^2))-sum(Theta1(:,1).^2)-sum(Theta2(:,1).^2) );

for i=1:size(Theta1,1)
for j=2:size(Theta1,2)
Theta1_grad(i,j) += lambda/m*Theta1(i,j);
end
end
for i=1:size(Theta2,1)
for j=2:size(Theta2,2)
Theta2_grad(i,j) += lambda/m*Theta2(i,j);
end
end


%%%%%%%%%%%%%%%%%%%%%%%% END %%%%%%%%%%%%%%%%%%%%%%%%%%%%

grad = [Theta1_grad(:) ; Theta2_grad(:)];

end

% 100 of 100
