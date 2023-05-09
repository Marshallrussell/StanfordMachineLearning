% separate the two l variables, take Theta1 and Theta2 in series...
% what about sum ( sum ( Theta_temp))

regularization = 0;
regularization += sum(sum(Theta1_grad .^ 2));
regularization += sum(sum(Theta2_grad .^ 2));

for i=1:size(Theta1_grad,1)
regularization -= Theta1_grad(i,1).^2;

for i=1:size(Theta2_grad,1)
regularization -= Theta2_grad(i,1).^2;


J += (lambda/(2*m)*regularization) ;
