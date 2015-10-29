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

% Add bias vector to the training data X
X = [ones(m, 1) X];

% Compute the hiddent layer values by Taking Simoid(dot product of observation and Theta)
% Doing matrix multiplication will evaluate final value for all observation
z2 = X * Theta1';	% Compute z2
HL = sigmoid(z2);   % Compute a2 by taking sigmoid of z2.

% Add bias to hidden layer i.e. a2
HL = [ones(size(HL, 1), 1) HL];

% Compute Hypothesis(sigmoid) again in same manner as done for Layer 2. Dimensionality reduces to m * k
OLHX = sigmoid(HL * Theta2');	% a3 compute by sigmoid of z2 = Theta2 * (a2+ 1)


% Initialize a Vector with all clases
clases = [1 : size(OLHX, 2)];
% Compute the cost by summing over k clases and then summing over 
for Oindx = 1:m
	label = (y(Oindx) == clases);
	Hypoth = OLHX(Oindx,: );
	Cost = (log(Hypoth) * -label') - (log(1 - Hypoth) * (1 - label)');
	J = J + Cost;
end
J = J / m;

ymatrix = eye(num_labels)(y,:);

%{
% Form a matrix of lables of 5000 * 10 with only one column
% value as 1 depending on class value ranking from 1 to 10(k)


c1 = log(OLHX) * -ymatrix';
c2 = log(1 - OLHX) * (1 - ymatrix)';
cost = c1 - c2 
% Calculate the diagnol sum and divide by m
J = trace(cost) / m;
%}

% Compute regularization cost
regcost = (lambda / (2 * m)) * ( sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)) );
J = J + regcost;


% Gradient computation

% Compute delta last. Result dimension is m * k
dlout = OLHX - ymatrix;

% Compute the hidden layer delta taking product of delta_Out and Theta from 2 to end
% Post that applying element wise scaled by sigmoid gradient of z2 i.e. (Theta * a1) 
dlhid = (dlout * Theta2(:,2:end)) .* sigmoidGradient(z2);

% Compute the Theta1_grad and Theta2_grad product of activation layer and next layer delta
% Compute DELTA now and scale them by 1/m
Theta1_grad = (1 / m) * dlhid' * X;
Theta2_grad = (1 / m) * dlout' * HL;

% Regularized gradient computation. Converting first column of both Theta with only zero values
Theta1(:,1) = zeros(size(Theta1,1),1);
Theta2(:,1) = zeros(size(Theta2,1),1);

% Update the gradient for both Theta1 and Theta2.
Theta1_grad = Theta1_grad + (lambda / m) * Theta1;
Theta2_grad = Theta2_grad + (lambda / m) * Theta2;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
