function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

% Compute the Sigmoid Vector as sigmoid of the hypothesis for each observation
Sig_V = sigmoid(X * theta);

% Now j(theta) is computed multiplying each value of -actual with each value Sig_V
% subtracted by each negated actual multiplied by each value (1 - Sig_V) Vector.
J_Vec = (-y .* log(Sig_V)) - (( 1 - y) .* log(1 - Sig_V));

% Finally compute J by summing all the value of J_Vec and dividing by number of observations.
J = (1.0 / m) * sum(J_Vec) + (lambda / (2.0 * m)) * sum(theta(2:end).^2);

% Hypothesis after sigmoid applied subtracted actual prediction
dif = Sig_V - y;

% Now since each feature vector dot prodout with curret Hypothese(sigmoid) represents
% new gradient value for each observation respectively.

% For regularized logisctic regression first grad is simply product of bias vector and difference
grad(1) = (1.0 / m) * (X(:, 1)' * dif);

% For rest we compute the matrix product of tranpose of matrix excluding bias vector and we
% add the lamdba/m times corresponding theta row value.
grad(2:end) = (1.0 / m) * (X(:, 2:end)' * dif) + (lambda / m) * theta(2:end);


% =============================================================

grad = grad(:);

end
