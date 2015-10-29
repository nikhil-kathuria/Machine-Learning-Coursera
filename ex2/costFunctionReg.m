function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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

% Allocate m observation vector to hold Theta dot product m FeatureRow i.e. m values 
% This is initial hypothesis value.
Hypoth_V = zeros(size(y));

for obindx = 1:m
	Hypoth_V(obindx) = X(obindx,:) * theta;
end

% Calculate sigmoid value for each hypothesis in Hypoth_V Vector. After sigmoid take log
% to fit the j(theta) Cost function formula. This returns the vector with m values.
Sig_V = sigmoid(Hypoth_V);

%disp('Log Vector')
%disp(size(LogHyp_V))

% Now j(theta) is computed multiplying each value of -actual with each value Hypoth_V
% subtracted by each negated actual multiplied by each value (1 - Hypoth_V) Vector.
J_Vec = (-y .* log(Sig_V)) - (( 1 - y) .* log(1 - Sig_V));

%disp('J Vector')
%disp(size(J_Vec))

% Finally compute J by summing all the value of J_Vec and dividing by number of observations.
Ntheta = theta(2:length(theta),:);
J = (1 / m) * sum(J_Vec) + (lambda / (2 * m)) * sum(Ntheta .* Ntheta);

%disp('J value')
%disp(J)


% Now we compute the gradient for each theta (0 to n) as product of (Hypoth - actual) 
% for each observation with each value of feature column i.e. column 0 to n
dif = Sig_V - y;

grad(1) = (1 / m) * sum(dif .* X(:, 1));

for feindx = 2:length(theta)
	grad(feindx) = (1 / m) * sum(dif .* X(:, feindx)) + (lambda / m) * theta(feindx);
end






% =============================================================

end
