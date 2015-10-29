function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    
    % Create a copy of Theta (n + 1) * 1 Matrix
    Th_prev = repmat(theta, 1)

    % Iterate over Theta value i.e. from 0 to n
    for findx = 1:length(Th_prev)
        totalsum = 0;

        % Iterate over each observation to calculate sum
        for oindx = 1:m
            featureRow = X(oindx,:);
            hypothesis = featureRow * Th_prev;
            totalsum = totalsum + (hypothesis - y(oindx)) * featureRow(findx);
        end
        % Update theta value at feature column findx
        theta(findx) = Th_prev(findx) - (alpha / m) * totalsum;
    end
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
