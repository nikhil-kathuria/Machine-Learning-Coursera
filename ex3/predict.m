function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

disp(size(p))

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

% Add bias vector to the training data X
X = [ones(m, 1) X];

% Compute the new layer values by Taking Simoid(dot product of observation and Theta)
% Doing matrix multiplication will evaluate final value for all observation
XL1 = sigmoid(X * Theta1');

% Add bias to second layer
XL1 = [ones(size(XL1, 1), 1) XL1];

% Compute Hypothesis again in same manner as done for Layer 2. Dimensionality reduces to m * 10
XL2 = sigmoid(XL1 * Theta2');

% Extract the index of highest element per row. So M size vector
[v, p] = max(XL2, [] , 2);







% =========================================================================


end
