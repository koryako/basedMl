function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%



h = X * theta;

h_error = h - y;

sum_squared_error = sum(h_error .^2);

reg_term_left = (1/(2*m)) * sum_squared_error;

theta_without_bias = theta(2:end);
sum_squared_theta = sum(theta_without_bias .^2);
reg_term_right = (lambda / (2 * m)) * sum_squared_theta;

J = reg_term_left + reg_term_right;


grad = (1/m) * X' * h_error;

grad(2:end) += (lambda / m) * theta_without_bias;


% =========================================================================

grad = grad(:);

end
