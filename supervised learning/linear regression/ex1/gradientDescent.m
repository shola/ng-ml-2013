function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Linear Algebra in theta calculation: 
%   2x1 = 2x1 - (2x97 * (97x2 * 2x1 - 97x1))
%   2x1 = 2x1 - (2x97 * 97x1)
%   2x1 = 2x1 - 2x1

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
   theta = theta - (alpha/m) * (X' * (X * theta - y));
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
end

end

% run this plot command from the command line to visualize J_history, and make sure it converges
% plot(1:iterations, jh)
% legend('Jhistory')
