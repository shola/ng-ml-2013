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

% size(nn_params) = 10285, unrolled thetas
% input_layer_size  = 400;  % 20x20 Input Images of Digits
% hidden_layer_size = 25;   % 25 hidden units
% num_labels = 10;          % 10 labels, from 1 to 10   
% size(X) = 5000 x 400
% size(y) = 5000 x 1

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
% size(Theta1) = 25 x 401
% size(Theta2) = 10 x 26
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
y = eye(num_labels)(y,:); % convert y to a 5000 x 10 vector

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

% remember to add the a0 values!
a1 = [ones(m,1), X]; % 5000 x 401
z2 = a1 * Theta1'; % 5000x401 * 401x25 => 5000x25
a2 = [ones(m,1), sigmoid(z2)]; % 5000 x 26
z3 = a2 * Theta2'; % 5000x26 * 26x10 => 5000x10
a3 = sigmoid(z3); % 5000 x 10
h_theta = a3;

for i = 1:m
	J -= (1/m)*(y(i,:) * log(h_theta(i,:))' + 
					  (1 - y(i,:)) * log(1 - h_theta(i,:))');
end

d3 = a3 - y; %5000x10
d2 = d3 * Theta2(:, 2:end) .* sigmoidGradient(z2); % 5000x10 * 10x25 .*5000x25=> 5000x25
Theta1_grad = d2' * a1; % 25x5000 * 5000x401=> 25x401
Theta2_grad = d3' * a2; % 10x5000 * 5000x26 => 10x26

Theta1_grad /= m;
Theta2_grad /= m;

%Regularize Theta_grads
Theta1_grad = [Theta1_grad(:, 1), (Theta1_grad(:, 2:end) + (lambda/m) * Theta1(:, 2:end))];
Theta2_grad = [Theta2_grad(:, 1), (Theta2_grad(:, 2:end) + (lambda/m) * Theta2(:, 2:end))];

% sum up everything except for the first columns
J += (lambda/(2*m)) * ...
		(sum(sum(Theta1(:, 2:end).*Theta1(:, 2:end))) + ...
		 sum(sum(Theta2(:, 2:end).*Theta2(:, 2:end))));

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
