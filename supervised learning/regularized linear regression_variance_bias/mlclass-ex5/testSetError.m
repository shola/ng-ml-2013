function [min_vals] = ...
    testSetError(lambda_vec, error_train, error_val)
%TESTSETERROR computes the test error using the best lambda found (that 
%	yields the lowest cross validation error)
%   function [min_vals] = ...
%    testSetError(lambda_vec, error_train, error_val) returns the best lambda,
%			training error, and cross validation error in a 3x1 vector.
%

% absurdly high start values
min_vals = [10000 10000 10000];

for i = 1:length(lambda_vec)
	if error_val(i) < min_vals(3)
		min_vals = [lambda_vec(i) error_train(i) error_val(i)];
	end
end


