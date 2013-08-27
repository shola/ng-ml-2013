function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.
%   NOTE: to avoid automatic broadcasting, make the dimensions of mu
%   and sigma match those of X_norm.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

mu = mean(X_norm);
mu = repmat(mu, size(X_norm, 1), 1);
X_norm = X_norm - mu;

sigma = std(X_norm);
sigma = repmat(sigma, size(X_norm,1), 1);
X_norm = X_norm ./ sigma;

% ============================================================

end
