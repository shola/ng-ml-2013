function plotData(x, y)
%PLOTDATA Plots the data points x and y into a new figure 
%   PLOTDATA(x,y) plots the data points and gives the figure axes labels of
%   population and profit.

plot(x, y, 'rx', 'Markersize', 10); 	% Plot the data
ylabel('Profit in $10,000s');		% Set the y-axis label
xlabel('Population of City in 10,000s');% Set the x-axis label

%figure; % open a new figure window

% ============================================================

end
