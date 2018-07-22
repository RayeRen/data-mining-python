load('TDT2_data', 'fea', 'gnd');

% YOUR CODE HERE
options = [];
options.NeighborMode = 'KNN';
options.k = 5;
options.WeightMode = 'Binary';
options.t = 1;
W = constructW(fea,options);

save('W_data','W')