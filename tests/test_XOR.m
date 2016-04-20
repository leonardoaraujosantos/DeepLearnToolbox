% XOR input for x1 and x2
X = [0 0; 0 1; 1 0; 1 1];
% Desired output of XOR
Y = [0;1;1;0];

% Initializa random number generator
rng(0,'v5uniform');

nn = nnsetup([2 2 1]);

% Change meta-parameters
nn.activation_function = 'sigm';
nn.learningRate = 2;
nn.scaling_learningRate = 1;
nn.weightPenaltyL2 = 0;
% Between 0 and 1
nn.momentum = 0.0; 
opts.numepochs =  6000;
% Batchsize of 4 (Batch gradient descent)
% Batchsize of 1 (Stochastic gradient descent)
% Batchsize of 2 (Mini-batch gradient descent)
opts.batchsize = 4; 
opts.plot=0;

% Pre-initialize vectors with known values to help debug
nn.W{1} = [-0.7690    0.6881   -0.2164; -0.0963    0.2379   -0.1385];
nn.W{2} = [-0.1433   -0.4840   -0.6903];
 
nn = nntrain(nn, X, Y, opts);

%to display the results
testInpx1 = [-1:0.1:1];
testInpx2 = [-1:0.1:1];
[X1, X2] = meshgrid(testInpx1, testInpx2);
testOutRows = size(X1, 1);
testOutCols = size(X1, 2);
testOut = zeros(testOutRows, testOutCols);
for row = [1:testOutRows]
    for col = [1:testOutCols]
        test = [X1(row, col), X2(row, col)];
        %% Forward pass
        nn = nnff(nn, test, zeros(size(test,1)));       
        testOut(row, col) =nn.a{3};
    end
end
figure(2);
surf(X1, X2, testOut);
title('Prediction surface');

figure(1);
plot (nn.L_vec);
title('Cost vs epochs');

