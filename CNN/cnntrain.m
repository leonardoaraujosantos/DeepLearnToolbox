function net = cnntrain(net, x, y, opts)
    % 'm' is the number of training examples.
    m = size(x, 3);
    numbatches = m / opts.batchsize;
    if rem(numbatches, 1) ~= 0
        error('numbatches not integer');
    end
    if isa(x,'gpuArray')
        net.rL = gpuArray(zeros(1,numbatches));
    else
        net.rL = [];
    end
    
    % For each of the training epochs (one training epoch is one pass over
    % the dataset)...
    for i = 1 : opts.numepochs
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);
        tic;
        % Create a row vector containing the values 1 to 'm' in random 
        % order. We'll use these as indeces to take the training examples 
        % in random order.
        %
        % Note that we'll be using a different random order for each epoch.
        
        kk = randperm(m);
        
        % For each batch...
        for l = 1 : numbatches
            
            % Select the training examples for the current batch. Use the
            % randomly sorted indeces in 'kk'.
            batch_x = x(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            batch_y = y(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            
            % Perform a feed-forward evaluation of the current network on
            % the training batch. This will populate all of the output
            % variables in the 'net' structure.
            net = cnnff(net, batch_x);
            
            % Calculate gradients using back-propagation.
            net = cnnbp(net, batch_y);
            
            % Update the parameters by applying the gradients.
            net = cnnapplygrads(net, opts);
            
            % Calculate the Loss
            if isempty(net.rL)
                net.rL(1) = net.L;
            end
            net.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * net.L;
        end
        toc;
    end
    
end
