function nn = nnbp(nn)
%NNBP performs backpropagation
% nn = nnbp(nn) returns an neural network structure with updated weights 
    
    n = nn.n;
    sparsityError = 0;
    switch nn.output
        case 'sigm'
            % Difference 1 (delta calculated differently) (They depend on
            % the cost function choosen?)
            % Yes it depends of the cost function because we're calculating
            % the derivative of the cost function related to the output
            d{n} = - nn.e .* (nn.a{n} .* (1 - nn.a{n}));
            % On my example we negate because error is calculated on different order
            % The way from andrew Ng tutorial converge faster here, but it
            % i showing numerical mismatch (TODO check why)
            %d{n} = - nn.e; 
        case {'softmax','linear'}
            d{n} = - nn.e;
    end
    for i = (n - 1) : -1 : 2
        % Derivative of the activation function
        switch nn.activation_function 
            case 'sigm'
                d_act = nn.a{i} .* (1 - nn.a{i});
                % This is the same, but the previous save one computation
                %d_act2 = dsigm(nn.z{i}(:,2:end));
                %d_act2 = [zeros(4,1) d_act2];
                %d_act = d_act2;                
            case 'tanh_opt'
                d_act = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * nn.a{i}.^2);
        end
        
        if(nn.nonSparsityPenalty>0)
            pi = repmat(nn.p{i}, size(nn.a{i}, 1), 1);
            sparsityError = [zeros(size(nn.a{i},1),1) nn.nonSparsityPenalty * (-nn.sparsityTarget ./ pi + (1 - nn.sparsityTarget) ./ (1 - pi))];
        end
        
        % Backpropagate first derivatives
        if i+1==n % in this case in d{n} there is not the bias term to be removed             
            d{i} = (d{i + 1} * nn.W{i} + sparsityError) .* d_act; % Bishop (5.56)
        else % in this case in d{i} the bias term has to be removed
            d{i} = (d{i + 1}(:,2:end) * nn.W{i} + sparsityError) .* d_act;
        end
        
        if(nn.dropoutFraction>0)
            d{i} = d{i} .* [ones(size(d{i},1),1) nn.dropOutMask{i}];
        end

    end
    
    % Divide the complete delta by the size of the batch
    for i = 1 : (n - 1)
        if i+1==n
            nn.dW{i} = (d{i + 1}' * nn.a{i}) / size(d{i + 1}, 1);
        else
            nn.dW{i} = (d{i + 1}(:,2:end)' * nn.a{i}) / size(d{i + 1}, 1);      
        end
    end
end
