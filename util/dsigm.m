function X = dsigm(P)
    t = 1./(1+exp(-P));    
    X  = t .* (1 - t);
end