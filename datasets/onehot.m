function X = onehot(K, x)
    X = bsxfun(@eq, (1:K).', x+1);
end
