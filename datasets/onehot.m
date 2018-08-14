function X = onehot(K, x)
    [nc, nb, nt] = size(x);
    assert(nc==1, 'Input channel not one dimensional')
    X = repmat((1:K).', 1, nb, nt);
    X = X==(x+1);
end
