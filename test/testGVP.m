nx = 2;
nh = 5;
nb = 100;
nt = 100;

rnn = RNN(nx, nh, 'tanh');
mdl = RNNLinearRegressor(rnn);
np = mdl.paramsize;

x = randn(nx,nb,nt);
y = sum(sum(x,3),1);

%%
[yh, l] = call(mdl, x, y);
f = zeros(np,1);
for i=1:nb
    u = zeros(1, nb); u(i) = 1;
    f = f + bdiff(mdl, u).^2;
end
f2 = fimdiag(mdl);

% from full Gauss-Newton Matrix
G = eye(np);
g = zeros(np,1);
for i=1:np
    [G(:,i), g(i)] = gvp(mdl, G(:,i));
end

[f2, f, diag(G)]