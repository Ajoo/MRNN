nx = 2;
nh = 100;
nb = 100;
nt = 30;

rnn = RNN(nx, nh, 'tanh');
mdl = RNNLinearRegressor(rnn);
np = mdl.paramsize;

x = randn(nx,nb,nt);
y = sum(sum(x,3),1);

%%
[yh, l] = call(mdl, x, y);
f = fimdiag(mdl);
%%
% from full Gauss-Newton Matrix
G = eye(np);
g = zeros(np,1);
for i=1:np
    [G(:,i), g(i)] = gvp(mdl, G(:,i));
end

[f, g, diag(G)]
%%
maxiter = 200;
reltol = 0.01;
maxstep = 10;

M = spdiags(f,0,np,np);
%M = speye(np);
b = -bdiff(mdl, [], 1);

[step, flag, relres, iter, resvec] = cpcg(@(v) gvp(mdl, v), b, reltol, maxiter, M, [], maxstep);
[step2, flag2, relres2, iter2, resvec2] = pcg(@(v) gvp(mdl, v), b, reltol, maxiter, M, [], []);


%fullstep = G\b;
