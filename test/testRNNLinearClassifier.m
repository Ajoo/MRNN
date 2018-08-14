nk = 3;
nx = 2;
nh = 5;
nb = 10;
nt = 100;
fullseq = true;

rnn = RNN(nx, nh, 'tanh');
mdl = RNNLinearClassifier(rnn, nk, [], fullseq);

x = randn(nx,nb,nt);
[~, i] = max(randn(nk,nb,nt), [], 1);
y = onehot(nk, i-1);
%%

np = mdl.paramsize;
[~, l] = call(mdl, x, y);

%%
p = mdl.params;
e = 1e-4;

u_l = bdiff(mdl, [], 1);
l_v = zeros(np, 1);
l_ve = l_v;
for i=1:np
    v = zeros(np, 1);
    v(i) = 1;
    [~, l_v(i)] = fdiff(mdl, v);
    
    mdl.params = p + v*e;
    [~, l_ve(i)] = evaluate(mdl, x, y);
    mdl.params = p;
end
l_ve = (l_ve - l)/e;

disp([l_ve, l_v, u_l])
%%
gv = gvp(mdl, v);

%%
