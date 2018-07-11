nx = 2;
nh = 5;
nb = 10;
nt = 100;

rnn = RNN(nx, nh, 'tanh');
mdl = RNNLinearRegressor(rnn);

x = randn(nx,nb,nt);
y = sum(sum(x,3),1);

np = mdl.paramsize;
i = 1;

v = zeros(np, 1);
v(i) = 1;

l = call(mdl, x, y);
l_v = fdiff(mdl, v);
u_l = bdiff(mdl, ones(1,nb));

disp([sum(l_v), u_l(i)])
%%
gv = gvp(mdl, v);

%%
p = mdl.params;
e = 1e-4;
mdl.params = p + v*e;
l_ve = call(mdl, x, y);

l_ve = (l_ve - l)/e;

disp([l_v, l_ve])