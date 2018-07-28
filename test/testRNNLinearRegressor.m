nx = 2;
nh = 5;
nb = 10;
nt = 100;
fullseq = true;

rnn = RNN(nx, nh, 'tanh');
mdl = RNNLinearRegressor(rnn, [], fullseq);

x = randn(nx,nb,nt);
if fullseq
    y = sum(x, 1);
else
    y = sum(sum(x,3),1);
end

np = mdl.paramsize;
yh = call(mdl, x, y);

%%
p = mdl.params;
e = 1e-4;

u_yh = bdiff(mdl, ones(size(y)));
yh_v = zeros(np, size(y,2), size(y,3));
yh_ve = yh_v;
for i=1:np
    v = zeros(np, 1);
    v(i) = 1;
    yh_v(i,:,:) = fdiff(mdl, v);
    
    mdl.params = p + v*e;
    yh_ve(i,:,:) = evaluate(mdl, x, y);
    mdl.params = p;
end
yh_ve = (yh_ve - yh)/e;

u_yhfe = sum(yh_ve(:,:), 2);
u_yhf = sum(yh_v(:,:), 2);
disp([u_yhfe, u_yhf, u_yh])
disp(norm(yh_v(:)-yh_ve(:), 'Inf'))
%%
gv = gvp(mdl, v);

%%
