nx = 2;
nh = 10;
nb = 30;
nt = 100;
rnn = RNN(nx,nh, 'tanh');
%rnn = IndRNN(nx, nh, sqrt(2)*nt, 'tanh');
x = randn(nx, nb, nt);
[h_end, h] = call(rnn, x);


%%
ip = 1;
v = zeros(rnn.paramsize, 1); v(ip) = 1;
[h_end_v, h_v] = fdiff(rnn, v);

e = 1e-4;
p0 = rnn.params;
rnn.params = p0 + e*v;
[h_end_ve, h_ve] = evaluate(rnn, x);
rnn.params = p0;
h_end_ve = (h_end_ve - h_end)/e;
h_ve = (h_ve - h)/e;

disp(max(max(abs(h_end_v-h_end_ve))))
assert(max(max(abs(h_end_v-h_end_ve))) < 10*e, 'fdiff result doesn''t match finite differences!')
assert(max(max(max(abs(h_v-h_ve)))) < 10*e, 'fdiff result doesn''t match finite differences!')
%%
np = rnn.paramsize;
ih = 1; 
ib = 2;
it = 100;

u = zeros(size(h));
u_end = zeros(size(h_end));
u(ih,ib,it) = 1;
% u_end(ih,ib) = 1;
[u_h_p] = bdiff(rnn, u_end, u);

u_h_pf = zeros(np,1);
for ip=1:np
    v = zeros(rnn.paramsize, 1); v(ip) = 1;
    [h_end_v, h_v] = fdiff(rnn, v);
    u_h_pf(ip) = h_v(ih, ib, it);
%     u_h_pf(ip) = h_end_v(ih, ib);
end
[u_h_p u_h_pf]
assert(max(abs(u_h_p-u_h_pf)) < 100*eps, 'bdiff result doesn''t match fdiff!')
