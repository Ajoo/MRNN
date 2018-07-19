rnn = RNN(2,5, 'tanh');
x = randn(2,10,100);
h_end = call(rnn, x);


%%
ip = 1;
v = zeros(rnn.paramsize, 1); v(ip) = 1;
h_end_v = fdiff(rnn, v);

e = 1e-4;
p0 = rnn.params;
rnn.params = p0 + e*v;
[h_end_ve] = evaluate(rnn, x);
rnn.params = p0;
h_end_ve = (h_end_ve - h_end)/e;

disp(max(max(abs(h_end_v-h_end_ve))))
assert(max(max(abs(h_end_v-h_end_ve))) < e, 'fdiff result doesn''t match finite differences!')
%%
np = rnn.paramsize;
ih = 1; 
ib = 1;

u = zeros(size(h_end));
u(ih,ib) = 1;
[u_h_p] = bdiff(rnn, u);
u_h_pf = zeros(np,1);
for ip=1:np
    v = zeros(rnn.paramsize, 1); v(ip) = 1;
    h_end_v = fdiff(rnn, v);
    u_h_pf(ip) = h_end_v(ih, ib);
end
assert(max(abs(u_h_p-u_h_pf)) < 100*eps, 'bdiff result doesn''t match fdiff!')
   
[u_h_p u_h_pf]