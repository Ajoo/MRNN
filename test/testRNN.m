rnn = RNN(2,5, 'tanh');
x = randn(2,10,100);
h_end = sim(rnn, x);

v = zeros(rnn.paramsize, 1); v(1) = 1;
h_end_v = fdiff(rnn, v);
%%
e = 1e-4;
rnn.params = rnn.params + e*v;
[h_end_ve] = sim(rnn, x);
h_end_ve = (h_end_ve - h_end)/e;

%%
u = zeros(size(h_end));
u(1,1) = 1;
[u_h_p] = bdiff(rnn, u);

u_h_p(1)
h_end_v(1,1)