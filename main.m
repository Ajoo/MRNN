T = 30;
BATCH_SIZE = 1000;
HIDDEN_SIZE = 100;

rnn = RNN(2, HIDDEN_SIZE);
mdl = RNNLinearRegressor(rnn);
opt = PCGOptimizer(mdl, 100, 'RelTol', 1e-5);

%%
[x, y] = sampleaddition(BATCH_SIZE, T);
%%
[~, l] = call(mdl, x, y);
%%
tic
[l, rho, flag, relres, iter] =  step(opt, l);
toc
disp(['loss: ', num2str(l/BATCH_SIZE)])
disp(['rho: ', num2str(rho)])
disp(['iter: ', num2str(iter)])
disp(['damping: ', num2str(opt.state.damping)])
disp(['step size: ', num2str(norm(opt.state.previousstep))])