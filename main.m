T = 30;
BATCH_SIZE = 100;
HIDDEN_SIZE = 100;

rnn = RNN(2, HIDDEN_SIZE);
mdl = RNNLinearRegressor(rnn);
p0 = mdl.params;

fimdiag = @(mdl) fimdiag(mdl, 100);

opt = PCGOptimizer(mdl, 100);
%%
opt.options.MaxDamping = Inf;
opt.options.Preconditioner = [];
opt.options.PreconditionerSupression = 1;
opt.options.RelTol = 1e-5;
opt.options.MaxIter = 100;
opt.options.RejectionThreshold = 0;
opt.options.HotRestart = 0;
%%
[x, y] = sampleaddition(BATCH_SIZE, T);
%%
[~, l] = call(mdl, x, y);
loss = []; rho = []; iter = []; damping = [];
stepsizes = []; flag = []; relres = [];
i = 1;
%%
N = 200;
loss = [loss; zeros(N,1)];
rho = [rho; zeros(N,1)];
iter = [iter; zeros(N,1)];
damping = [damping; zeros(N,1)];
stepsizes = [stepsizes; zeros(N,1)];
flag = [flag; zeros(N,1)];
relres = [relres; zeros(N,1)];

for i=i:i+N
    [l, rho(i), flag(i), relres(i), iter(i)] =  step(opt, l);
    stepsizes(i) = norm(opt.state.previousstep);
    damping(i) = opt.state.damping;
    loss(i) = 2*l/BATCH_SIZE;
    if mod(i, 10) == 0
        i
    end
end
% disp(['loss: ', num2str(2*l/BATCH_SIZE)])
% disp(['rho: ', num2str(rho)])
% disp(['iter: ', num2str(iter)])
% disp(['damping: ', num2str(opt.state.damping)])
% disp(['step size: ', num2str(norm(opt.state.previousstep))])
%%
[xval, yval] = sampleaddition(BATCH_SIZE, T);
[yhval, lval] = call(mdl, xval, yval);
lval = lval*2/BATCH_SIZE