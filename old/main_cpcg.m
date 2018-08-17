T = 30;
BATCH_SIZE = 1000;
HIDDEN_SIZE = 100;

rnn = RNN(2, HIDDEN_SIZE);
mdl = RNNLinearRegressor(rnn);
p0 = mdl.params;

fimdiag = @(mdl) fimdiag(mdl, 100);

opt = PCGSteinhaugOptimizer(mdl, 1);
%%
alpha = 1;
opt.options.ThrustRadiusReductionFactor = 0.1;
opt.options.ThrustRadiusIncreaseFactor = 2;
opt.options.MaxThrustRadius = Inf;
opt.options.Preconditioner = [];
opt.options.RelTol = 1e-5;
opt.options.MaxIter = 100;
opt.options.RejectionThreshold = 0;
%%
[x, y] = sampleaddition(BATCH_SIZE, T);
[xval, yval] = sampleaddition(BATCH_SIZE, T);
%%
[~, l] = call(mdl, x, y);
[~, valloss] = evaluate(mdl, xval, yval);
valloss = valloss*2/BATCH_SIZE;

loss = []; rho = []; iter = []; thrustradius = [];
stepsizes = []; flag = []; relres = [];
i = 1;
%%
N = 300;
DECIMATION = 10;

loss = [loss; zeros(N,1)];
rho = [rho; zeros(N,1)];
iter = [iter; zeros(N,1)];
thrustradius = [thrustradius; zeros(N,1)];
stepsizes = [stepsizes; zeros(N,1)];
flag = [flag; zeros(N,1)];
relres = [relres; zeros(N,1)];

for i=i:i+N
    [l, rho(i), flag(i), relres(i), iter(i)] =  step(opt, l);
    stepsizes(i) = norm(opt.state.previousstep);
    thrustradius(i) = opt.state.thrustradius;
    loss(i) = 2*l/BATCH_SIZE;
    if mod(i, DECIMATION) == 0
        [~, vl] = evaluate(mdl, xval, yval);
        valloss(end+1) = 2*vl/BATCH_SIZE;
        fprintf('Iter: %i, Train Error: %1.4f, Val Error: %1.4f\n', ...
            i, loss(i), valloss(end));
    end
end


%%
figure(1); plot(loss), hold on, plot(DECIMATION*(1:numel(valloss)), valloss, '.-')

figure(2); 
subplot(2,2,1); plot(max(rho,0)), hold on
subplot(2,2,2); plot(iter), hold on
subplot(2,2,3); plot(thrustradius), hold on
subplot(2,2,4); plot(stepsizes)