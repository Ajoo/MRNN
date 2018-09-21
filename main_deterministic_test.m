% Dataset
T = 30;
BATCH_SIZE = 100;
BATCH_SIZE = 250; % enough for T=30
samplebatch = @(B) sampleaddition(B, T);

mnist = MNISTDataset('train', BATCH_SIZE);

% Model
HIDDEN_SIZE = 100;
rnn1 = RNN(2, HIDDEN_SIZE);
rnn2 = RNN(2, HIDDEN_SIZE);
mdl1 = RNNLinearRegressor(rnn1);
mdl2 = RNNLinearRegressor(rnn2);

mdl2.params = mdl1.params;
% fimdiag = @(mdl) fimdiag(mdl, 100);

%% optimizer

% CG Steihaug
opt1 = OldPCGSteihaugOptimizer(mdl1, 1);
alpha = 1;
opt1.thrustradius_decrease = 0.25;
opt1.thrustradius_increase = 2;
opt1.thrustradius_max      = Inf;
opt1.preconditioner        = [];
opt1.reltol                = 1e-5;
opt1.maxiter               = 100;
opt1.rejection_threshold   = 0;

opt2 = PCGSteihaugOptimizer(mdl2, 1);
opt2.reltol                = 1e-5;
opt2.maxiter               = 100;

% ASGD
% opt = ASGDOptimizer(mdl, 1e-4);
% opt.momentum = 0.7;

% ADAM
% opt = ADAMOptimizer(mdl, 1e-4);

% opt.accept = true;
% opt.lr_increase = 10^(1/10e3);
% opt.lr_decrease = 0.1^(1/10);
% opt.lr_max = Inf;
% opt.rejection_threshold = 0;
%% Sample Data and compute initial losses
[x, y] = samplebatch(BATCH_SIZE);
[~, l1] = call(mdl1, x, y);
[~, l2] = call(mdl2, x, y);
%%
loss = []; i = 1;
%%
N = 250;

for i=1:N
    l1 =  step(opt1, l1);
    loss(i,1) = 2*l1/BATCH_SIZE;
end

for i=1:N
    l2 =  step(opt2, l2);
    loss(i,2) = 2*l2/BATCH_SIZE;
    i
end
%%
figure(1); plot(loss)

figure(2); plot(opt1)

figure(2); plot(opt2)