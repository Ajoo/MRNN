% Dataset
T = 100;
BATCH_SIZE = 100;
BATCH_SIZE = 1000; % enough for T=30
samplebatch = @(B) sampleaddition(B, T);

mnist = MNISTDataset('train', BATCH_SIZE);

% Model
HIDDEN_SIZE = 100;
rnn = RNN(2, HIDDEN_SIZE);
mdl = RNNLinearRegressor(rnn);
% fimdiag = @(mdl) fimdiag(mdl, 100);

%% optimizer

% CG Steihaug
opt = PCGSteihaugOptimizer(mdl, 1);
alpha = 1;
opt.thrustradius_decrease = 0.25;
opt.thrustradius_increase = 2;
opt.thrustradius_max      = Inf;
opt.preconditioner        = [];
opt.reltol                = 1e-5;
opt.maxiter               = 100;
opt.rejection_threshold   = 0;

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
%% Sample Validation Data and compute initial losses
[xval, yval] = samplebatch(BATCH_SIZE);
[~, valloss] = evaluate(mdl, xval, yval);
valloss = valloss*2/size(yval, 2);
%% Sample Data and compute initial losses
[x, y] = samplebatch(BATCH_SIZE);
[~, l] = call(mdl, x, y);
%%
loss = []; i = 1;
%%
N = 1000;
DECIMATION = 10;

for i=i:i+N
    l =  step(opt, l);
    loss(i) = 2*l/BATCH_SIZE;
    if mod(i, DECIMATION) == 0
        [~, vl] = evaluate(mdl, xval, yval);
        valloss(end+1) = 2*vl/size(yval, 2);
        fprintf('Iter: %i, Train Error: %1.4f, Val Error: %1.4f\n', ...
            i, loss(i), valloss(end));
    end
end
%%
figure(1); plot(loss), hold on, plot(DECIMATION*(1:numel(valloss)), valloss, '.-')

figure(2); plot(opt)