% Dataset
T = 30;
BATCH_SIZE = 100;
BATCH_SIZE = 250; % enough for T=30
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
opt.reltol                = 1e-5;
opt.maxiter               = 100;

% ASGD
% opt = ASGDOptimizer(mdl, 1e-4);
% opt.momentum = 0.7;

% ADAM
% opt = ADAMOptimizer(mdl, 1e-4);
%% meta-optimizer

mopt = MetaOptimizer(opt);
mopt.nsubsteps = 10;

%% Sample Validation Data and compute initial losses
[xval, yval] = samplebatch(BATCH_SIZE);
[~, valloss] = evaluate(mdl, xval, yval);
valloss = valloss*2/size(yval, 2);
%%
N = 10;
loss = [];
for i=1:N
    [x, y] = samplebatch(BATCH_SIZE);
    [~, l0] = call(mdl, x, y);
    lf =  step(mopt, l0);
    loss(i,:) = 2*[l0 lf]/BATCH_SIZE;
    
    [~, vl] = evaluate(mdl, xval, yval);
    valloss(end+1) = 2*vl/size(yval, 2);
    fprintf('Iter: %i, Train Error: %1.4f:%1.4f, Val Error: %1.4f\n', ...
            i, loss(i,1), loss(i,2), valloss(end));
end
%%
figure(1); plot(loss), hold on, plot(DECIMATION*(1:numel(valloss)), valloss, '.-')

figure(2); plot(opt)