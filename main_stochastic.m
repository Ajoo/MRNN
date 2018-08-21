T = 30;
BATCH_SIZE = 250;
samplebatch = @(B) sampleaddition(B, T);

HIDDEN_SIZE = 100;
rnn = RNN(2, HIDDEN_SIZE);
mdl = RNNLinearRegressor(rnn);

opt = ADAMOptimizer(mdl, 2e-4); % for T <= 50
% opt = ADAMOptimizer(mdl, 2e-4);
% opt = SGDOptimizer(mdl, 1e-5, 'momentum', 0.9);
% opt.accept = true;

i = 1;
loss = [];
ploss = [];
%%
N = 50000;
DECIMATION = 100;
loss = [loss; zeros(N,1)];
ploss = [ploss; zeros(N,1)];

%[x, y] = samplebatch(BATCH_SIZE);
for i=i:i+N
    [x, y] = samplebatch(BATCH_SIZE);
    [~, loss(i)] = call(mdl, x, y);
    ploss(i) = step(opt, loss(i));
    loss(i) = loss(i)*2/BATCH_SIZE;
    ploss(i) = ploss(i)*2/BATCH_SIZE;
    
    if mod(i, DECIMATION) == 0
        fprintf('Iter: %i, Error: %1.4f\n', i, mean(loss(i-DECIMATION+1:i)));
    end
end

%%
plot(loss);
hold on
