T = 30;
BATCH_SIZE = 100;
HIDDEN_SIZE = 100;

samplebatch = @() sampleaddition(BATCH_SIZE, T);

rnn = RNN(2, HIDDEN_SIZE);
mdl = RNNLinearRegressor(rnn);

% opt = ADAMOptimizer(mdl, 2e-4); % for T <= 50
% opt = ADAMOptimizer(mdl, 2e-4);
opt = SGDOptimizer(mdl, 1e-5, 'momentum', 0.9);
opt.accept = true;

i = 1;
loss = [];
ploss = [];
%%
N = 50000;
DECIMATION = 100;
loss = [loss; zeros(N,1)];
ploss = [ploss; zeros(N,1)];

for i=i:i+N
    [x, y] = samplebatch();
    [~, loss(i)] = call(mdl, x, y);
    ploss(i) = step(opt, loss(i));
    loss(i) = loss(i)*2/BATCH_SIZE;
    ploss(i) = ploss(i)*2/BATCH_SIZE;
    
    if mod(i, DECIMATION) == 0
        fprintf('Iter: %i, Train Error: %1.4f\n', i, mean(loss(i-DECIMATION+1:i)));
    end
end

%%
plot(loss);
hold on
