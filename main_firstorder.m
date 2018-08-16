T = 100;
BATCH_SIZE = 100;
HIDDEN_SIZE = 100;

samplebatch = @() sampleaddition(BATCH_SIZE, T);

rnn = RNN(2, HIDDEN_SIZE);
mdl = RNNLinearRegressor(rnn);

opt = ADAMOptimizer(mdl, 2e-3);
% opt = SGDOptimizer(mdl, 1e-4, 'momentum', 0.9);
opt.accept = false;
%%

[x, y] = samplebatch();

loss = [];
ploss = [];
lr = [];
i = 1;
%%
N = 1000;
DECIMATION = 10;
loss = [loss; zeros(N,1)];
ploss = [ploss; zeros(N,1)];
lr = [lr; zeros(N,1)];


for i=i:i+N
    [~, loss(i)] = call(mdl, x, y);
    lr(i) = opt.lr;
    newloss(i) = step(opt, loss(i));
%     loss(i+1) = newloss(i);
    
    [x, y] = samplebatch();
    if mod(i, DECIMATION) == 0
        fprintf('Iter: %i, Loss: %1.4f\n', i, loss(i)*2/BATCH_SIZE);
    end
end


%%
figure(1)
plot(loss*2/BATCH_SIZE);
hold on;
plot(newloss*2/BATCH_SIZE);

figure(2)
plot(lr)