BATCH_SIZE = 100;
IDX = 339:439;
mnist = MNISTDataset('train', BATCH_SIZE);
samplebatch = @(B) truncate_seq(@() get_batch(mnist, B), IDX);

% Model
HIDDEN_SIZE = 100;
rnn = RNN(1, HIDDEN_SIZE);
mdl = RNNLinearClassifier(rnn, 10);

opt = ADAMOptimizer(mdl, 2e-4); % for T <= 50
% opt = SGDOptimizer(mdl, 1e-5, 'momentum', 0.9);

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
    [ah, loss(i)] = call(mdl, x, y);
    ploss(i) = step(opt, loss(i));
    loss(i) = loss(i)/BATCH_SIZE;
    ploss(i) = ploss(i)/BATCH_SIZE;
    
    if mod(i, DECIMATION) == 0
        fprintf('Iter: %i, Error: %1.4f\n', i, mean(loss(i-DECIMATION+1:i)));
    end
end

%%
plot(loss);
hold on

% is sequential MNIST a hard problem? can we get by subsampling only the
% last few pixels?

%%
[x, y] = samplebatch(1000);
[ah, l] = evaluate(mdl, x, y);

[~, i] = max(y, [], 1);
[~, ih] = max(ah, [], 1);
acc = mean(i==ih)