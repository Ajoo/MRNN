T = 100;
BATCH_SIZE = 100;
HIDDEN_SIZE = 100;
resample = true;

rnn = RNN(2, HIDDEN_SIZE);
mdl = RNNLinearRegressor(rnn);
p0 = mdl.params;

% opt = ADAMOptimizer(mdl, 2e-3);
opt = SGDOptimizer(mdl, 1e-4, 'momentum', 0.9);

%%
[x, y] = sampleaddition(BATCH_SIZE, T);
[~, loss] = call(mdl, x, y); loss = loss*2/BATCH_SIZE;

if ~resample
    [xval, yval] = sampleaddition(BATCH_SIZE, T);
    [~, valloss] = evaluate(mdl, xval, yval); valloss = valloss*2/BATCH_SIZE;
end
i = 1;
%%
N = 1000;
DECIMATION = 10;
loss = [loss; zeros(N,1)];

for i=i:i+N
    step(opt);
    [~, loss(i)] = recall(mdl);
    loss(i) = loss(i)*2/BATCH_SIZE;
    
    if resample
        [x, y] = sampleaddition(BATCH_SIZE, T);
        if mod(i, DECIMATION) == 0
            fprintf('Iter: %i, Loss: %1.4f\n', i, loss(i));
        end
    elseif mod(i, DECIMATION) == 0
        [~, valloss(end+1)] = evaluate(mdl, xval, yval);
        valloss(end) = 2*valloss(end)/BATCH_SIZE;
        fprintf('Iter: %i, Train Error: %1.4f, Val Error: %1.4f\n', ...
            i, loss(i), valloss(end));
    end
end
% disp(['loss: ', num2str(2*l/BATCH_SIZE)])
% disp(['rho: ', num2str(rho)])
% disp(['iter: ', num2str(iter)])
% disp(['damping: ', num2str(opt.state.damping)])
% disp(['step size: ', num2str(norm(opt.state.previousstep))])


%%
plot(loss);
hold on;
if ~resample
    plot(1:DECIMATION:numel(loss), valloss, '.-')
end
