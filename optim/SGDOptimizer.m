classdef SGDOptimizer < handle
    %ADAMOPTIMIZER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties % algorithm options
        model
        lr = 1e-3;
        momentum = 0;
        damping = 0;
        nesterov = true;
        l2 = 0; % regularization
        
        % acceptence test
        accept = false;
        accept_threshold = 0;
        lr_increase = 1;
        lr_decrease = 1;
    end
    properties (Access=private) % algorithm state
        m = 0;
    end
    
    methods
        function opt = SGDOptimizer(model, lr, varargin)
            %PCGOPTIMIZER Construct an instance of this class
            %   Detailed explanation goes here
            opt.model = model;
            if nargin >= 2 && ~isempty(lr)
                opt.lr = lr;
            end
            for i=1:2:numel(varargin)
                opt.(varargin{i}) = varargin{i+1};
            end
        end
        
        function reset(opt)
            opt.m = 0;
        end
        
        function newloss = step(opt, loss)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            lr_ = opt.lr;
            momentum_ = opt.momentum;
            damping_ = opt.damping;
            nesterov_ = opt.nesterov;
            l2_ = opt.l2;
            
            g = bdiff(opt.model, [], 1);
            if l2_ > 0
                g = g + l2_*opt.model.params;
            end
            if momentum_ > 0
                opt.m = momentum_*opt.m + (1-damping_)*g;
                if nesterov_
                    g = g + momentum_*opt.m;
                else
                    g = opt.m;
                end
            end
            opt.model.params = opt.model.params - lr_*g;
            
            newloss = loss;
            if opt.accept
                [~, newloss] = recall(opt.model);
                if ~update(opt, newloss-loss)
                    opt.model.params = opt.model.params + lr_*g;
                    newloss = loss;
                end
            end
        end
        
        function accept = update(opt, dloss)
            accept = dloss <= opt.accept_threshold;
            if accept
                opt.lr = opt.lr*opt.lr_increase;
            else
                opt.lr = opt.lr*opt.lr_decrease;
            end
        end
    end
end