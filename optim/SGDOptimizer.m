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
        lr_max = Inf;
        lr_increase = 2;
        lr_decrease = 1/4;
        rejection_threshold = 0;
        
        log = Log('lr', 'reductionratio');
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
        
        function plot(opt)
%             figure(1);
%             plot(getfield(opt.log, 'loss')), hold on

%             figure(2);
            subplot(2,1,1); plot(max(getfield(opt.log, 'reductionratio'),0)), hold on
            subplot(2,1,2); semilogy(getfield(opt.log, 'lr')), hold on
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
            step = g;
            if momentum_ > 0
                opt.m = momentum_*opt.m + (1-damping_)*g;
                if nesterov_
                    step = step + momentum_*opt.m;
                else
                    step = opt.m;
                end
            end
            step = - lr_*step;
            opt.model.params = opt.model.params + step;
            
            newloss = loss;
            if opt.accept
                [~, newloss] = recall(opt.model);
                predchange = g'*step;
                reductionratio = (newloss-loss)/predchange;
                if ~update(opt, reductionratio)
                    opt.model.params = opt.model.params - step;
                    newloss = loss;
                end
                append(opt.log, lr_, reductionratio)
            else
                append(opt.log, lr_, NaN)
            end
        end
        
        function accept = update(opt, reductionratio)
            if reductionratio < 0.25
                opt.lr = opt.lr*opt.lr_decrease;
            elseif reductionratio > 0.75
                opt.lr = opt.lr*opt.lr_increase;
                opt.lr = min(opt.lr, opt.lr_max);
            end
            accept = (reductionratio > opt.rejection_threshold);
        end
    end
end