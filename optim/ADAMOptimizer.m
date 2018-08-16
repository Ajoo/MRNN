classdef ADAMOptimizer < handle
    %ADAMOPTIMIZER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties % algorithm options
        model
        lr = 1e-3;
        beta = [0.9 0.999];
        eps = 1e-8;
        % regularization
        l2 = 0;
        weightdecay = 0;
        % acceptence test
        accept = false;
        lr_increase = 1;
        lr_decrease = 1;
    end
    properties (Access=private) % algorithm state
        m = 0;
        v = 0;
        t = 0;
    end
    
    methods
        function opt = ADAMOptimizer(model, lr, varargin)
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
            opt.v = 0;
            opt.t = 0;
        end
        
        function newloss = step(opt, loss)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            lr_ = opt.lr;
            beta_ = opt.beta;
            eps_ = opt.eps;
            weightdecay_ = opt.weightdecay;
            l2_ = opt.l2;
            
            opt.t = opt.t + 1;
            t_ = opt.t;
            g = bdiff(opt.model, [], 1);
            if l2_ > 0
                g = g + l2_*opt.model.params;
            end

            % update running moments
            opt.m = beta_(1)*opt.m + (1-beta_(1))*g;
            opt.v = beta_(2)*opt.v + (1-beta_(2))*g.^2;
            % compute bias corrected step size
            alpha = lr_*sqrt(1-beta_(2)^t_)/(1-opt.beta(1)^t_);
            
            if weightdecay_ > 0
                opt.model.params = (1-lr_*weightdecay_)*opt.model.params;
            end
            step = -alpha*opt.m./(sqrt(opt.v) + eps_);
            opt.model.params = opt.model.params + step;
            
            newloss = loss;
            if opt.accept
                [~, newloss] = recall(opt.model);
                if ~update(opt, newloss-loss)
                    opt.model.params = opt.model.params - step;
                    newloss = loss;
                end
            end
        end
        
        function accept = update(opt, dloss)
            accept = dloss <= 0;
            if accept
                opt.lr = opt.lr*opt.lr_increase;
            else
                opt.lr = opt.lr*opt.lr_decrease;
            end
        end
    end
end

