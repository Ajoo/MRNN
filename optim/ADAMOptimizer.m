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
        
        function step(opt)
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
            opt.model.params = opt.model.params ...
                -alpha*opt.m./(sqrt(opt.v) + eps_);
        end
    end
end
