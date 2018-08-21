classdef ADAMOptimizer < FirstOrderOptimizer
    %ADAMOPTIMIZER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties % algorithm options
        beta = [0.9 0.999];
        eps = 1e-8;
    end
    properties (Access=private) % algorithm state
        m = 0;
        v = 0;
        t = 0;
    end
    
    methods
        function reset(opt)
            opt.m = 0;
            opt.v = 0;
            opt.t = 0;
        end
        
        function step = computestep(opt, g)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            beta_ = opt.beta;
            eps_ = opt.eps;
            
            opt.t = opt.t + 1;
            t_ = opt.t;

            % update running moments
            opt.m = beta_(1)*opt.m + (1-beta_(1))*g;
            opt.v = beta_(2)*opt.v + (1-beta_(2))*g.^2;
            % compute bias corrected step size
            alpha = sqrt(1-beta_(2)^t_)/(1-opt.beta(1)^t_);
            
            % compute step
            step = -alpha*opt.m./(sqrt(opt.v) + eps_);
        end
    end
end

