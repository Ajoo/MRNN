classdef ASGDOptimizer < FirstOrderOptimizer
    %ADAMOPTIMIZER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties % algorithm options
        momentum = 0;
        damping = 0;
        nesterov = true;
    end
    properties (Access=private) % algorithm state
        m = 0;
    end
    
    methods
        function reset(opt)
            opt.m = 0;
        end
        
        function step = computestep(opt, g)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            momentum_ = opt.momentum;
            
            if momentum_ > 0
                opt.m = momentum_*opt.m + (1-opt.damping)*g;
                if opt.nesterov
                    g = g + momentum_*opt.m;
                else
                    g = opt.m;
                end
            end
            step = -g;
        end
    end
end