classdef OptimizerBase < handle
    %ADAMOPTIMIZER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties % algorithm options
        model
        % regularization
        l2 = 0;
    end
    
    methods
        function opt = OptimizerBase(model)
            %PCGOPTIMIZER Construct an instance of this class
            %   Detailed explanation goes here
            opt.model = model;
        end
        
        function g = grad(opt)
            g = bdiff(opt.model, [], 1);
            if opt.l2 > 0
                g = g + opt.l2*opt.model.params;
            end
        end
        
        function gv = gvp(opt, v)
            gv = gvp(opt.model, v) + opt.l2*v;
        end
        
        function make_step(opt, step)
            opt.model.params = opt.model.params + step;
        end
        
        function step = compute_step(opt)
            step = -grad(opt);
        end
    end
    
end