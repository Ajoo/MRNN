classdef ModelOptimizer < handle
    %OPTIMIZERBASE Summary of this class goes here
    %   Detailed explanation goes here
    
    properties % algorithm options
        model
        % regularization
        l2 = 0;
    end
    
    methods
        function opt = ModelOptimizer(model)
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
            gv = gvp(opt.model, v);
            if opt.l2 > 0
                 gv = gv + opt.l2*v;
            end
        end
        
        function hv = hvp(opt, v)
            hv = hvp(opt.model, v);
            if opt.l2 > 0
                 hv = hv + opt.l2*v;
            end
        end
        
        function make_step(opt, step)
            opt.model.params = opt.model.params + step;
        end
        
        function step = compute_step(opt, maxstep)
            step = -grad(opt);
            if nargin >= 2 && ~isempty(maxstep)
                step = step/norm(step)*maxstep;
            end
        end
    end

end