classdef PCGOptimizer < handle
    %PCGOPTIMIZER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        model
        options
        state
        log
    end
    
    methods
        function opt = PCGOptimizer(model, initialdamping, varargin)
            %PCGOPTIMIZER Construct an instance of this class
            %   Detailed explanation goes here
            opt.model = model;
            opt.state = struct('damping', initialdamping);
            
            % default options
            opt.options = struct('MaxDamping', Inf, ...
                                 'RejectionThreshold', 0, ...
                                 'RelTol', 1e-6, ...
                                 'MaxIter', model.paramsize);
            
            for i=1:numel(varargin)/2
                opt.options.(varargin{i}) = varargin{i+1};
            end       
        end
        
        function step(opt, loss)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            [~, g] = bdiff(opt.model, 1);
            ng = norm(g);
            if isnumeric(opt.options.RelTol)
                tol = opt.options.RelTol;
            else
                tol = opt.options.RelTol(ng);
            end
            
            step = pcg(@gvp, -g, tol, opt.options.MaxIter);
            opt.model.params = opt.model.params + step;
            
            damping = opt.state.damping;
            function gv = gvp(v)
                gv = gvp(opt.model, v) + damping*v;
            end
        end
    end
end

