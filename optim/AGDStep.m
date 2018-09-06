classdef AGDStep < handle
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
    
    methods (Sealed)
        function reset(opt)
            opt.m = 0;
        end
        
        function [step, predchange] = computestep(opt)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            g = grad(opt);
            momentum_ = opt.momentum;
            
            if momentum_ > 0
                opt.m = momentum_*opt.m + (1-opt.damping)*g;
                if opt.nesterov
                    step = -g - momentum_*opt.m;
                else
                    step = -opt.m;
                end
            else
                step = -g;
            end
            predchange = dot(step, g);
        end
    end
end