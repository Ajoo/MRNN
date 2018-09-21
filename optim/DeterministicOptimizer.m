classdef DeterministicOptimizer < ModelOptimizer
    %ADAMOPTIMIZER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties % algorithm options
        update_params = struct();
        
        % acceptence test
        rejection_threshold = 0;
    end
    
    methods
        function opt = DeterministicOptimizer(model)
            %PCGOPTIMIZER Construct an instance of this class
            %   Detailed explanation goes here
            opt@ModelOptimizer(model);
        end
        
        function accept = update(opt, reductionratio, limited)
            fields = fieldnames(opt.update_params);
            direction = 0;
            if reductionratio < 0.25
                direction = 1;
            elseif reductionratio > 0.75 && limited
                direction = 2;
            end            
            if direction ~= 0
                for i=1:numel(fields)
                    field = fields{i};
                    factor = opt.update_params.(field).factors(direction);
                    limit = opt.update_params.(field).limits(direction);
                    opt.(field) = opt.(field) * factor;
                    if factor > 1
                        opt.(field) = min(opt.(field), limit);
                    elseif factor < 1
                        opt.(field) = max(opt.(field), limit);
                    end
                end
            end
            
            accept = (reductionratio > opt.rejection_threshold);
        end
    end
    
end