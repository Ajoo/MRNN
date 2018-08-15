classdef BatchPersistentMetaOptimizer < handle
    %BATCHPERSISTENTOPTIMIZER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        model
        opt
        N
    end
    
    properties (Dependent)
        gamma
    end
    
    methods
        function mopt = BatchPersistentMetaOptimizer(mdl, opt, gamma0, steps_per_batch)
            %BATCHPERSISTENTOPTIMIZER Construct an instance of this class
            %   Detailed explanation goes here
            mopt.model = RegularizedModel(mdl, gamma0);
            mopt.optimizer = opt;
            mopt.N = steps_per_batch;
        end
        
        function step(opt, loss)
            
        end
        

        function set.gamma(mopt, gamma)
            mopt.model.gamma = gamma;
        end
        function gamma = get.gamma(mopt)
            gamma = mopt.model.gamma;
        end
    end
end

