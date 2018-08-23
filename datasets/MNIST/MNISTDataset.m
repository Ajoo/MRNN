classdef MNISTDataset < handle
    %MNISTDATASET Summary of this class goes here
    %   Detailed explanation goes here
    properties (Constant)
        TRAIN_IMAGES_FILEPATH = 'train-images-idx3-ubyte';
        TRAIN_LABELS_FILEPATH = 'train-labels-idx1-ubyte';
        TEST_IMAGES_FILEPATH = 't10k-images-idx3-ubyte';
        TEST_LABELS_FILEPATH = 't10k-labels-idx1-ubyte';
    end
    properties
        images;
        labels;
        perm = ':';
        cursor = 0;
    end
    properties (Dependent)
        size;
        epoch;
    end
    
    methods
        function h = MNISTDataset(type, permuted, shuffle)
            %MNISTDATASET Construct an instance of this class
            %   Detailed explanation goes here
            if nargin < 1 || isempty(type)
                type = 'train';
            end 
            
            switch lower(type)
                case 'train'
                    images_path = MNISTDataset.TRAIN_IMAGES_FILEPATH;
                    labels_path = MNISTDataset.TRAIN_LABELS_FILEPATH;
                case 'test'
                    images_path = MNISTDataset.TEST_IMAGES_FILEPATH;
                    labels_path = MNISTDataset.TEST_LABELS_FILEPATH;
                otherwise
                    error('Valid options for dataset type are ''train'' or ''test''')
            end
            
            images_file = IDXFile(images_path, '*');
            labels_file = IDXFile(labels_path, '*');
            h.images = read(images_file);
            h.labels = read(labels_file);
            
            if nargin >= 2 && ~isempty(permuted) && permuted
                h.perm = randperm(size(h.images, 1));
            end
            if nargin >= 3 && ~isempty(shuffle) && shuffle
                shuffle = randperm(size(h.images, 2));
            else
                shuffle = ':';
            end
            h.images = h.images(h.perm, shuffle);
            h.labels = h.labels(1, shuffle);
        end
        
        function [x, y] = get_batch(h, batchsize)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            batchidx = 1+mod(h.cursor:h.cursor+batchsize-1, h.size);
            x = double(reshape(h.images(:,batchidx).', 1, batchsize, []));
            y = onehot(10, double(h.labels(1,batchidx)));
            h.cursor = h.cursor + batchsize;
        end
        
        function e = get.epoch(h)
            e = 1 + floor(h.cursor/h.size);
        end
        function n = get.size(h)
            n = size(h.images,2);
        end
    end
end

