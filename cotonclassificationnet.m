
%Note: this is a ligtweigt model which includes only 5.4M learnable
%parameters, as compared to resnet18 which contains 11.18 million learnable parameters.


%loading dataset
digitDatasetPath = fullfile("E:\upcoming research\Cotton leaves Dataset for diseases classification using EfficienNet Model\final results\Training");
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
%[imdsTrain,imdsValidation] = splitEachLabel(imds,0.8);


%loading testing dataset
digitDatasetPath1 = fullfile("E:\upcoming research\Cotton leaves Dataset for diseases classification using EfficienNet Model\final results\Testing");
imds1 = imageDatastore(digitDatasetPath1, ...
   'IncludeSubfolders',true,'LabelSource','foldernames');

%googlenet code
augmenter = imageDataAugmenter('RandXReflection', true);
    % Resizing all training images to [224 224] for ResNet architecture
auimdtests = augmentedImageDatastore([224 224],imds1, 'ColorPreprocessing', 'gray2rgb', 'DataAugmentation', augmenter);

 

% Determine the split up
%total_split=countEachLabel(imds)
% Number of Images
num_images=length(imds.Labels);
% Visualize random images

% Number of Images      
%num_images=length(imdsTrain.Labels);



%%K-fold Validation
% Number of folds
num_folds=5;

% Loop for each fold
for fold_idx=1:num_folds
    
    fprintf('Processing %d among %d folds \n',fold_idx,num_folds);
    
  %  Test Indices for current fold
    test_idx=fold_idx:num_folds:num_images;

    % Test cases for current fold
    imdsTest = subset(imds,test_idx);
    
    % Train indices for current fold
    train_idx=setdiff(1:length(imds.Files),test_idx);
    
    % Train cases for current fold
    imdsTrain = subset(imds,train_idx);
    
%end


%googlenet code
augmenter = imageDataAugmenter('RandXReflection', true);
    % Resizing all training images to [224 224] for ResNet architecture
    auimds = augmentedImageDatastore([224 224],imdsTrain, 'ColorPreprocessing', 'gray2rgb', 'DataAugmentation', augmenter);

% Resizing all testing images to [224 224] for ResNet architecture   
    augValidationimds = augmentedImageDatastore([224 224],imdsTest, 'ColorPreprocessing', 'gray2rgb', 'DataAugmentation', augmenter);
    
    
% Number of Images
num_images=length(imds.Labels);

%%K-fold Validation
% Number of folds
%num_folds=10;
 
 %Relu layer is replaced with leaky relu, every second block of convolutional layers are removed
lgraph = layerGraph;
tempLayers = [
    imageInputLayer([224 224 3],"Name","data","Normalization","zscore")
    convolution2dLayer([7 7],64,"Name","conv1","BiasLearnRateFactor",0,"Padding",[3 3 3 3],"Stride",[2 2])
    batchNormalizationLayer("Name","bn_conv1")
    leakyReluLayer("Name","conv1_relu")

    maxPooling2dLayer([3 3],"Name","pool1","Padding",[1 1 1 1],"Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Name","res2a_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn2a_branch2a")
    leakyReluLayer("Name","res2a_branch2a_relu")
    convolution2dLayer([3 3],64,"Name","res2a_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn2a_branch2b")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","res2a")
    leakyReluLayer("Name","res2a_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],128,"Name","res3a_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"Stride",[2 2])
    batchNormalizationLayer("Name","bn3a_branch2a")
    leakyReluLayer("Name","res3a_branch2a_relu")
    convolution2dLayer([3 3],128,"Name","res3a_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","1res3a_relu")
    leakyReluLayer("Name","bn3a_branch2b")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","res3a_branch1","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","bn3a_branch1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","res3a")
    leakyReluLayer("Name","res3a_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4a_branch1","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","123bn4a_branch1")
    leakyReluLayer("Name","bn4a_branch1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","res4a_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"Stride",[2 2])
    batchNormalizationLayer("Name","bn4a_branch2a")
    leakyReluLayer("Name","res4a_branch2a_relu")
    convolution2dLayer([3 3],256,"Name","res4a_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","bn4a_branch2b")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","res4a")
    leakyReluLayer("Name","res4a_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],512,"Name","res5a_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"Stride",[2 2])
    batchNormalizationLayer("Name","bn5a_branch2a")
    leakyReluLayer("Name","res5a_branch2a_relu")
    convolution2dLayer([3 3],512,"Name","res5a_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1])
    batchNormalizationLayer("Name","qwebn5a_branch2b")
    leakyReluLayer("Name","bn5a_branch2b")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],512,"Name","res5a_branch1","BiasLearnRateFactor",0,"Stride",[2 2])
    batchNormalizationLayer("Name","bn5a_branch1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","res5a")
    leakyReluLayer("Name","res5a_relu")];
lgraph = addLayers(lgraph,tempLayers);


tempLayers = [
    additionLayer(2,"Name","res5b")    
     globalAveragePooling2dLayer("Name","pool5")       
     fullyConnectedLayer(1096,"Name","fc7","BiasLearnRateFactor",2)
     leakyReluLayer("Name","relu7")
     batchNormalizationLayer("Name","nodjcge_14")
     dropoutLayer(0.5,"Name","drop7")
    
     fullyConnectedLayer(4,"Name","fc8","BiasLearnRateFactor",2)
     softmaxLayer("Name","prob")
     classificationLayer("Name","output")];


lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;

lgraph = connectLayers(lgraph,"pool1","res2a_branch2a");
lgraph = connectLayers(lgraph,"pool1","res2a/in2");
lgraph = connectLayers(lgraph,"bn2a_branch2b","res2a/in1");
%lgraph = connectLayers(lgraph,"res2a_relu","res2b_branch2a");
%graph = connectLayers(lgraph,"res2a_relu","res3a_branch2a/in1");
%lgraph = connectLayers(lgraph,"bn2b_branch2b","res2b/in1");
lgraph = connectLayers(lgraph,"res2a_relu","res3a_branch2a");
lgraph = connectLayers(lgraph,"res2a_relu","res3a_branch1");
lgraph = connectLayers(lgraph,"bn3a_branch1","res3a/in2");
lgraph = connectLayers(lgraph,"bn3a_branch2b","res3a/in1");
%lgraph = connectLayers(lgraph,"res3a_relu","res3b_branch2a");
%lgraph = connectLayers(lgraph,"res3a_relu","res3b/in2");
%lgraph = connectLayers(lgraph,"bn3b_branch2b","res3b/in1");
lgraph = connectLayers(lgraph,"res3a_relu","res4a_branch1");
lgraph = connectLayers(lgraph,"res3a_relu","res4a_branch2a");
lgraph = connectLayers(lgraph,"bn4a_branch1","res4a/in2");
lgraph = connectLayers(lgraph,"bn4a_branch2b","res4a/in1");
%lgraph = connectLayers(lgraph,"res4a_relu","res4b_branch2a");
%lgraph = connectLayers(lgraph,"res4a_relu","res4b/in2");
%lgraph = connectLayers(lgraph,"bn4b_branch2b","res4b/in1");
lgraph = connectLayers(lgraph,"res4a_relu","res5a_branch2a");
lgraph = connectLayers(lgraph,"res4a_relu","res5a_branch1");
lgraph = connectLayers(lgraph,"bn5a_branch2b","res5a/in1");
lgraph = connectLayers(lgraph,"bn5a_branch1","res5a/in2");
%lgraph = connectLayers(lgraph,"res5a_relu","res5b_branch2a");
lgraph = connectLayers(lgraph,"res5a_relu","res5b/in2");
lgraph = connectLayers(lgraph,"bn5a_branch1","res5b/in1");




options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',45, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augValidationimds, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');



    
    % Training 
    
   cottonclassificationnet45 = trainNetwork(auimds,lgraph,options);
    
   model_name = strcat("cottonclassificationnet45_", num2str(fold_idx), ".mat");
   save(model_name, "cottonclassificationnet45");

   [predicted_labels, scores] = classify(cottonclassificationnet45,augValidationimds);

% Actual Labels
   actual_labels=imdsTest.Labels;
% Confusion Matrix
   figure;
   C = confusionmat(actual_labels,predicted_labels);

   confusionchart(C);
%title('Confusion Matrix: TumorResNet');
% Testing and their corresponding Labels and Posterior for each Case
   [predicted_labels, scores] = classify(cottonclassificationnet45,auimdtests);
%%Performance Study
% Actual Labels
   actual_labels=imds1.Labels;
% Confusion Matrix
   figure;
   C = confusionmat(actual_labels,predicted_labels);
   cm = confusionchart(C);
   cm.Title = 'Testing Confusion Matrix';
%analyzeNetwork(net1)


end