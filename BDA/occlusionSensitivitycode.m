%-------------------------------------------------------------------%
% Comprehensive script for occlusion sensitivity analysis           %
%-------------------------------------------------------------------%

clc; clear; close all;

% Load the trained model
modelPath = 'BDA_Model.mat';
load(modelPath, 'model');

% Load the image
imagePath = 'G:\upcoming research\Cotton leaves Dataset for diseases classification using EfficienNet Model\final results\BDA\fus287.jpg';
img = imread(imagePath);

% Resize and preprocess the image similarly to how the model was trained
imgResized = imresize(img, [224 224]); % Example size, adjust as necessary
imgResized = double(imgResized);       % Convert to double
imgResized = (imgResized - mean(imgResized(:))) / std(imgResized(:)); % Normalization

% Parameters for occlusion
occSize = 32; % Size of occlusion patch
stride = 16; % Stride for sliding the occlusion window

% Initialize sensitivity map
sensitivityMap = zeros(size(imgResized, 1), size(imgResized, 2));

% Slide occlusion window over the image
for row = 1:stride:size(imgResized, 1) - occSize + 1
    for col = 1:stride:size(imgResized, 2) - occSize + 1
        % Create a copy of the image
        imgOcc = imgResized;
        
        % Apply occlusion
        imgOcc(row:row+occSize-1, col:col+occSize-1) = 0;
        
        % Flatten and preprocess the occluded image (adjust this according to actual model input requirements)
        imgFeature = reshape(imgOcc, 1, []); % Flatten the image

        % Predict using the model (ensure imgFeature matches training input)
        [label, score] = predict(model, imgFeature); % Error-check for dimensionality is essential here
        
        % Store the change in probability (you might need to adjust this part depending on your output)
        sensitivityMap(row:row+occSize-1, col:col+occSize-1) = sensitivityMap(row:row+occSize-1, col:col+occSize-1) + score(1); % Assume class '1' is of interest
    end
end

% Normalize the sensitivity map
sensitivityMap = sensitivityMap / max(sensitivityMap(:));

% Display the results
figure;
subplot(1, 2, 1);
imshow(img);
title('Original Image');

subplot(1, 2, 2);
imagesc(sensitivityMap);
axis image;
title('Occlusion Sensitivity');
colorbar;

