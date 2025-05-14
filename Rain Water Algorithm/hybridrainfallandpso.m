clear; clc; close all;

% Load features and labels
load('bestCCnetfeatureswithlables45fc8.mat');  % This file should contain 'features' and 'labels'

% Algorithm parameters
numParticles = 50;
numFeatures = size(features, 2);
numIterations = 100;
c1 = 1.5; % Cognitive coefficient
c2 = 1.5; % Social coefficient
w = 0.9; % Inertia weight, decreasing
alpha = 0.1; % Decay rate for Rainfall Algorithm
G = 9.81; % Gravity constant for Rainfall

% Initialize PSO
positions = rand(numParticles, numFeatures) > 0.5; % Initial random positions
velocities = zeros(numParticles, numFeatures); % Initial velocities
personalBest = positions; % Initial best positions are the initial positions
personalBestScores = arrayfun(@(i) fitnessFunction(features(:, positions(i, :)), labels), 1:numParticles, 'UniformOutput', false);
personalBestScores = cell2mat(personalBestScores); % Convert cell array to numeric array
[globalBestScore, bestIdx] = min(personalBestScores);
globalBest = personalBest(bestIdx, :); % Best position across all particles

% Main optimization loop
for iter = 1:numIterations
    w = w * 0.95; % Update inertia weight (optional: decrease over iterations)
    for i = 1:numParticles
        % Update velocities and positions
        r1 = rand();
        r2 = rand();
        velocities(i, :) = w * velocities(i, :) + ...
                           c1 * r1 .* (personalBest(i, :) - positions(i, :)) + ...
                           c2 * r2 .* (globalBest - positions(i, :));
        positions(i, :) = positions(i, :) + velocities(i, :) > 0.5; % Binary decision
        positions(i, :) = positions(i, :) > 0.5; % Ensure binary positions
        
        % Rainfall mechanism for position updates
        for j = 1:numFeatures
            positions(i, j) = positions(i, j) + G * (globalBest(j) - positions(i, j)) * exp(-alpha * iter) > 0.5;
        end
        
        % Evaluate fitness
        currentScore = fitnessFunction(features(:, positions(i, :)), labels);
        % Update personal best
        if currentScore < personalBestScores(i)
            personalBestScores(i) = currentScore;
            personalBest(i, :) = positions(i, :);
        end
    end
    
    % Update global best
    [currentBestScore, currentBestIdx] = min(personalBestScores);
    if currentBestScore < globalBestScore
        globalBestScore = currentBestScore;
        globalBest = personalBest(currentBestIdx, :);
    end
end

% Function to calculate fitness based on k-fold cross-validation using KNN
function score = fitnessFunction(subFeatures, subLabels)
    if isempty(subFeatures)
        score = inf; % Penalize empty feature sets
    else
        mdl = fitcknn(subFeatures, subLabels, 'NumNeighbors', 5);
        cvmdl = crossval(mdl, 'KFold', 5);
        score = kfoldLoss(cvmdl); % Lower loss is better
    end
end

% After optimization, use globalBest to select features and train/test classifiers
selectedFeatures = features(:, globalBest);
cv = cvpartition(labels, 'HoldOut', 0.3);
trainIdx = cv.training();
testIdx = cv.test();

% Evaluate with different classifiers
classifiers = {@fitcknn, @fitcecoc, @fitcnb, @fitctree};
classifierNames = {'KNN', 'SVM', 'Naive Bayes', 'Decision Tree'};

for i = 1:length(classifiers)
    model = classifiers{i}(selectedFeatures(:, trainIdx), labels(trainIdx));
    predictions = predict(model, selectedFeatures(:, testIdx));
    C = confusionmat(labels(testIdx), predictions);

    figure;
    confusionchart(C);
    title([classifierNames{i} ' Confusion Matrix']);
end
